"""SKIVE fused-decode attention + L1-of-contribution scoring kernel.

This module provides ``skive_paged_decode_attention``, a single Triton
kernel that:

1. Computes standard paged-decode attention output for one query per
   request, against a paged KV cache in vLLM V1 layout
   ``[num_blocks, num_kv_heads, head_size, block_size]``.
2. Computes per-token L1-of-contribution scores
   ``S_i = ||p_i * v_i||_1`` (the SKIVE importance metric) and
   atomic-adds them into a per-request score buffer.

The score buffer shape is ``[num_decode_reqs, max_seq_len,
num_kv_heads]``, dtype ``float32``.  Each call to this kernel
contributes one layer's scores; the caller (typically
``FlashAttentionImpl.forward``) is invoked once per decoder layer, and
the same buffer is reused across layers.

The kernel is designed to be a drop-in replacement for the decode
branch of vLLM's FlashAttention path.  When SKIVE is disabled, this
module is never imported (see ``flash_attn.py``'s guarded import).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - CPU-only test environments
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def _skive_paged_decode_attention_fallback(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    scale: float,
    score_buf: torch.Tensor | None,
    out: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
) -> None:
    """Pure-PyTorch fallback for the SKIVE fused-decode kernel.

    Used when Triton is unavailable (e.g. during CPU-only tests).  It
    performs the same computation as the Triton kernel but is
    dramatically slower; it exists only so the call site does not
    crash in environments without a GPU/Triton.

    Args:
        q: ``[num_decode_reqs, num_heads * head_size]`` (one query per
            request, flattened).
        key_cache, value_cache: paged KV cache in vLLM V1 layout
            ``[num_blocks, num_kv_heads, head_size, block_size]``.
        block_table: ``[num_decode_reqs, max_blocks_per_seq]`` int32.
        seq_lens: ``[num_decode_reqs]`` int32.
        max_seq_len: maximum sequence length in this decode batch.
        scale: softmax scale.
        score_buf: ``[num_decode_reqs, max_seq_len, num_kv_heads]``
            float32 buffer; the kernel atomic-adds scores here.  May be
            ``None`` for testing in which case scores are not written.
        out: preallocated output of shape ``[num_decode_reqs,
            num_heads * head_size]``.
        num_kv_heads: number of KV heads (GQA-aware).
        head_size: per-head dimension.
    """
    num_decode_reqs = q.shape[0]
    block_size = key_cache.shape[-1]
    num_heads = q.shape[1] // head_size
    groups = num_heads // num_kv_heads

    # For each request, gather its KV cache pages and compute attention
    # + scores.  Loop over requests because the per-request seq_len
    # is dynamic.
    for req_idx in range(num_decode_reqs):
        seq_len = int(seq_lens[req_idx].item())
        if seq_len <= 0:
            continue

        # Gather the request's KV pages into dense tensors.
        num_blocks = (seq_len + block_size - 1) // block_size
        pages = block_table[req_idx, :num_blocks].to(torch.long)

        # Shape: [seq_len, num_kv_heads, head_size]
        k_req = torch.zeros(
            seq_len, num_kv_heads, head_size,
            dtype=key_cache.dtype, device=key_cache.device,
        )
        v_req = torch.zeros_like(k_req)
        for tok in range(seq_len):
            blk_idx = tok // block_size
            tok_in_blk = tok % block_size
            phys_block = pages[blk_idx]
            k_req[tok] = key_cache[phys_block, :, :, tok_in_blk]
            v_req[tok] = value_cache[phys_block, :, :, tok_in_blk]

        # GQA expansion: replicate KV heads for query groups.
        # k_req shape: [seq_len, num_kv_heads, head_size]
        # We want: [seq_len, num_heads, head_size] where num_heads = num_kv_heads * groups
        k_exp = k_req.unsqueeze(1).expand(-1, groups, -1, -1)
        k_exp = k_exp.reshape(seq_len, num_heads, head_size)
        v_exp = v_req.unsqueeze(1).expand(-1, groups, -1, -1)
        v_exp = v_exp.reshape(seq_len, num_heads, head_size)

        # Attention logits and softmax.
        q_req = q[req_idx].view(num_heads, head_size)
        logits = torch.einsum("hd,thd->ht", q_req.float(),
                              k_exp.float()) * scale
        probs = torch.softmax(logits, dim=-1)  # [num_heads, seq_len]

        # Attention output.
        out_req = torch.einsum("ht,thd->hd", probs, v_exp.float())
        out[req_idx] = out_req.view(-1).to(out.dtype)

        # SKIVE scores: L1-of-contribution, per (kv_head, token).
        if score_buf is not None:
            # probs: [num_heads, seq_len], v_exp: [seq_len, num_heads, head_size]
            # contribution_i = p_i * v_i, shape: [seq_len, num_heads, head_size]
            contribution = (
                probs.unsqueeze(-1).permute(1, 0, 2) * v_exp
            )  # [seq_len, num_heads, head_size]
            # Average across query groups (GQA: same contribution for
            # all queries within a group).
            contribution = contribution.view(
                seq_len, num_kv_heads, groups, head_size
            ).mean(dim=2)  # [seq_len, num_kv_heads, head_size]
            l1 = contribution.abs().sum(dim=-1)  # [seq_len, num_kv_heads]
            # Atomic-add into the buffer (shape [max_seq_len, num_kv_heads]).
            score_buf[req_idx, :seq_len, :] += l1.to(score_buf.dtype)


if _TRITON_AVAILABLE:

    @triton.jit
    def _skive_paged_decode_kernel(
        Q_ptr,              # [num_decode, num_heads * head_size]
        K_ptr,              # [num_blocks, num_kv_heads, head_size, block_size]
        V_ptr,              # same as K
        BlockTable_ptr,     # [num_decode, max_blocks]
        SeqLens_ptr,        # [num_decode]
        ScoreBuf_ptr,       # [num_decode, max_seq_len, num_kv_heads]
        Out_ptr,            # [num_decode, num_heads * head_size]
        # strides (element-level)
        stride_qd, stride_qh,
        stride_kb, stride_kh, stride_ks, stride_kt,
        stride_vb, stride_vh, stride_vs, stride_vt,
        stride_btd, stride_btb,
        stride_sd, stride_sl, stride_sh,
        stride_od, stride_oh,
        # constants
        SCALE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_SIZE: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        GROUPS: tl.constexpr,         # num_heads // num_kv_heads
        MAX_SEQ_LEN: tl.constexpr,
    ):
        """One program instance per (request, kv_head).

        Loops over the request's KV blocks, computes attention logits
        against the query, applies online softmax, accumulates the
        output, and atomic-adds the L1-of-contribution score for each
        token into ``ScoreBuf_ptr``.

        For GQA, ``GROUPS > 1``, the same scores are valid for all
        queries within a group; we compute the score once per
        ``(request, kv_head)`` pair.
        """
        req_id = tl.program_id(0)
        kv_head = tl.program_id(1)

        seq_len = tl.load(SeqLens_ptr + req_id)

        # Load the query for this request's first query head that
        # maps to this kv_head.  In GQA, queries for head h use kv_head
        # h // GROUPS.
        q_head = kv_head * GROUPS
        q_offset = q_head * HEAD_SIZE
        q = tl.load(
            Q_ptr + req_id * stride_qd + q_offset + tl.arange(0, HEAD_SIZE)
        )  # [HEAD_SIZE]

        # Online softmax accumulators.
        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([HEAD_SIZE], dtype=tl.float32)

        # Loop over KV blocks.
        num_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
        for blk in range(num_blocks):
            phys_block = tl.load(
                BlockTable_ptr + req_id * stride_btd + blk * stride_btb
            )
            offs_t = tl.arange(0, BLOCK_SIZE)
            valid = blk * BLOCK_SIZE + offs_t < seq_len

            # Load K and V tiles for this block.
            k_offs = (
                phys_block * stride_kb
                + kv_head * stride_kh
                + tl.arange(0, HEAD_SIZE)[:, None] * stride_ks
                + offs_t[None, :] * stride_kt
            )
            v_offs = (
                phys_block * stride_vb
                + kv_head * stride_vh
                + offs_t[:, None] * stride_vs
                + tl.arange(0, HEAD_SIZE)[None, :] * stride_vt
            )
            k_tile = tl.load(K_ptr + k_offs, mask=valid[None, :],
                             other=0.0)  # [HEAD_SIZE, BLOCK_SIZE]
            v_tile = tl.load(V_ptr + v_offs, mask=valid[:, None],
                             other=0.0)  # [BLOCK_SIZE, HEAD_SIZE]

            # Attention logits: q @ k_tile  ->  [BLOCK_SIZE]
            s = tl.sum(q[:, None].to(tl.float32) * k_tile.to(tl.float32),
                       axis=0) * SCALE
            s = tl.where(valid, s, -float("inf"))

            # Online softmax update.
            m_new = tl.maximum(m_i, tl.max(s, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)  # [BLOCK_SIZE]
            l_i = l_i * alpha + tl.sum(p, axis=0)
            # Accumulate weighted V (cast to fp32 for the add).
            acc = acc * alpha + tl.sum(
                p[:, None] * v_tile.to(tl.float32), axis=0
            )
            m_i = m_new

            # ---- SKIVE score write ----
            # contribution_i = p_i * v_i, L1 norm = sum of |contribution_i|
            # Skip if no GQA expansion (compute once per kv_head).
            contrib = p[:, None] * v_tile.to(tl.float32)  # [BLOCK, HEAD]
            l1 = tl.sum(tl.abs(contrib), axis=1)  # [BLOCK]

            # Compute global token positions for this block.
            tok_positions = blk * BLOCK_SIZE + offs_t
            # Atomic-add into ScoreBuf[req_id, tok_pos, kv_head].
            sb_offs = (
                req_id * stride_sd
                + tok_positions * stride_sl
                + kv_head * stride_sh
            )
            tl.atomic_add(ScoreBuf_ptr + sb_offs, l1,
                          mask=valid)

        # Final attention output.
        out = (acc / l_i).to(Out_ptr.dtype.element_ty)
        out_offs = (
            req_id * stride_od
            + (q_head + tl.arange(0, GROUPS))[:, None] * stride_oh
            + tl.arange(0, HEAD_SIZE)[None, :]
        )
        # Broadcast the computed output across the GROUPS query heads.
        # out has shape [HEAD_SIZE]; replicate across GROUPS rows.
        out_2d = tl.broadcast_to(
            out[None, :], (GROUPS, HEAD_SIZE)
        )
        tl.store(Out_ptr + out_offs, out_2d)

else:
    _skive_paged_decode_kernel = None  # type: ignore[assignment]


def skive_paged_decode_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    scale: float,
    score_buf: torch.Tensor | None,
    out: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
) -> None:
    """Fused-decode attention + L1-of-contribution scoring.

    See module docstring for details.

    Args:
        q: ``[num_decode_reqs, num_heads * head_size]``
        key_cache, value_cache: ``[num_blocks, num_kv_heads,
            head_size, block_size]``
        block_table: ``[num_decode_reqs, max_blocks]`` int32
        seq_lens: ``[num_decode_reqs]`` int32
        max_seq_len: int
        scale: float
        score_buf: ``[num_decode_reqs, max_seq_len, num_kv_heads]``
            float32, or ``None`` (scores not written).
        out: ``[num_decode_reqs, num_heads * head_size]``
        num_kv_heads: int
        head_size: int
    """
    if not _TRITON_AVAILABLE or _skive_paged_decode_kernel is None:
        _skive_paged_decode_attention_fallback(
            q=q,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            scale=scale,
            score_buf=score_buf,
            out=out,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
        return

    num_decode_reqs = q.shape[0]
    num_heads = q.shape[1] // head_size
    groups = num_heads // num_kv_heads
    block_size = key_cache.shape[-1]

    grid = (num_decode_reqs, num_kv_heads)

    _skive_paged_decode_kernel[grid](
        q,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        score_buf if score_buf is not None else q,  # dummy if None
        out,
        q.stride(0), q.stride(1),
        key_cache.stride(0), key_cache.stride(1),
        key_cache.stride(2), key_cache.stride(3),
        value_cache.stride(0), value_cache.stride(1),
        value_cache.stride(2), value_cache.stride(3),
        block_table.stride(0), block_table.stride(1),
        (score_buf.stride(0) if score_buf is not None else 1),
        (score_buf.stride(1) if score_buf is not None else 1),
        (score_buf.stride(2) if score_buf is not None else 1),
        out.stride(0), out.stride(1),
        SCALE=scale,
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        NUM_KV_HEADS=num_kv_heads,
        GROUPS=groups,
        MAX_SEQ_LEN=max_seq_len,
    )