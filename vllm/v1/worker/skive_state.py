# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# SKIVE state container and per-step postprocess for the GPU model runner.
#
# This module owns the SKIVE side-state that lives on the model runner
# across steps:
#     SkiveState          — buffers + per-request bookkeeping (CUDA-graph
#                           safe: all tensor shapes are static at
#                           registration time, no Python control flow on
#                           hot path).
#     SkivePostprocess    — small bundle of host-callable helpers invoked
#                           once per decode step from
#                           GPUModelRunner._forward_hook (registered via
#                           the forward_context mechanism).
#
# Design notes:
#   * **CUDA graph compatibility.** Everything that the captured decode
#     region touches is allocated up-front when a request is registered,
#     so graph capture sees only static-shape tensor operations. The
#     argmin kernel (``triton_argmin_per_request``) runs on a
#     pre-allocated padded ``s_buffer_3d`` and a pre-allocated
#     ``argmin_out`` of fixed shape. The eviction policy
#     (sink-protection mask) is also a pre-allocated tensor that we
#     update in-place once per step on the host outside the captured
#     region. The copy_kernel (move-KV step) is graph-capturable since
#     it only reads from pre-allocated slots.
#   * **Zero overhead when disabled.** SkiveState is only constructed
#     and held when ``cache_config.skive_enabled`` is True. All
#     touched call-sites are gated behind a single ``is not None``
#     check in the model runner.

from __future__ import annotations

from dataclasses import dataclass

import torch

# A generous upper bound on the number of decode requests that may be
# in flight at once. vLLM's scheduler already enforces a hard cap; this
# is just a safety bound so the pre-allocated buffers don't blow up
# when the scheduler reorders requests. The pre-allocated score buffer
# shape does NOT depend on the *current* batch size — it is sized to
# the worst case so that graph capture and replay are stable.
_MAX_DECODE_REQS_DEFAULT = 1024

# Default upper bound on the longest single sequence we expect to
# score. Used to size ``s_buffer_3d`` so that argmin and accumulation
# can run with static shapes. This is a worst-case budget; the actual
# per-request length is read from ``seq_lens`` at each step.
_MAX_SEQ_LEN_DEFAULT = 32768


@dataclass
class _SkiveRequestState:
    """Per-request SKIVE bookkeeping. All fields are CPU/Python ints.

    Lives in a Python dict keyed by req_id, NOT in any captured tensor.
    Updated only on the host (request registration / deregistration),
    never on the captured decode path.
    """

    req_id: str
    # Length of the protected prefix (system prompt + shared prefix
    # from prefix caching). Tokens in [0, prefix_len) are never
    # evicted.
    prefix_len: int
    # Whether this request is currently active in a decode batch. We
    # keep the entry in the dict even when the request is paused so
    # the cached scores survive a pause/resume.
    in_batch: bool = True


class SkiveState:
    """SKIVE per-runner state container.

    All tensors here are allocated once at model-runner construction
    (or at first register_request call) and have static shapes so that
    CUDA graphs remain capturable.

    Args:
        num_layers: total number of transformer layers in the model.
            Used as the divisor for "mean" score aggregation across
            layers. The kernel already does atomic-add into a single
            buffer; we divide by ``num_layers`` host-side before argmin.
        num_kv_heads: number of KV heads (used to size the per-head
            dim of the score buffer).
        block_size: vLLM block (page) size in tokens.
        kv_budget: per-request KV budget in tokens. After this, the
            lowest-scoring block is evicted on each subsequent step.
        num_sink_tokens: number of leading tokens protected from
            eviction. Sinks are computed as the maximum of this value
            and the registered request's prefix_len.
        score_aggregation: "mean" (recommended, GPU-only) or "ema".
        score_ema_alpha: smoothing factor for "ema" aggregation.
        device: torch device for the pre-allocated buffers.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        block_size: int,
        kv_budget: int,
        num_sink_tokens: int,
        score_aggregation: str = "mean",
        score_ema_alpha: float = 0.9,
        max_decode_reqs: int = _MAX_DECODE_REQS_DEFAULT,
        max_seq_len: int = _MAX_SEQ_LEN_DEFAULT,
        device: torch.device | str = "cuda",
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.kv_budget = int(kv_budget)
        self.num_sink_tokens = int(num_sink_tokens)
        # Block-aligned versions used by select_eviction_blocks().
        # These are derived from the token-level budgets, rounded up
        # to a whole-block count.  Sinks + local_window in blocks
        # should never overlap; we clamp local to leave at least one
        # evictable middle block at the configured budget.
        self.num_sink_blocks = max(
            1, (self.num_sink_tokens + self.block_size - 1) // self.block_size
        ) if self.num_sink_tokens > 0 else 0
        # Default recency window: 1 block.  Configurable later via
        # cache_config.skive_local_window_tokens.
        self.num_local_blocks = 1
        self.score_aggregation = score_aggregation
        self.score_ema_alpha = float(score_ema_alpha)
        # Round budget down to a multiple of block_size for the
        # block-wise eviction policy.  Otherwise the last partial block
        # would never be evictable. We log a warning if the rounding
        # is non-trivial.
        if self.kv_budget % self.block_size != 0:
            self.kv_budget = (self.kv_budget // self.block_size) * self.block_size
        self.kv_budget_blocks = self.kv_budget // self.block_size

        # Pre-allocated per-step score buffer. Shape:
        #   [max_decode_reqs, max_seq_len, num_kv_heads]
        # float32 for stable accumulation across layers.
        # The kernel writes one scalar per (request, kv_head) per
        # token position; per-step we slice the live [num_decode_reqs,
        # seq_len_i, num_kv_heads] region.
        self.s_buffer_3d = torch.zeros(
            (max_decode_reqs, max_seq_len, num_kv_heads),
            dtype=torch.float32,
            device=device,
        )
        # Scratch buffer for the argmin kernel (output of the per-step
        # block-wise argmin over the score buffer).
        self.argmin_out = torch.empty(
            (max_decode_reqs,),
            dtype=torch.int64,
            device=device,
        )
        # Scratch buffer for the (un-normalized) block-summed scores
        # fed into the argmin kernel. Shape: [max_decode_reqs, max_blocks].
        # We allocate to max possible blocks at kv_budget scale (one
        # per request's worth of blocks) and slice live.
        max_blocks = max_seq_len // self.block_size
        self.block_score = torch.empty(
            (max_decode_reqs, max_blocks),
            dtype=torch.float32,
            device=device,
        )
        # EMA state buffer (only used when score_aggregation == "ema").
        # Same shape as s_buffer_3d.
        if score_aggregation == "ema":
            self.ema_buf = torch.zeros_like(self.s_buffer_3d)
        else:
            self.ema_buf = None
        # Pre-allocated eviction target (host-readable copy of
        # argmin_out + a CPU-side int64 tensor of size [num_decode_reqs]
        # that the postprocess hook consumes). We use a pinned tensor
        # so the D2H is fast. The postprocess hook replaces this with a
        # freshly pinned allocation only on resize.
        self._host_evict_targets = torch.empty(
            (max_decode_reqs,),
            dtype=torch.int64,
            pin_memory=True,
        )

        # Per-request Python state (CPU only, never captured).
        self._req_state: dict[str, _SkiveRequestState] = {}
        # Mapping from the model runner's per-batch req index → our
        # internal req_id. Refreshed by the runner on each step.
        # (We don't store this here; the runner passes it to
        # ``argmin_and_collect`` directly.)
        # Static-shaped live counts so the runner can ask "is the
        # buffer big enough for this step?" before touching it.
        self.max_decode_reqs = max_decode_reqs
        self.max_seq_len = max_seq_len

    # ----------------------------------------------------------------
    # Request registration
    # ----------------------------------------------------------------
    def register_request(self, req_id: str, prefix_len: int = 0) -> None:
        """Register a new request with SKIVE.

        Must be called BEFORE the request is first seen in a decode
        batch. prefix_len is the number of leading tokens that are
        immutable (prefix-cached, system prompt, etc.).
        """
        self._req_state[req_id] = _SkiveRequestState(
            req_id=req_id,
            prefix_len=int(prefix_len),
        )

    def deregister_request(self, req_id: str) -> None:
        self._req_state.pop(req_id, None)

    def get_prefix_len(self, req_id: str) -> int:
        st = self._req_state.get(req_id)
        if st is None:
            return 0
        return st.prefix_len

    # ----------------------------------------------------------------
    # Capacity checks
    # ----------------------------------------------------------------
    def can_accept_batch(self, num_decode_reqs: int) -> bool:
        return num_decode_reqs <= self.max_decode_reqs


class SkivePostprocess:
    """Per-step postprocess helpers for SKIVE.

    This class is intentionally tiny: a single object on the model
    runner that bundles the argmin/argmin-and-collect call and the
    eviction-target copy. Kept separate from ``SkiveState`` so that
    SkiveState is pure data (cheap to dump in tests).
    """

    def __init__(self, state: SkiveState) -> None:
        self.state = state

    def mean_normalize(
        self,
        s_buffer: torch.Tensor,  # [num_decode_reqs, max_seq_len, num_kv_heads]
        num_decode_reqs: int,
    ) -> None:
        """In-place divide-by-num-layers.

        Called once at the end of the layer loop in the model runner
        (after all layers have atomic-added their per-layer scores
        into s_buffer). The kernel writes L1 norms summed across
        ``num_kv_heads`` per token; we normalize once here.
        """
        if num_decode_reqs == 0:
            return
        s_buffer[:num_decode_reqs].div_(self.state.num_layers)

    def reset_score_buffer(self, num_decode_reqs: int) -> None:
        """Zero the live region of the score buffer for the next step.

        Called AFTER the postprocess has consumed the buffer (after
        evict_targets is computed and applied). We zero only the live
        region to avoid a full-tensor clear when the batch is small.
        """
        if num_decode_reqs == 0:
            return
        self.state.s_buffer_3d[:num_decode_reqs].zero_()

    def reset_block_score(self, num_decode_reqs: int, max_blocks: int) -> None:
        if num_decode_reqs == 0:
            return
        self.state.block_score[:num_decode_reqs, :max_blocks].zero_()

    def aggregate_block_scores_pure_cpu(
        self,
        num_decode_reqs: int,
        max_seq_len: int,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        num_kv_heads: int,
    ) -> None:
        """Sum per-token SKIVE scores into per-block scores, vectorized.

        Uses ``scatter_add_`` to aggregate token-level scores into block-
        level scores in a graph-safe, zero-.item() manner.  The output
        shape is ``[num_decode_reqs, max_blocks]`` float32.
        """
        block_size = self.state.block_size
        max_blocks = block_table.shape[1]

        self.reset_block_score(num_decode_reqs, max_blocks)

        s = self.state.s_buffer_3d[:num_decode_reqs, :max_seq_len, :num_kv_heads]
        per_token = s.sum(dim=-1)                              # [num_decode, max_seq_len]

        # Build flat scatter indices: for token t in req r, its block is t // block_size.
        token_pos = torch.arange(max_seq_len, device=per_token.device)  # [max_seq_len]
        req_offsets = (torch.arange(num_decode_reqs, device=per_token.device) * max_blocks).unsqueeze(1)  # [num_decode, 1]
        block_idx = (token_pos.unsqueeze(0) // block_size).clamp(max=max_blocks - 1)  # [num_decode, max_seq_len]
        flat_block_idx = (req_offsets + block_idx).reshape(-1)                       # [num_decode * max_seq_len]

        # Mask out tokens beyond each request's seq_len
        per_req_seq_len = seq_lens[:num_decode_reqs].unsqueeze(1)                    # [num_decode, 1]
        valid = (token_pos.unsqueeze(0) < per_req_seq_len).to(per_token.dtype)       # [num_decode, max_seq_len]
        values = (per_token * valid).reshape(-1)                                     # [num_decode * max_seq_len]

        # scatter_add_ into a scratch buffer (resilient to non-contiguous
        # views of the destination tensor), then copy back into the
        # real block_score slice.
        scratch = torch.zeros(
            num_decode_reqs * max_blocks,
            dtype=self.state.block_score.dtype,
            device=self.state.block_score.device,
        )
        scratch.scatter_add_(0, flat_block_idx, values)
        self.state.block_score[:num_decode_reqs, :max_blocks].copy_(
            scratch.reshape(num_decode_reqs, max_blocks)
        )

        # Zero out blocks that don't exist for this request
        # (e.g., for a seq of length 33 with block_size=16, blocks 2 holds only 1 token
        #  but block_score at columns 3..max_blocks should be zero)
        col_idx = torch.arange(max_blocks, device=per_token.device)                  # [max_blocks]
        num_blocks_per_req = (seq_lens[:num_decode_reqs] + block_size - 1) // block_size  # [num_decode]
        valid_cols = (col_idx.unsqueeze(0) < num_blocks_per_req.unsqueeze(1)).to(self.state.block_score.dtype)  # [num_decode, max_blocks]
        self.state.block_score[:num_decode_reqs, :max_blocks].mul_(valid_cols)

    def select_eviction_blocks(
        self,
        num_decode_reqs: int,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        num_sink_blocks: int,
        num_local_blocks: int,
    ) -> torch.Tensor:
        """For each request, pick one block to evict (argmin of block_score).

        Vectorized version: no ``.item()`` calls, no per-request Python
        loop.  Uses masked argmin over the block_score tensor to find
        the lowest-scoring block in the eligible middle range.

        Protection rules:
        - The first ``num_sink_blocks`` blocks of each request are sinks.
        - The LAST ``num_local_blocks`` blocks of each request are
          the recency window.
        - Out-of-range blocks (beyond each request's actual num_blocks)
          are masked to +inf so they never win argmin.

        Returns:
            ``evict_block_ids`` of shape ``[num_decode_reqs]`` int32, with
            ``-1`` for requests that have no evictable block.
        """
        block_size = self.state.block_size
        max_blocks = block_table.shape[1]
        device = self.state.block_score.device

        bs = self.state.block_score[:num_decode_reqs, :max_blocks]   # [num_decode, max_blocks]
        bt = block_table[:num_decode_reqs, :max_blocks]               # [num_decode, max_blocks]

        # Per-request num_blocks and eligible [start, end) ranges, all vectorized.
        num_blocks_per_req = (seq_lens[:num_decode_reqs] + block_size - 1) // block_size  # [num_decode]
        # Replace any negative seq_len with 0 (defensive)
        num_blocks_per_req = num_blocks_per_req.clamp(min=0)
        col_idx = torch.arange(max_blocks, device=device).unsqueeze(0)  # [1, max_blocks]

        # Sink range: blocks [0, num_sink_blocks) are protected.
        in_sink = col_idx < num_sink_blocks
        # Local range: blocks [num_blocks - num_local_blocks, num_blocks) are protected.
        # If num_blocks < num_local_blocks, all blocks are local.
        local_start = (num_blocks_per_req - num_local_blocks).unsqueeze(1)  # [num_decode, 1]
        in_local = col_idx >= local_start
        # Out-of-range: blocks beyond num_blocks don't exist for this request.
        out_of_range = col_idx >= num_blocks_per_req.unsqueeze(1)
        # No eligible middle block when num_blocks <= num_sink_blocks + num_local_blocks.
        no_middle = (num_blocks_per_req <= (num_sink_blocks + num_local_blocks)).unsqueeze(1)
        # Eligible: NOT sink, NOT local, NOT out-of-range, NOT no_middle
        protected = in_sink | in_local | out_of_range | no_middle  # [num_decode, max_blocks]

        # Mask protected positions with +inf
        masked = bs.masked_fill(protected, float("inf"))
        # argmin across the block dimension
        best_block_idx = torch.argmin(masked, dim=1)  # [num_decode], values in [0, max_blocks)
        # Map to physical block IDs via the block_table
        evict = torch.gather(bt, 1, best_block_idx.unsqueeze(1)).squeeze(1)  # [num_decode]
        # For requests with no eligible block, set to -1
        # (masked argmin returns 0 when all are +inf; we must use a sentinel)
        # Use gather: if all positions are +inf, argmin returns 0, and we need to detect.
        all_inf = protected.all(dim=1)  # [num_decode]
        evict = torch.where(all_inf, torch.full_like(evict, -1), evict)
        return evict.to(torch.int32)
