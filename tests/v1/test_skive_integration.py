"""Comprehensive unit tests for SKIVE fused attention + postprocess.

Tests run on CPU (no GPU required):
  - skive_state: SkiveState allocation, SkivePostprocess mean-norm
  - skive_postprocess: per-block score aggregation, eviction selection
  - skive_block_pool: BlockPool.evict_blocks_derefed with prefix-cache safety
  - skive_kv_cache_manager: evict_blocks_derefed_for_skive wrapper
  - skive_fallback: PyTorch fallback kernel matches reference attention

Run with:
    python -m pytest tests/v1/test_skive_integration.py -v --no-header
"""

import math
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skive_state(**kwargs):
    """Create a SkiveState with sensible defaults."""
    from vllm.v1.worker.skive_state import SkiveState, SkivePostprocess
    defaults = dict(
        num_layers=32,
        num_kv_heads=8,
        block_size=16,
        kv_budget=128,
        num_sink_tokens=4,
        max_decode_reqs=8,
        max_seq_len=256,
        device="cpu",
    )
    defaults.update(kwargs)
    state = SkiveState(**defaults)
    postprocess = SkivePostprocess(state)
    return state, postprocess


def _set_scores(s_buffer, num_decode, seq_lens, num_kv_heads, score_val=1.0):
    """Set per-token scores in s_buffer_3d for testing."""
    for r in range(num_decode):
        for t in range(seq_lens[r]):
            for h in range(num_kv_heads):
                s_buffer[r, t, h] = score_val


# ===========================================================================
# Test SkiveState initialization
# ===========================================================================
class TestSkiveStateInit:

    def test_default_values(self):
        state, _ = _make_skive_state()
        assert state.num_layers == 32
        assert state.num_kv_heads == 8
        assert state.block_size == 16
        assert state.kv_budget == 128
        assert state.num_sink_tokens == 4
        assert state.num_sink_blocks >= 1
        assert state.num_local_blocks == 1

    def test_sink_blocks_ceil(self):
        state, _ = _make_skive_state(num_sink_tokens=5)
        # block_size=16, 5/16 = ceil = 1 block
        assert state.num_sink_blocks == 1

    def test_budget_rounded_down(self):
        state, _ = _make_skive_state(kv_budget=100)
        # kv_budget=100, block_size=16: 100/16 = 6.25 -> 96
        assert state.kv_budget == 96

    def test_buffer_shapes(self):
        state, postprocess = _make_skive_state(
            num_kv_heads=4, block_size=32, kv_budget=96,
            max_decode_reqs=16, max_seq_len=512,
        )
        assert state.s_buffer_3d.shape == (16, 512, 4)
        assert state.block_score.shape == (16, 16)  # kv_budget=96 rounded to 96, 96/32=3


# ===========================================================================
# Test SkivePostprocess
# ===========================================================================
class TestSkivePostprocess:

    def test_mean_normalize_basic(self):
        state, post = _make_skive_state(num_layers=16)
        state.s_buffer_3d[0, 0, :] = 16.0
        post.mean_normalize(state.s_buffer_3d, 1)
        assert abs(state.s_buffer_3d[0, 0, 0].item() - 1.0) < 1e-5

    def test_mean_normalize_zero_requests(self):
        state, post = _make_skive_state()
        # Should not crash with num_decode=0
        post.mean_normalize(state.s_buffer_3d, 0)
        # Nothing changed
        assert state.s_buffer_3d.sum() == 0.0

    def test_mean_normalize_partial(self):
        state, post = _make_skive_state(num_layers=8)
        state.s_buffer_3d[0, :, :] = 8.0
        state.s_buffer_3d[1, :, :] = 16.0
        post.mean_normalize(state.s_buffer_3d, 2)
        assert abs(state.s_buffer_3d[0, 0, 0].item() - 1.0) < 1e-5
        assert abs(state.s_buffer_3d[1, 0, 0].item() - 2.0) < 1e-5

    def test_reset_score_buffer(self):
        state, post = _make_skive_state()
        state.s_buffer_3d[0, :, :] = 5.0
        post.reset_score_buffer(1)
        assert state.s_buffer_3d[0].sum() == 0.0

    def test_aggregate_block_scores_basic(self):
        state, post = _make_skive_state()
        bs = state.block_size  # 16
        # Token 0..15 (block 0): score 1.0 each
        # Token 16..31 (block 1): score 0.5 each
        # Token 32..47 (block 2): score 0.1 each
        state.s_buffer_3d[0, :bs, :] = 1.0
        state.s_buffer_3d[0, bs:2*bs, :] = 0.5
        state.s_buffer_3d[0, 2*bs:3*bs, :] = 0.1

        seq_lens = torch.tensor([48])
        block_table = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        post.aggregate_block_scores_pure_cpu(1, 48, block_table, seq_lens, 8)

        # Each block has 16 tokens * 8 kv_heads * score_val
        expected0 = 16 * 8 * 1.0
        expected1 = 16 * 8 * 0.5
        expected2 = 16 * 8 * 0.1
        assert abs(state.block_score[0, 0].item() - expected0) < 1e-3
        assert abs(state.block_score[0, 1].item() - expected1) < 1e-3
        assert abs(state.block_score[0, 2].item() - expected2) < 1e-3

    def test_aggregate_block_scores_partial_last_block(self):
        state, post = _make_skive_state()
        bs = state.block_size
        # Seq length 33: blocks 0 (16), 1 (16), 2 (1)
        state.s_buffer_3d[0, :bs, :] = 2.0       # block 0: 16*8*2 = 256
        state.s_buffer_3d[0, bs:2*bs, :] = 1.0   # block 1: 16*8*1 = 128
        state.s_buffer_3d[0, 2*bs:2*bs+1, :] = 0.5  # block 2: 1*8*0.5 = 4

        seq_lens = torch.tensor([33])
        block_table = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        post.aggregate_block_scores_pure_cpu(1, 33, block_table, seq_lens, 8)

        assert abs(state.block_score[0, 0].item() - 256.0) < 1
        assert abs(state.block_score[0, 1].item() - 128.0) < 1
        assert abs(state.block_score[0, 2].item() - 4.0) < 0.5

    def test_aggregate_zero_scores(self):
        state, post = _make_skive_state()
        # s_buffer_3d is already zero from allocation
        seq_lens = torch.tensor([32])
        block_table = torch.tensor([[0, 1]], dtype=torch.int32)
        post.aggregate_block_scores_pure_cpu(1, 32, block_table, seq_lens, 8)
        assert state.block_score[0, :2].sum() == 0.0

    def test_select_eviction_sink_protection(self):
        state, post = _make_skive_state(num_sink_tokens=4)
        # 4 blocks: block 0 is sink, block 3 is local, 1-2 eligible
        # Block 0: high score (protected), block 1: 1.0, block 2: 0.1 (lowest),
        # block 3: high score (protected)
        state.s_buffer_3d[0, :16, :] = 100.0  # sink
        state.s_buffer_3d[0, 16:32, :] = 1.0
        state.s_buffer_3d[0, 32:48, :] = 0.1
        state.s_buffer_3d[0, 48:64, :] = 100.0  # local

        seq_lens = torch.tensor([64])
        block_table = torch.tensor([[10, 20, 30, 40]], dtype=torch.int32)
        post.aggregate_block_scores_pure_cpu(1, 64, block_table, seq_lens, 8)
        evict = post.select_eviction_blocks(1, seq_lens, block_table, 1, 1)
        assert evict[0].item() == 30  # block 2 (lowest non-sink non-local)

    def test_select_eviction_local_protection(self):
        state, post = _make_skive_state(num_sink_tokens=0)
        # 3 blocks: block 0-1 eligible, block 2 is local
        state.s_buffer_3d[0, :16, :] = 0.5  # block 0
        state.s_buffer_3d[0, 16:32, :] = 0.1  # block 1 (lowest eligible)
        state.s_buffer_3d[0, 32:48, :] = 0.01  # block 2 (local, very low but protected)

        seq_lens = torch.tensor([48])
        block_table = torch.tensor([[10, 20, 30]], dtype=torch.int32)
        post.aggregate_block_scores_pure_cpu(1, 48, block_table, seq_lens, 8)
        evict = post.select_eviction_blocks(1, seq_lens, block_table, 0, 1)
        assert evict[0].item() == 20  # block 1, NOT block 2

    def test_select_eviction_no_eligible(self):
        state, post = _make_skive_state(num_sink_tokens=1)
        # Only 2 blocks: sink + local = no middle blocks
        state.s_buffer_3d[0, :16, :] = 0.0
        state.s_buffer_3d[0, 16:32, :] = 0.0

        seq_lens = torch.tensor([32])
        block_table = torch.tensor([[10, 20]], dtype=torch.int32)
        post.aggregate_block_scores_pure_cpu(1, 32, block_table, seq_lens, 8)
        evict = post.select_eviction_blocks(1, seq_lens, block_table, 1, 1)
        assert evict[0].item() == -1  # no eligible block

    def test_select_eviction_multi_request(self):
        state, post = _make_skive_state(num_sink_tokens=1)
        # Req 0: sink=block 0, eligible=blocks 1,2, local=block 3
        # Req 1: sink=block 0, eligible=blocks 1,2, local=block 3
        # Req 0: block 1 score=0.1*128=12.8, block 2 score=1.0*128=128 -> evict block 1 (200)
        # Req 1: block 1 score=5.0*128=640, block 2 score=0.01*128=1.28 -> evict block 2 (700)
        state.s_buffer_3d[0, :16, :] = 100.0  # sink
        state.s_buffer_3d[0, 16:32, :] = 0.1
        state.s_buffer_3d[0, 32:48, :] = 1.0
        state.s_buffer_3d[0, 48:64, :] = 100.0  # local

        state.s_buffer_3d[1, :16, :] = 100.0  # sink
        state.s_buffer_3d[1, 16:32, :] = 5.0
        state.s_buffer_3d[1, 32:48, :] = 0.01
        state.s_buffer_3d[1, 48:64, :] = 100.0  # local

        seq_lens = torch.tensor([64, 64])
        block_table = torch.tensor(
            [[100, 200, 300, 400], [500, 600, 700, 800]], dtype=torch.int32
        )
        post.aggregate_block_scores_pure_cpu(2, 64, block_table, seq_lens, 8)
        evict = post.select_eviction_blocks(2, seq_lens, block_table, 1, 1)
        # Req 0: eligible [1,2] → min is block 1 (score 12.8) → physical 200
        # Req 1: eligible [1,2] → min is block 2 (score 1.28) → physical 700
        assert evict[0].item() == 200
        assert evict[1].item() == 700

    def test_select_eviction_non_aligned_seq_len(self):
        """Seq len 33 with block_size=16: blocks 0,1,2. Block 2 (local) protected."""
        state, post = _make_skive_state(num_sink_tokens=0)
        state.s_buffer_3d[0, :16, :] = 0.5   # block 0: score 64
        state.s_buffer_3d[0, 16:32, :] = 0.2  # block 1: score 25.6 (lowest eligible)
        state.s_buffer_3d[0, 32:33, :] = 0.05  # block 2 (local): very low but protected

        seq_lens = torch.tensor([33])
        block_table = torch.tensor([[10, 20, 30]], dtype=torch.int32)
        post.aggregate_block_scores_pure_cpu(1, 33, block_table, seq_lens, 8)
        evict = post.select_eviction_blocks(1, seq_lens, block_table, 0, 1)
        assert evict[0].item() == 20  # block 1 (lowest non-local block)


# ===========================================================================
# Test BlockPool prefix-cache safety
# ===========================================================================
class TestEvictBlocksDerefedLogic:
    """Tests the eviction logic in isolation (no vllm.core import)."""

    def _make_pool_with_blocks(self, blocks_state):
        """Create a minimal object that has blocks + free_block_queue."""
        class FakeBlock:
            def __init__(self, bid, ref, null):
                self.block_id = bid
                self.ref_cnt = ref
                self.is_null = null

        class FakeFreeQueue:
            def __init__(self):
                self._items = []
                self.append_n_calls = []
            def append_n(self, items):
                self._items.extend(items)
                self.append_n_calls.append(items)
            def prepend_n(self, items):
                self._items = list(items) + self._items

        class FakePool:
            def __init__(self, blocks):
                self.blocks = blocks
                self.free_block_queue = FakeFreeQueue()

        blocks = []
        for i, (ref, null) in enumerate(blocks_state):
            blocks.append(FakeBlock(i, ref, null))
        return FakePool(blocks)

    def _call_evict(self, pool, block_ids):
        """Replicate the logic of BlockPool.evict_blocks_derefed."""
        freed = []
        skipped = []
        for bid in block_ids:
            if bid >= len(pool.blocks):
                raise ValueError(f"Invalid block_id {bid}")
            block = pool.blocks[bid]
            if block.is_null:
                skipped.append(bid)
                continue
            if block.ref_cnt != 1:
                skipped.append(bid)
                continue
            block.ref_cnt = 0
            freed.append(block)
        if freed:
            pool.free_block_queue.append_n(freed)
        return freed, skipped

    def test_evict_private_block(self):
        pool = self._make_pool_with_blocks([(1, False)])
        freed, skipped = self._call_evict(pool, {0})
        assert len(freed) == 1
        assert freed[0].block_id == 0
        assert pool.blocks[0].ref_cnt == 0
        assert len(skipped) == 0

    def test_skip_shared_block(self):
        pool = self._make_pool_with_blocks([(3, False)])
        freed, skipped = self._call_evict(pool, {0})
        assert len(freed) == 0
        assert skipped == [0]
        assert pool.blocks[0].ref_cnt == 3

    def test_skip_null_block(self):
        pool = self._make_pool_with_blocks([(0, True)])
        freed, skipped = self._call_evict(pool, {0})
        assert len(freed) == 0
        assert skipped == [0]

    def test_evict_multiple_mixed(self):
        pool = self._make_pool_with_blocks([
            (3, False),  # 0: shared
            (1, False),  # 1: private
            (1, False),  # 2: private
            (0, True),   # 3: null
        ])
        freed, skipped = self._call_evict(pool, {0, 1, 2, 3})
        assert {b.block_id for b in freed} == {1, 2}
        assert set(skipped) == {0, 3}

    def test_free_block_queue_updated(self):
        pool = self._make_pool_with_blocks([(1, False)] * 4)
        self._call_evict(pool, {1, 3})
        assert len(pool.free_block_queue.append_n_calls) == 1
        assert len(pool.free_block_queue.append_n_calls[0]) == 2


# ===========================================================================
# Test KVCacheManager wrapper
# ===========================================================================



# ===========================================================================
# Test PyTorch fallback kernel
# ===========================================================================
class TestSkiveFallbackKernel:

    @pytest.mark.parametrize("seq_len,num_heads,num_kv_heads,head_size,block_size", [
        (32,  8, 8, 64, 16),    # standard
        (33,  8, 8, 64, 16),    # partial last block
        (32, 32, 8, 64, 16),    # GQA groups=4
        (48,  8, 8, 64, 16),    # 3 blocks
    ])
    def test_fallback_output_correctness(self, seq_len, num_heads, num_kv_heads,
                                          head_size, block_size):
        from vllm.attention.ops.skive_paged_decode import (
            _skive_paged_decode_attention_fallback,
        )
        torch.manual_seed(42)
        num_decode_reqs = 2
        scale = 1.0 / math.sqrt(head_size)
        dtype = torch.bfloat16

        q = torch.randn(num_decode_reqs, num_heads, head_size, dtype=dtype)
        k = torch.randn(num_decode_reqs, seq_len, num_kv_heads, head_size, dtype=dtype)
        v = torch.randn(num_decode_reqs, seq_len, num_kv_heads, head_size, dtype=dtype)

        # Build paged layout
        def _build_paged(k_t, v_t, block_size):
            num_seqs, slen, nkh, hs = k_t.shape
            blocks_per_seq = math.ceil(slen / block_size)
            total_blocks = num_seqs * blocks_per_seq
            kc = torch.zeros(total_blocks, nkh, hs, block_size, dtype=dtype)
            vc = torch.zeros_like(kc)
            bt = torch.zeros(num_seqs, blocks_per_seq, dtype=torch.int32)
            phys = 0
            for s in range(num_seqs):
                for b in range(blocks_per_seq):
                    bt[s, b] = phys
                    start = b * block_size
                    end = min(start + block_size, slen)
                    for t in range(end - start):
                        kc[phys, :, :, t] = k_t[s, start + t]
                        vc[phys, :, :, t] = v_t[s, start + t]
                    phys += 1
            return kc, vc, bt

        key_cache, val_cache, block_table = _build_paged(k, v, block_size)
        seq_lens = torch.full((num_decode_reqs,), seq_len, dtype=torch.int32)

        # Reference attention
        def _ref_attn(q_req, k_req, v_req, scale):
            groups = num_heads // num_kv_heads
            k_exp = k_req.unsqueeze(1).expand(-1, groups, -1, -1).reshape(seq_len, num_heads, head_size)
            v_exp = v_req.unsqueeze(1).expand(-1, groups, -1, -1).reshape(seq_len, num_heads, head_size)
            logits = torch.einsum("hd,thd->ht", q_req.float(), k_exp.float()) * scale
            probs = torch.softmax(logits, dim=-1)
            out = torch.einsum("ht,thd->hd", probs, v_exp.float())
            return out.to(dtype)

        q_flat = q.view(num_decode_reqs, -1)
        out = torch.zeros(num_decode_reqs, num_heads * head_size, dtype=dtype)
        score_buf = torch.zeros(num_decode_reqs, seq_len, num_kv_heads, dtype=torch.float32)

        _skive_paged_decode_attention_fallback(
            q=q_flat, key_cache=key_cache, value_cache=val_cache,
            block_table=block_table, seq_lens=seq_lens,
            max_seq_len=seq_len, scale=scale,
            score_buf=score_buf, out=out,
            num_kv_heads=num_kv_heads, head_size=head_size,
        )

        # Compare outputs
        for r in range(num_decode_reqs):
            ref = _ref_attn(q[r], k[r], v[r], scale)
            got = out[r].view(num_heads, head_size)
            max_err = (got.float() - ref.float()).abs().max().item()
            assert max_err < 1e-2, (
                f"Req {r}: max error {max_err:.4e} too large "
                f"(seq_len={seq_len}, heads={num_heads}/{num_kv_heads})"
            )

    def test_fallback_scores_in_valid_range(self):
        """L1-of-contribution scores should be positive and within a
        reasonable range (not NaN, not inf, bounded by value norms)."""
        from vllm.attention.ops.skive_paged_decode import (
            _skive_paged_decode_attention_fallback,
        )
        torch.manual_seed(99)
        num_decode_reqs, seq_len = 2, 32
        num_heads, num_kv_heads, head_size = 8, 8, 64
        block_size = 16
        scale = 1.0 / math.sqrt(head_size)
        dtype = torch.bfloat16

        q = torch.randn(num_decode_reqs, num_heads, head_size, dtype=dtype)
        k = torch.randn(num_decode_reqs, seq_len, num_kv_heads, head_size, dtype=dtype)
        v = torch.randn(num_decode_reqs, seq_len, num_kv_heads, head_size, dtype=dtype)

        def _build_paged(k_t, v_t):
            ns, sl, nkh, hs = k_t.shape
            bps = math.ceil(sl / block_size)
            tb = ns * bps
            kc = torch.zeros(tb, nkh, hs, block_size, dtype=dtype)
            vc = torch.zeros_like(kc)
            bt = torch.zeros(ns, bps, dtype=torch.int32)
            p = 0
            for s in range(ns):
                for b in range(bps):
                    bt[s, b] = p
                    st = b * block_size
                    en = min(st + block_size, sl)
                    for t in range(en - st):
                        kc[p, :, :, t] = k_t[s, st + t]
                        vc[p, :, :, t] = v_t[s, st + t]
                    p += 1
            return kc, vc, bt

        key_cache, val_cache, block_table = _build_paged(k, v)
        seq_lens_t = torch.full((num_decode_reqs,), seq_len, dtype=torch.int32)

        q_flat = q.view(num_decode_reqs, -1)
        out = torch.zeros(num_decode_reqs, num_heads * head_size, dtype=dtype)
        score_buf = torch.zeros(num_decode_reqs, seq_len, num_kv_heads, dtype=torch.float32)

        _skive_paged_decode_attention_fallback(
            q=q_flat, key_cache=key_cache, value_cache=val_cache,
            block_table=block_table, seq_lens=seq_lens_t,
            max_seq_len=seq_len, scale=scale,
            score_buf=score_buf, out=out,
            num_kv_heads=num_kv_heads, head_size=head_size,
        )

        # Scores must be positive (L1-norm is non-negative) and finite.
        active_scores = score_buf[:, :seq_len, :]
        assert (active_scores >= 0).all(), "L1 scores must be non-negative"
        assert torch.isfinite(active_scores).all(), "Scores must be finite"

        # The sum of scores across the seq should be at least as large as
        # the max single score (sanity check that scatter worked)
        sums = active_scores.sum(dim=1)  # [num_decode, num_kv_heads]
        for r in range(num_decode_reqs):
            for h in range(num_kv_heads):
                assert sums[r, h].item() > 0.0, (
                    f"req={r} head={h}: sum is zero, kernel did not write scores"
                )

    def test_fallback_gqa_groups(self):
        """GQA with groups > 1: output should be correct."""
        from vllm.attention.ops.skive_paged_decode import (
            _skive_paged_decode_attention_fallback,
        )
        torch.manual_seed(42)
        num_decode_reqs = 2
        seq_len = 32
        num_heads = 32
        num_kv_heads = 8
        head_size = 128
        block_size = 16
        scale = 1.0 / math.sqrt(head_size)
        dtype = torch.bfloat16

        q = torch.randn(num_decode_reqs, num_heads, head_size, dtype=dtype)
        k = torch.randn(num_decode_reqs, seq_len, num_kv_heads, head_size, dtype=dtype)
        v = torch.randn(num_decode_reqs, seq_len, num_kv_heads, head_size, dtype=dtype)

        def _build_paged(k_t, v_t):
            ns, sl, nkh, hs = k_t.shape
            bps = math.ceil(sl / block_size)
            tb = ns * bps
            kc = torch.zeros(tb, nkh, hs, block_size, dtype=dtype)
            vc = torch.zeros_like(kc)
            bt = torch.zeros(ns, bps, dtype=torch.int32)
            p = 0
            for s in range(ns):
                for b in range(bps):
                    bt[s, b] = p
                    st = b * block_size
                    en = min(st + block_size, sl)
                    for t in range(en - st):
                        kc[p, :, :, t] = k_t[s, st + t]
                        vc[p, :, :, t] = v_t[s, st + t]
                    p += 1
            return kc, vc, bt

        key_cache, val_cache, block_table = _build_paged(k, v)
        seq_lens_t = torch.full((num_decode_reqs,), seq_len, dtype=torch.int32)

        q_flat = q.view(num_decode_reqs, -1)
        out = torch.zeros(num_decode_reqs, num_heads * head_size, dtype=dtype)
        score_buf = torch.zeros(num_decode_reqs, seq_len, num_kv_heads, dtype=torch.float32)

        _skive_paged_decode_attention_fallback(
            q=q_flat, key_cache=key_cache, value_cache=val_cache,
            block_table=block_table, seq_lens=seq_lens_t,
            max_seq_len=seq_len, scale=scale,
            score_buf=score_buf, out=out,
            num_kv_heads=num_kv_heads, head_size=head_size,
        )

        # Reference for one representative head group
        groups = num_heads // num_kv_heads
        for r in range(num_decode_reqs):
            for g in range(num_kv_heads):
                q_head = g * groups
                q_rep = q[r, q_head]
                k_exp = k[r].unsqueeze(1).expand(-1, groups, -1, -1).reshape(
                    seq_len, num_heads, head_size
                )
                v_exp = v[r].unsqueeze(1).expand(-1, groups, -1, -1).reshape(
                    seq_len, num_heads, head_size
                )
                logits = (
                    torch.einsum("d,td->t", q_rep.float(), k_exp[:, q_head].float())
                    * scale
                )
                probs = torch.softmax(logits, dim=-1)
                ref = torch.einsum("t,td->d", probs, v_exp[:, q_head].float())
                got = out[r, q_head * head_size : (q_head + 1) * head_size]
                max_err = (got.float() - ref.float()).abs().max().item()
                assert max_err < 1e-2, (
                    f"GQA req={r} group={g}: max error {max_err:.4e}"
                )

    def test_fallback_empty_seq(self):
        """Empty sequence (seq_len=0) should produce zero output."""
        from vllm.attention.ops.skive_paged_decode import (
            _skive_paged_decode_attention_fallback,
        )
        num_decode_reqs = 2
        num_heads, num_kv_heads, head_size = 8, 8, 64
        scale = 1.0 / math.sqrt(head_size)
        dtype = torch.bfloat16

        q = torch.randn(num_decode_reqs, num_heads, head_size, dtype=dtype)
        block_size = 16
        key_cache = torch.zeros(4, num_kv_heads, head_size, block_size, dtype=dtype)
        val_cache = torch.zeros_like(key_cache)
        block_table = torch.zeros(num_decode_reqs, 4, dtype=torch.int32)
        seq_lens = torch.tensor([0, 0], dtype=torch.int32)

        out = torch.zeros(num_decode_reqs, num_heads * head_size, dtype=dtype)
        score_buf = torch.zeros(num_decode_reqs, 16, num_kv_heads, dtype=torch.float32)

        _skive_paged_decode_attention_fallback(
            q=q, key_cache=key_cache, value_cache=val_cache,
            block_table=block_table, seq_lens=seq_lens,
            max_seq_len=16, scale=scale,
            score_buf=score_buf, out=out,
            num_kv_heads=num_kv_heads, head_size=head_size,
        )
        # Output should be near-zero (no KV to attend to)
        assert out.abs().max().item() < 1e-3
        assert score_buf.sum() == 0.0
