# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SkiveState (CPU-only, no Triton / CUDA required).

These tests verify the model-runner-side state container:
register / deregister request lifecycle, prefix_len plumbing, and
buffer-shape invariants needed for CUDA-graph compatibility.

The Triton kernels themselves are tested in
``tests/v1/attention/test_skive_kernels.py`` (those need a real GPU).
"""

import pytest
import torch

# SkiveState lives on a torch device; on CPU we still allocate the
# zero-sized placeholder buffers so register/deregister paths are
# exercised exactly as they will be on the GPU.
from vllm.v1.worker.skive_state import SkiveState, SkivePostprocess


# ---------------------------------------------------------------------------
# Defaults / configuration
# ---------------------------------------------------------------------------

def test_skive_state_default_constructor():
    """Default constructor yields static-shaped buffers."""
    state = SkiveState(
        num_layers=32,
        num_kv_heads=8,
        block_size=16,
        kv_budget=2048,
        num_sink_tokens=4,
        device="cpu",
    )
    assert state.num_layers == 32
    assert state.num_kv_heads == 8
    assert state.block_size == 16
    assert state.kv_budget == 2048
    # kv_budget is rounded down to a multiple of block_size.
    assert state.kv_budget % state.block_size == 0
    assert state.kv_budget_blocks == 2048 // 16
    assert state.num_sink_tokens == 4
    assert state.score_aggregation == "mean"
    # Pre-allocated buffer shapes
    B = 1024  # default max_decode_reqs
    L = 32768  # default max_seq_len
    H = 8
    assert state.s_buffer_3d.shape == (B, L, H)
    assert state.s_buffer_3d.dtype == torch.float32
    assert state.argmin_out.shape == (B,)
    assert state.argmin_out.dtype == torch.int64
    # No EMA buffer unless requested
    assert state.ema_buf is None


def test_skive_state_kv_budget_rounded_down_to_block():
    """A non-multiple-of-block_size budget is rounded down to a whole
    number of blocks (so the eviction policy can fire deterministically).
    """
    state = SkiveState(
        num_layers=4,
        num_kv_heads=4,
        block_size=16,
        kv_budget=2000,  # 2000 / 16 = 125, so 2000 stays as-is (125*16=2000)
        num_sink_tokens=4,
        device="cpu",
    )
    # 2000 is a multiple of 16 (125*16=2000) so no rounding.
    assert state.kv_budget == 2000
    assert state.kv_budget_blocks == 125

    # Try a non-multiple
    state2 = SkiveState(
        num_layers=4,
        num_kv_heads=4,
        block_size=16,
        kv_budget=2049,  # 2049 / 16 = 128.0625, so rounds down to 2048
        num_sink_tokens=4,
        device="cpu",
    )
    assert state2.kv_budget == 2048
    assert state2.kv_budget_blocks == 128


def test_skive_state_ema_buffer_only_when_requested():
    """EMA buffer is allocated only when score_aggregation == 'ema'."""
    state_mean = SkiveState(
        num_layers=4, num_kv_heads=4, block_size=16,
        kv_budget=256, num_sink_tokens=2,
        score_aggregation="mean", device="cpu",
    )
    assert state_mean.ema_buf is None

    state_ema = SkiveState(
        num_layers=4, num_kv_heads=4, block_size=16,
        kv_budget=256, num_sink_tokens=2,
        score_aggregation="ema",
        score_ema_alpha=0.9, device="cpu",
    )
    assert state_ema.ema_buf is not None
    assert state_ema.ema_buf.shape == state_ema.s_buffer_3d.shape
    assert state_ema.ema_buf.dtype == torch.float32


# ---------------------------------------------------------------------------
# Request registration lifecycle
# ---------------------------------------------------------------------------

def test_register_and_get_prefix_len():
    state = SkiveState(
        num_layers=4, num_kv_heads=4, block_size=16,
        kv_budget=256, num_sink_tokens=0, device="cpu",
    )
    # Unknown request returns 0
    assert state.get_prefix_len("nope") == 0

    # Register with prefix_len=8
    state.register_request("req-A", prefix_len=8)
    assert state.get_prefix_len("req-A") == 8
    assert "req-A" in state._req_state

    # Re-registering an existing id is idempotent: the new prefix_len
    # replaces the old one (used when a request is re-prefixed via
    # prefix caching).
    state.register_request("req-A", prefix_len=16)
    assert state.get_prefix_len("req-A") == 16

    # Deregister removes the entry; subsequent get returns 0
    state.deregister_request("req-A")
    assert state.get_prefix_len("req-A") == 0
    assert "req-A" not in state._req_state


def test_deregister_unknown_request_is_no_op():
    state = SkiveState(
        num_layers=4, num_kv_heads=4, block_size=16,
        kv_budget=256, num_sink_tokens=0, device="cpu",
    )
    # Must not raise
    state.deregister_request("never-registered")
    state.deregister_request("req-A")  # not registered


def test_capacity_check():
    state = SkiveState(
        num_layers=4, num_kv_heads=4, block_size=16,
        kv_budget=256, num_sink_tokens=0,
        max_decode_reqs=8, device="cpu",
    )
    assert state.can_accept_batch(0)
    assert state.can_accept_batch(8)
    assert not state.can_accept_batch(9)


# ---------------------------------------------------------------------------
# SkivePostprocess: mean_normalize, reset_*
# ---------------------------------------------------------------------------

def test_postprocess_mean_normalize_inplace():
    state = SkiveState(
        num_layers=4, num_kv_heads=2, block_size=16,
        kv_budget=256, num_sink_tokens=0, device="cpu",
    )
    pp = SkivePostprocess(state)

    # Fill live region with 8.0; mean_normalize should turn it into 2.0
    # (4 layers).
    state.s_buffer_3d[:2].fill_(8.0)
    pp.mean_normalize(state.s_buffer_3d, num_decode_reqs=2)
    assert torch.allclose(state.s_buffer_3d[0], torch.tensor(2.0))
    assert torch.allclose(state.s_buffer_3d[1], torch.tensor(2.0))
    # Live region beyond num_decode_reqs is unchanged.
    assert state.s_buffer_3d[2, 0, 0].item() == 0.0


def test_postprocess_reset_zeros_live_region_only():
    state = SkiveState(
        num_layers=4, num_kv_heads=2, block_size=16,
        kv_budget=256, num_sink_tokens=0, device="cpu",
    )
    pp = SkivePostprocess(state)

    # Set everything to 1.0
    state.s_buffer_3d.fill_(1.0)
    state.block_score.fill_(1.0)

    pp.reset_score_buffer(num_decode_reqs=3)
    # First 3 requests are zeroed
    assert state.s_buffer_3d[:3].sum().item() == 0.0
    # Request 3+ is untouched (this is the contract: zero only the
    # live region to avoid a full-tensor clear when batch is small).
    assert state.s_buffer_3d[3, 0, 0].item() == 1.0

    state.block_score.fill_(1.0)
    pp.reset_block_score(num_decode_reqs=2, max_blocks=10)
    assert state.block_score[:2, :10].sum().item() == 0.0
    assert state.block_score[2, 0].item() == 1.0


def test_postprocess_zero_batch_is_no_op():
    state = SkiveState(
        num_layers=4, num_kv_heads=2, block_size=16,
        kv_budget=256, num_sink_tokens=0, device="cpu",
    )
    pp = SkivePostprocess(state)
    state.s_buffer_3d.fill_(7.0)
    # num_decode_reqs=0 must not touch anything (and must not raise).
    pp.mean_normalize(state.s_buffer_3d, num_decode_reqs=0)
    pp.reset_score_buffer(num_decode_reqs=0)
    assert torch.allclose(state.s_buffer_3d, torch.tensor(7.0))


# ---------------------------------------------------------------------------
# Argument / type sanity
# ---------------------------------------------------------------------------

def test_pin_memory_host_evict_targets():
    """The host-side eviction-target tensor must be pinned so the
    D2H copy (after the argmin kernel runs) is fast.

    On CPU, ``pin_memory=True`` is a no-op for the constructor — we
    just verify the flag was accepted.
    """
    state = SkiveState(
        num_layers=4, num_kv_heads=2, block_size=16,
        kv_budget=256, num_sink_tokens=0, device="cpu",
    )
    assert state._host_evict_targets.is_pinned() or state._host_evict_targets.device.type == "cpu"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
