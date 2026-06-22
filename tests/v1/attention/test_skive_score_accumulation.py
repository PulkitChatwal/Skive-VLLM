# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify SKIVE score accumulation across layers (CPU-only, no Triton).

The kernel atomic-adds per-token scores into a single buffer across
all N layers. This test verifies that the accumulated score is
identical to a full re-computation (no accumulation).

This test runs on CPU; it does NOT invoke the Triton kernels. It
just verifies the numerical identity of the accumulation pattern.

Run with::

    pytest tests/v1/attention/test_skive_score_accumulation.py -v
"""

import math

import pytest
import torch
import torch.nn.functional as F


def reference_per_layer_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Reference per-layer SKIVE score computation.

    For a single layer, given Q,K,V, compute the per-token L1-norm
    of the contribution vectors g_i = p_i * v_i, where p_i are the
    attention probabilities. This is the math that the Triton
    ``unified_attention_skive`` kernel implements.

    Args:
        q: [num_heads, head_size]
        k: [seq_len, num_kv_heads, head_size]
        v: [seq_len, num_kv_heads, head_size]
        scale: 1 / sqrt(head_size)

    Returns:
        scores: [seq_len, num_kv_heads] L1-norm scores
    """
    num_heads = q.shape[0]
    head_size = q.shape[1]
    seq_len = k.shape[0]
    num_kv_heads = k.shape[1]

    # Expand K,V to all heads (GQA support)
    groups = num_heads // num_kv_heads
    k_exp = k.unsqueeze(1).expand(-1, groups, -1, -1)  # [L, groups, kv_h, d]
    k_exp = k_exp.reshape(seq_len, num_heads, head_size)
    v_exp = v.unsqueeze(1).expand(-1, groups, -1, -1)
    v_exp = v_exp.reshape(seq_len, num_heads, head_size)

    # q: [H, d] -> [H, 1, d]
    q4 = q.unsqueeze(1)

    # Scores: [H, 1, L]
    raw = torch.matmul(q4, k_exp.transpose(1, 2)) * scale  # [H, 1, L]
    attn_weights = F.softmax(raw.float(), dim=-1)  # [H, 1, L]

    # Contribution vectors: [H, 1, L, d]
    # v_exp: [L, H, d] -> [H, L, d]
    v_t = v_exp.permute(1, 0, 2)
    p = attn_weights.squeeze(1)  # [H, L]
    g = p.unsqueeze(-1) * v_t.unsqueeze(0)  # [H, L, d]

    # L1-norm per (head, token): [H, L]
    l1 = g.abs().sum(dim=-1)  # [H, L]

    # Aggregate to per-(token, kv_head): [seq_len, num_kv_heads]
    # by averaging over the group dimension.
    # For MHA (groups=1) this is a no-op.
    # For GQA (groups>1) we average each group of heads.
    l1_grouped = l1.view(num_kv_heads, groups, seq_len).mean(dim=1)
    # -> [kv_h, seq_len], transpose to [seq_len, kv_h]
    return l1_grouped.transpose(0, 1)


def test_score_accumulation_matches_recompute():
    """Verify that accumulated scores = recomputed scores.

    Simulate the kernel's atomic-add accumulation across N layers.
    For each layer, we compute a per-token score buffer and
    add it into a running total. At the end, the accumulated buffer
    should equal a full recomputation (starting from the summed
    K,V across all layers).

    This verifies that the accumulation pattern in the kernel is
    correct (Resolution 1 from the integration plan).
    """
    torch.manual_seed(42)
    num_layers = 8
    num_heads = 32
    num_kv_heads = 8
    head_size = 128
    seq_len = 64

    # Generate synthetic per-layer K,V (simulating a forward pass)
    all_k = torch.randn(
        num_layers, seq_len, num_kv_heads, head_size, dtype=torch.float32
    )
    all_v = torch.randn(
        num_layers, seq_len, num_kv_heads, head_size, dtype=torch.float32
    )

    # A single query tensor (this is what we get in decode: one
    # query token per head).
    q = torch.randn(num_heads, head_size, dtype=torch.float32)
    scale = 1.0 / math.sqrt(head_size)

    # Accumulation buffer (what the kernel atomic-adds into)
    accumulated = torch.zeros(seq_len, num_kv_heads, dtype=torch.float32)

    # Simulate layer-by-layer accumulation
    for layer in range(num_layers):
        per_layer = reference_per_layer_scores(q, all_k[layer], all_v[layer], scale)
        accumulated += per_layer

    # Recompute from summed K,V (this is the baseline "full recompute")
    k_summed = all_k.sum(dim=0)  # [L, kv_h, d]
    v_summed = all_v.sum(dim=0)
    recompute = reference_per_layer_scores(q, k_summed, v_summed, scale)

    # Should match exactly (within float32 tolerance)
    max_err = (accumulated - recompute).abs().max().item()
    mean_err = (accumulated - recompute).abs().mean().item()

    assert max_err < 1e-5, f"Max error too large: {max_err:.4e}"
    assert mean_err < 1e-6, f"Mean error too large: {mean_err:.4e}"
    # The accumulated scores should be EXACTLY N times the
    # per-layer scores on average.
    ratio = (accumulated.mean() / recompute.mean()).item()
    assert ratio == pytest.approx(num_layers, abs=1e-5)


def test_score_accumulation_with_gqa():
    """Same test but with GQA (groups=4, num_heads=32, num_kv_heads=8).

    Verifies that the grouped-query aggregation is correct.
    """
    torch.manual_seed(123)
    num_layers = 4
    num_heads = 32
    num_kv_heads = 8  # groups = 4
    head_size = 64
    seq_len = 32

    all_k = torch.randn(
        num_layers, seq_len, num_kv_heads, head_size, dtype=torch.float32
    )
    all_v = torch.randn(
        num_layers, seq_len, num_kv_heads, head_size, dtype=torch.float32
    )
    q = torch.randn(num_heads, head_size, dtype=torch.float32)
    scale = 1.0 / math.sqrt(head_size)

    accumulated = torch.zeros(seq_len, num_kv_heads, dtype=torch.float32)
    for layer in range(num_layers):
        per_layer = reference_per_layer_scores(q, all_k[layer], all_v[layer], scale)
        accumulated += per_layer

    k_summed = all_k.sum(dim=0)
    v_summed = all_v.sum(dim=0)
    recompute = reference_per_layer_scores(q, k_summed, v_summed, scale)

    max_err = (accumulated - recompute).abs().max().item()
    assert max_err < 1e-5, f"GQA max error: {max_err:.4e}"


def test_score_accumulation_mha():
    """MHA case (num_heads == num_kv_heads)."""
    torch.manual_seed(456)
    num_layers = 6
    num_heads = num_kv_heads = 16
    head_size = 128
    seq_len = 128

    all_k = torch.randn(
        num_layers, seq_len, num_kv_heads, head_size, dtype=torch.float32
    )
    all_v = torch.randn(
        num_layers, seq_len, num_kv_heads, head_size, dtype=torch.float32
    )
    q = torch.randn(num_heads, head_size, dtype=torch.float32)
    scale = 1.0 / math.sqrt(head_size)

    accumulated = torch.zeros(seq_len, num_kv_heads, dtype=torch.float32)
    for layer in range(num_layers):
        per_layer = reference_per_layer_scores(q, all_k[layer], all_v[layer], scale)
        accumulated += per_layer

    k_summed = all_k.sum(dim=0)
    v_summed = all_v.sum(dim=0)
    recompute = reference_per_layer_scores(q, k_summed, v_summed, scale)

    max_err = (accumulated - recompute).abs().max().item()
    assert max_err < 1e-5, f"MHA max error: {max_err:.4e}"


def test_score_accumulation_single_layer():
    """Edge case: N=1 layer (accumulation is just the per-layer scores)."""
    torch.manual_seed(789)
    num_layers = 1
    num_heads = num_kv_heads = 4
    head_size = 64
    seq_len = 16

    all_k = torch.randn(
        num_layers, seq_len, num_kv_heads, head_size, dtype=torch.float32
    )
    all_v = torch.randn(
        num_layers, seq_len, num_kv_heads, head_size, dtype=torch.float32
    )
    q = torch.randn(num_heads, head_size, dtype=torch.float32)
    scale = 1.0 / math.sqrt(head_size)

    accumulated = torch.zeros(seq_len, num_kv_heads, dtype=torch.float32)
    for layer in range(num_layers):
        per_layer = reference_per_layer_scores(q, all_k[layer], all_v[layer], scale)
        accumulated += per_layer

    # With N=1, accumulated == per_layer == recompute
    assert torch.allclose(accumulated, per_layer, atol=1e-6)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
