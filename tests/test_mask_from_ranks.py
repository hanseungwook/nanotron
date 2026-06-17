"""Tests for ``mask_from_ranks`` — the top-K drop policy shared by self-scoring
and offline reference (teacher) ranks.

These import only ``nanotron.parallel.tensor_parallel.functional`` (torch-only,
no model/config/flash-attn chain) so the pure masking policy is testable in a
minimal environment. They exercise the policy directly on hand-built ranks and
pin that ``compute_topk_loss_mask`` is exactly
``mask_from_ranks(compute_target_token_ranks(...), ...)``.
"""

import torch

from nanotron.parallel.tensor_parallel.functional import (
    compute_target_token_ranks,
    compute_topk_loss_mask,
    mask_from_ranks,
)


def _logits(rows):
    return torch.tensor([rows], dtype=torch.float)


def test_mask_from_ranks_matches_compute_topk_loss_mask():
    # The self-scoring wrapper must delegate to mask_from_ranks bit-for-bit.
    logits = _logits([[3.0, 1.0, 2.0, 0.0]] * 4)
    labels = torch.tensor([[0, 2, 1, 3]])
    mask = torch.ones((1, 4), dtype=torch.bool)
    ranks = compute_target_token_ranks(logits, labels, group=None)
    for k in (1, 2, 3):
        for drop in (100.0, 50.0, 0.0):
            assert torch.equal(
                mask_from_ranks(ranks, mask, k=k, max_drop_percent=drop),
                compute_topk_loss_mask(logits, labels, mask, tp_pg=None, k=k, max_drop_percent=drop),
            )


def test_mask_from_ranks_drops_above_k_keeps_at_k():
    ranks = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones((1, 4), dtype=torch.bool)

    assert mask_from_ranks(ranks, mask, k=2).tolist() == [[True, True, False, False]]


def test_mask_from_ranks_respects_existing_mask():
    ranks = torch.tensor([[5, 5, 1, 5]])
    mask = torch.tensor([[True, False, True, True]])

    # already-inactive position stays inactive; high-rank actives drop; rank-1 kept.
    assert mask_from_ranks(ranks, mask, k=2).tolist() == [[False, False, True, False]]


def test_mask_from_ranks_preserves_non_binary_float_mask_values():
    ranks = torch.tensor([[1, 3, 1, 4]])
    mask = torch.tensor([[0.5, 2.0, 0.5, 2.0]], dtype=torch.float32)

    out = mask_from_ranks(ranks, mask, k=2)
    assert out.dtype == torch.float32
    assert out.tolist() == [[0.5, 0.0, 0.5, 0.0]]


def test_mask_from_ranks_cap_drops_highest_rank_first():
    ranks = torch.tensor([[1, 10, 1, 100]])
    mask = torch.ones((1, 4), dtype=torch.bool)

    # k=2 -> candidates ranks 10 and 100; cap 25% of 4 active = 1 drop -> drop highest (100).
    assert mask_from_ranks(ranks, mask, k=2, max_drop_percent=25.0).tolist() == [[True, True, True, False]]


def test_mask_from_ranks_tie_break_is_deterministic_by_index():
    ranks = torch.tensor([[7, 7, 7, 7]])
    mask = torch.ones((1, 4), dtype=torch.bool)

    # all rank 7 > k=1; cap 50% -> 2 drops; equal ranks broken by lowest flat index.
    assert mask_from_ranks(ranks, mask, k=1, max_drop_percent=50.0).tolist() == [[False, False, True, True]]


def test_mask_from_ranks_empty_guard_returns_original():
    ranks = torch.tensor([[5, 5]])
    mask = torch.tensor([[0.5, 2.0]], dtype=torch.float32)

    # would drop every active token -> original mask returned to avoid empty loss.
    assert mask_from_ranks(ranks, mask, k=1).tolist() == [[0.5, 2.0]]


def test_mask_from_ranks_handles_uint16_clamped_ranks():
    # Offline sidecar stores min(rank, 65535); very-high (clamped) ranks must still drop.
    ranks = torch.tensor([[1, 65535, 50, 200]])
    mask = torch.ones((1, 4), dtype=torch.bool)

    assert mask_from_ranks(ranks, mask, k=100).tolist() == [[True, False, True, False]]
