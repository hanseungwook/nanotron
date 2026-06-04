"""Tests for self-scored top-K unlikely-token loss masking.

These tests exercise the pure (non-distributed) logic of the masking:
  * ``compute_target_token_ranks`` — the self-scored rank of each target token.
  * ``compute_topk_loss_mask``     — turning ranks into an updated label_mask.
  * ``Qwen2Config`` validation     — config-level guard rails.

They run single-process (tensor-parallel group ``None``, i.e. the full vocab is
local), which is enough to cover the ranking convention, the ``rank > K`` (not
``>=``) boundary, label_mask preservation, the no-op/disabled paths and the
``max_drop_percent`` cap. The tensor-parallel all-reduce paths are exercised by
the existing distributed test harness for ``sharded_cross_entropy``.
"""

import pytest
import torch

from nanotron.config.models_config import Qwen2Config
from nanotron.parallel.tensor_parallel.functional import (
    compute_target_token_ranks,
    compute_topk_loss_mask,
)


def _logits(rows):
    """Build a [1, len(rows), vocab] logits tensor from a list of per-position rows."""
    return torch.tensor([rows], dtype=torch.float)


# ---------------------------------------------------------------------------
# compute_target_token_ranks
# ---------------------------------------------------------------------------


def test_rank_is_one_for_argmax_target():
    # Position picks the highest-logit token => rank 1.
    logits = _logits([[3.0, 1.0, 2.0, 0.0]])
    target = torch.tensor([[0]])  # token 0 has the max logit
    ranks = compute_target_token_ranks(logits, target, group=None)
    assert ranks.tolist() == [[1]]


def test_rank_counts_strictly_greater_logits():
    # logits sorted desc: token0(3) > token2(2) > token1(1) > token3(0)
    logits = _logits([[3.0, 1.0, 2.0, 0.0]])
    # target token 2 has 1 logit strictly greater (token0) => rank 2
    assert compute_target_token_ranks(logits, torch.tensor([[2]]), group=None).tolist() == [[2]]
    # target token 1 has 2 strictly greater (token0, token2) => rank 3
    assert compute_target_token_ranks(logits, torch.tensor([[1]]), group=None).tolist() == [[3]]
    # target token 3 (lowest) => rank 4
    assert compute_target_token_ranks(logits, torch.tensor([[3]]), group=None).tolist() == [[4]]


def test_rank_tie_does_not_increase_rank():
    # Two tokens tie for the top logit. Ties (equal logits) are NOT counted as
    # "strictly greater", matching score_topk.py's `rank = 1 + count(logit > target)`.
    logits = _logits([[5.0, 5.0, 1.0]])
    # token0 and token1 both have 0 strictly-greater logits => both rank 1.
    assert compute_target_token_ranks(logits, torch.tensor([[0]]), group=None).tolist() == [[1]]
    assert compute_target_token_ranks(logits, torch.tensor([[1]]), group=None).tolist() == [[1]]
    # token2 has 2 strictly greater => rank 3.
    assert compute_target_token_ranks(logits, torch.tensor([[2]]), group=None).tolist() == [[3]]


# ---------------------------------------------------------------------------
# compute_topk_loss_mask
# ---------------------------------------------------------------------------


def test_mask_drops_tokens_outside_topk():
    # Ranks per position: token0->1, token2->2, token1->3, token3->4 (same logits each pos).
    logits = _logits([[3.0, 1.0, 2.0, 0.0]] * 4)
    labels = torch.tensor([[0, 2, 1, 3]])  # ranks 1, 2, 3, 4
    mask = torch.ones((1, 4), dtype=torch.bool)
    out = compute_topk_loss_mask(logits, labels, mask, tp_pg=None, k=2)
    # rank <= 2 kept (positions 0,1), rank > 2 dropped (positions 2,3).
    assert out.tolist() == [[True, True, False, False]]


def test_boundary_uses_strict_greater_than_k():
    # A token whose rank == K must be KEPT (drop uses rank > K, not >=).
    logits = _logits([[3.0, 1.0, 2.0, 0.0]] * 2)
    labels = torch.tensor([[2, 1]])  # ranks 2 and 3
    mask = torch.ones((1, 2), dtype=torch.bool)
    out = compute_topk_loss_mask(logits, labels, mask, tp_pg=None, k=2)
    assert out.tolist() == [[True, False]]  # rank 2 kept, rank 3 dropped


def test_preserves_existing_label_mask():
    # An already-masked position (rank 1, would otherwise be kept) must stay masked.
    logits = _logits([[3.0, 1.0, 2.0, 0.0]] * 4)
    labels = torch.tensor([[0, 0, 1, 3]])  # ranks 1, 1, 3, 4
    mask = torch.tensor([[True, False, True, True]])
    out = compute_topk_loss_mask(logits, labels, mask, tp_pg=None, k=2)
    # pos0 rank1 kept; pos1 already False -> stays False; pos2/pos3 rank>2 dropped.
    assert out.tolist() == [[True, False, False, False]]
    # Returned mask is always a subset of the incoming mask.
    assert bool((out.bool() & ~mask.bool()).any()) is False


def test_mask_preserves_dtype():
    logits = _logits([[3.0, 1.0, 2.0, 0.0]] * 2)
    labels = torch.tensor([[0, 3]])
    mask = torch.ones((1, 2), dtype=torch.float32)
    out = compute_topk_loss_mask(logits, labels, mask, tp_pg=None, k=2)
    assert out.dtype == torch.float32
    assert out.tolist() == [[1.0, 0.0]]


def test_preserves_non_binary_float_mask_values():
    # A weighted (non-binary) float mask: kept positions must keep their original
    # weight, dropped positions become 0 (not 1.0).
    logits = _logits([[3.0, 1.0, 2.0, 0.0]] * 4)
    labels = torch.tensor([[0, 2, 1, 3]])  # ranks 1, 2, 3, 4
    mask = torch.tensor([[0.5, 2.0, 0.5, 2.0]], dtype=torch.float32)
    out = compute_topk_loss_mask(logits, labels, mask, tp_pg=None, k=2)
    # rank <= 2 kept with original weight (0.5, 2.0); rank > 2 dropped to 0.0.
    assert out.dtype == torch.float32
    assert out.tolist() == [[0.5, 2.0, 0.0, 0.0]]


def test_high_k_drops_nothing():
    # With K >= vocab_size every target is within the top-K, so nothing is dropped.
    logits = _logits([[3.0, 1.0, 2.0, 0.0]] * 4)
    labels = torch.tensor([[0, 1, 2, 3]])
    mask = torch.ones((1, 4), dtype=torch.bool)
    out = compute_topk_loss_mask(logits, labels, mask, tp_pg=None, k=4)
    assert out.tolist() == [[True, True, True, True]]


# ---------------------------------------------------------------------------
# max_drop_percent cap
# ---------------------------------------------------------------------------


def test_max_drop_percent_caps_number_of_drops():
    # 4 active tokens, 3 of them rank outside top-1 and would be dropped.
    logits = _logits([[3.0, 1.0, 2.0, 0.0]] * 4)
    labels = torch.tensor([[0, 2, 1, 3]])  # ranks 1, 2, 3, 4
    mask = torch.ones((1, 4), dtype=torch.bool)

    # Cap drops at 50% of 4 active => at most 2 drops, the two HIGHEST ranks (4 then 3).
    out = compute_topk_loss_mask(logits, labels, mask, tp_pg=None, k=1, max_drop_percent=50.0)
    assert int(out.bool().sum()) == 2  # 2 dropped, 2 kept
    # The most-unlikely tokens (ranks 4 and 3 at positions 3 and 2) are the ones dropped.
    assert out.tolist() == [[True, True, False, False]]


def test_max_drop_percent_tie_break_is_deterministic_by_index():
    # All four active tokens share the same rank (4), so the cap must choose which to
    # drop by a tie-break. The tie-break drops the lowest flat indices first.
    logits = _logits([[3.0, 1.0, 2.0, 0.0]] * 4)
    labels = torch.tensor([[3, 3, 3, 3]])  # every position -> rank 4
    mask = torch.ones((1, 4), dtype=torch.bool)
    # 50% of 4 active => drop exactly 2; with equal ranks, positions 0 and 1 are dropped.
    out = compute_topk_loss_mask(logits, labels, mask, tp_pg=None, k=1, max_drop_percent=50.0)
    assert int(out.bool().sum()) == 2
    assert out.tolist() == [[False, False, True, True]]


def test_max_drop_percent_zero_drops_nothing():
    logits = _logits([[3.0, 1.0, 2.0, 0.0]] * 4)
    labels = torch.tensor([[0, 2, 1, 3]])
    mask = torch.ones((1, 4), dtype=torch.bool)
    out = compute_topk_loss_mask(logits, labels, mask, tp_pg=None, k=1, max_drop_percent=0.0)
    assert out.tolist() == [[True, True, True, True]]


def test_default_max_drop_percent_no_cap():
    # Default 100.0 => no cap, all out-of-topK tokens dropped.
    logits = _logits([[3.0, 1.0, 2.0, 0.0]] * 4)
    labels = torch.tensor([[0, 2, 1, 3]])
    mask = torch.ones((1, 4), dtype=torch.bool)
    out = compute_topk_loss_mask(logits, labels, mask, tp_pg=None, k=1)
    assert out.tolist() == [[True, False, False, False]]


# ---------------------------------------------------------------------------
# Qwen2Config validation
# ---------------------------------------------------------------------------


def test_config_disabled_by_default():
    cfg = Qwen2Config(vocab_size=100)
    assert cfg.topk_loss_mask_enabled is False
    assert cfg.topk_loss_mask_k is None
    assert cfg.topk_loss_mask_max_drop_percent == 100.0


def test_config_enabled_requires_k():
    with pytest.raises(AssertionError):
        Qwen2Config(vocab_size=100, topk_loss_mask_enabled=True)


def test_config_k_must_be_positive():
    with pytest.raises(AssertionError):
        Qwen2Config(vocab_size=100, topk_loss_mask_enabled=True, topk_loss_mask_k=0)


def test_config_k_must_be_less_than_vocab_size():
    with pytest.raises(AssertionError):
        Qwen2Config(vocab_size=100, topk_loss_mask_enabled=True, topk_loss_mask_k=100)


def test_config_valid_when_enabled_with_k():
    cfg = Qwen2Config(vocab_size=100, topk_loss_mask_enabled=True, topk_loss_mask_k=8)
    assert cfg.topk_loss_mask_k == 8


def test_config_max_drop_percent_out_of_range_rejected():
    with pytest.raises(AssertionError):
        Qwen2Config(vocab_size=100, topk_loss_mask_enabled=True, topk_loss_mask_k=8, topk_loss_mask_max_drop_percent=150.0)
    with pytest.raises(AssertionError):
        Qwen2Config(vocab_size=100, topk_loss_mask_enabled=True, topk_loss_mask_k=8, topk_loss_mask_max_drop_percent=-1.0)
