import pytest
import torch

from nanotron.config.models_config import Qwen2Config
from nanotron.parallel.tensor_parallel.functional import compute_target_token_ranks, compute_topk_loss_mask


def _logits(rows):
    return torch.tensor([rows], dtype=torch.float)


def test_rank_is_one_for_argmax_target():
    logits = _logits([[3.0, 1.0, 2.0, 0.0]])
    target = torch.tensor([[0]])

    assert compute_target_token_ranks(logits, target, group=None).tolist() == [[1]]


def test_rank_counts_strictly_greater_logits():
    logits = _logits([[3.0, 1.0, 2.0, 0.0]])

    assert compute_target_token_ranks(logits, torch.tensor([[2]]), group=None).tolist() == [[2]]
    assert compute_target_token_ranks(logits, torch.tensor([[1]]), group=None).tolist() == [[3]]
    assert compute_target_token_ranks(logits, torch.tensor([[3]]), group=None).tolist() == [[4]]


def test_rank_tie_does_not_increase_rank():
    logits = _logits([[5.0, 5.0, 1.0]])

    assert compute_target_token_ranks(logits, torch.tensor([[0]]), group=None).tolist() == [[1]]
    assert compute_target_token_ranks(logits, torch.tensor([[1]]), group=None).tolist() == [[1]]
    assert compute_target_token_ranks(logits, torch.tensor([[2]]), group=None).tolist() == [[3]]


def test_mask_drops_tokens_outside_topk():
    logits = _logits([[3.0, 1.0, 2.0, 0.0]] * 4)
    labels = torch.tensor([[0, 2, 1, 3]])
    mask = torch.ones((1, 4), dtype=torch.bool)

    out = compute_topk_loss_mask(logits, labels, mask, tp_pg=None, k=2)

    assert out.tolist() == [[True, True, False, False]]


def test_boundary_keeps_token_rank_equal_to_k():
    logits = _logits([[3.0, 1.0, 2.0, 0.0]] * 2)
    labels = torch.tensor([[2, 1]])
    mask = torch.ones((1, 2), dtype=torch.bool)

    out = compute_topk_loss_mask(logits, labels, mask, tp_pg=None, k=2)

    assert out.tolist() == [[True, False]]


def test_preserves_existing_label_mask():
    logits = _logits([[3.0, 1.0, 2.0, 0.0]] * 4)
    labels = torch.tensor([[0, 0, 1, 3]])
    mask = torch.tensor([[True, False, True, True]])

    out = compute_topk_loss_mask(logits, labels, mask, tp_pg=None, k=2)

    assert out.tolist() == [[True, False, False, False]]
    assert bool((out.bool() & ~mask.bool()).any()) is False


def test_preserves_non_binary_float_mask_values():
    logits = _logits([[3.0, 1.0, 2.0, 0.0]] * 4)
    labels = torch.tensor([[0, 2, 1, 3]])
    mask = torch.tensor([[0.5, 2.0, 0.5, 2.0]], dtype=torch.float32)

    out = compute_topk_loss_mask(logits, labels, mask, tp_pg=None, k=2)

    assert out.dtype == torch.float32
    assert out.tolist() == [[0.5, 2.0, 0.0, 0.0]]


def test_all_dropped_candidates_preserve_original_mask():
    logits = _logits([[3.0, 1.0, 2.0, 0.0]] * 2)
    labels = torch.tensor([[1, 3]])
    mask = torch.tensor([[0.5, 2.0]], dtype=torch.float32)

    out = compute_topk_loss_mask(logits, labels, mask, tp_pg=None, k=1)

    assert out.tolist() == [[0.5, 2.0]]


def test_max_drop_percent_caps_number_of_drops():
    logits = _logits([[3.0, 1.0, 2.0, 0.0]] * 4)
    labels = torch.tensor([[0, 2, 1, 3]])
    mask = torch.ones((1, 4), dtype=torch.bool)

    out = compute_topk_loss_mask(logits, labels, mask, tp_pg=None, k=1, max_drop_percent=50.0)

    assert out.tolist() == [[True, True, False, False]]


def test_max_drop_percent_tie_break_is_deterministic_by_index():
    logits = _logits([[3.0, 1.0, 2.0, 0.0]] * 4)
    labels = torch.tensor([[3, 3, 3, 3]])
    mask = torch.ones((1, 4), dtype=torch.bool)

    out = compute_topk_loss_mask(logits, labels, mask, tp_pg=None, k=1, max_drop_percent=50.0)

    assert out.tolist() == [[False, False, True, True]]


def test_max_drop_percent_zero_drops_nothing():
    logits = _logits([[3.0, 1.0, 2.0, 0.0]] * 4)
    labels = torch.tensor([[0, 2, 1, 3]])
    mask = torch.ones((1, 4), dtype=torch.bool)

    out = compute_topk_loss_mask(logits, labels, mask, tp_pg=None, k=1, max_drop_percent=0.0)

    assert out.tolist() == [[True, True, True, True]]


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


def test_config_rejects_high_loss_and_topk_together():
    with pytest.raises(AssertionError):
        Qwen2Config(
            vocab_size=100,
            high_loss_mask_top_percent=1.0,
            topk_loss_mask_enabled=True,
            topk_loss_mask_k=8,
        )


def test_config_max_drop_percent_out_of_range_rejected():
    with pytest.raises(AssertionError):
        Qwen2Config(vocab_size=100, topk_loss_mask_enabled=True, topk_loss_mask_k=8, topk_loss_mask_max_drop_percent=150.0)
    with pytest.raises(AssertionError):
        Qwen2Config(vocab_size=100, topk_loss_mask_enabled=True, topk_loss_mask_k=8, topk_loss_mask_max_drop_percent=-1.0)


def test_stats_eligible_uses_bool_count_not_sum_of_weights():
    logits = _logits([[3.0, 1.0, 2.0, 0.0]] * 4)
    labels = torch.tensor([[0, 2, 1, 3]])
    orig_mask = torch.tensor([[0.5, 2.0, 0.5, 2.0]], dtype=torch.float32)
    new_mask = compute_topk_loss_mask(logits, labels, orig_mask, tp_pg=None, k=2)
    drop_mask = orig_mask.bool() & ~new_mask.bool()

    assert int(orig_mask.bool().sum()) == 4
    assert int(drop_mask.sum()) == 2
