import torch

from nanotron.models.qwen import _apply_high_loss_token_mask


def test_top_percent_zero_returns_full_mask():
    loss = torch.tensor([[1.0, 2.0, 0.5, 4.0]])
    label_ids = torch.tensor([[10, 20, 30, 40]])
    label_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)
    mask, stats = _apply_high_loss_token_mask(loss, label_ids, label_mask, top_percent=0.0)
    assert torch.equal(mask, label_mask)
    assert stats == {}


def test_self_loss_drops_argmax():
    loss = torch.tensor([[1.0, 2.0, 0.5, 4.0]])
    label_ids = torch.tensor([[10, 20, 30, 40]])
    label_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)
    mask, stats = _apply_high_loss_token_mask(loss, label_ids, label_mask, top_percent=25.0)
    assert mask.tolist() == [[True, True, True, False]]
    assert stats["high_loss_masked_tokens"].item() == 1.0


def test_reference_loss_drops_its_own_argmax():
    loss = torch.tensor([[1.0, 2.0, 0.5, 4.0]])
    label_ids = torch.tensor([[10, 20, 30, 40]])
    label_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)
    ref = torch.tensor([[5.0, 0.1, 0.2, 0.3]])
    mask, stats = _apply_high_loss_token_mask(
        loss, label_ids, label_mask, top_percent=25.0, reference_loss=ref
    )
    assert mask.tolist() == [[False, True, True, True]]
    assert stats["high_loss_masked_tokens"].item() == 1.0


def test_returned_mask_shape_consistent():
    loss = torch.randn(2, 8)
    label_ids = torch.randint(0, 100, (2, 8))
    label_mask = torch.ones(2, 8, dtype=torch.bool)
    mask, _ = _apply_high_loss_token_mask(loss, label_ids, label_mask, top_percent=10.0)
    assert mask.shape == label_mask.shape
    assert mask.dtype == torch.bool
