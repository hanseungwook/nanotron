"""Unit tests for the reference-model pre-loop wiring in nanotron.trainer.

These exercise the pure per-microbatch logic (no real model, no CUDA): the reference
forward and the sharded cross-entropy are mocked, so the tests run on CPU. They verify
that `_compute_reference_score_for_microbatch` attaches a [B, S] `reference_loss` and the
correct `reference_logits` payload (full logits vs. an empty dummy), and that the
config guard `_check_reference_requires_masking` fails fast on a misconfiguration.
"""
import pytest
import torch

import nanotron.trainer as trainer_mod
from nanotron.trainer import (
    _check_reference_requires_masking,
    _compute_reference_score_for_microbatch,
)

B, S, V = 2, 4, 8


def _make_micro():
    return {
        "input_ids": torch.randint(0, V, (B, S)),
        "position_ids": torch.arange(S).unsqueeze(0).expand(B, S).contiguous(),
        "label_ids": torch.randint(0, V, (B, S)),
        "label_mask": torch.ones(B, S, dtype=torch.bool),
    }


def _patch_ref(monkeypatch, captured=None):
    """Mock the reference forward (returns [B*S, V]) and the sharded CE (returns [B, S])."""

    def fake_call_reference_forward(ref_model, ref_cls_name, micro):
        return torch.randn(B * S, V)

    def fake_sharded_cross_entropy(sharded_logits, target, group=None, dtype=None, z_loss_coef=0.0):
        if captured is not None:
            captured["logits_shape"] = tuple(sharded_logits.shape)
            captured["target_shape"] = tuple(target.shape)
            captured["group"] = group
            captured["dtype"] = dtype
        return torch.full((B, S), 1.5)

    monkeypatch.setattr(trainer_mod, "call_reference_forward", fake_call_reference_forward)
    monkeypatch.setattr(trainer_mod, "sharded_cross_entropy", fake_sharded_cross_entropy)


def test_attaches_reference_loss_and_full_logits(monkeypatch):
    captured = {}
    _patch_ref(monkeypatch, captured)
    micro = _make_micro()
    sentinel_group = object()

    _compute_reference_score_for_microbatch(
        ref_model=object(),
        ref_cls_name="Qwen2Config",
        ref_tp_pg=sentinel_group,
        micro=micro,
        return_full_logits=True,
        ref_stream=None,
    )

    assert micro["reference_loss"].shape == (B, S)
    assert micro["reference_loss"].dtype == torch.float32
    # full logits attached and detached from any graph
    assert micro["reference_logits"].shape == (B, S, V)
    assert micro["reference_logits"].requires_grad is False
    # CE is computed on the reference's TP group, over [B, S, V] logits vs [B, S] targets
    assert captured["logits_shape"] == (B, S, V)
    assert captured["target_shape"] == (B, S)
    assert captured["group"] is sentinel_group
    assert captured["dtype"] == torch.float


def test_attaches_dummy_logits_when_not_returning_full(monkeypatch):
    _patch_ref(monkeypatch)
    micro = _make_micro()

    _compute_reference_score_for_microbatch(
        ref_model=object(),
        ref_cls_name="Qwen2Config",
        ref_tp_pg=object(),
        micro=micro,
        return_full_logits=False,
        ref_stream=None,
    )

    assert micro["reference_loss"].shape == (B, S)
    dummy = micro["reference_logits"]
    assert dummy.numel() == 0
    assert dummy.dtype == torch.float32


def test_check_requires_masking_raises_when_enabled_without_masking():
    with pytest.raises(ValueError, match="high_loss_mask_top_percent"):
        _check_reference_requires_masking(reference_enabled=True, high_loss_mask_top_percent=0.0)


def test_check_requires_masking_accepts_valid_and_disabled():
    # enabled + masking on -> ok
    _check_reference_requires_masking(reference_enabled=True, high_loss_mask_top_percent=5.0)
    # disabled -> ok regardless of masking
    _check_reference_requires_masking(reference_enabled=False, high_loss_mask_top_percent=0.0)
    # None masking value is treated as "off" and must raise when enabled
    with pytest.raises(ValueError):
        _check_reference_requires_masking(reference_enabled=True, high_loss_mask_top_percent=None)
