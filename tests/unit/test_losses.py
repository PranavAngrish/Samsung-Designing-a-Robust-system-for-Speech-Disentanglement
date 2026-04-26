"""Unit tests for all loss functions."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from solospeak.losses.adversarial import adversarial_lambda, grad_reverse
from solospeak.losses.combined import CombinedLoss
from solospeak.losses.orthogonality import orthogonality_loss
from solospeak.losses.supcon import SupConLoss
from solospeak.utils.config import LossWeights
from solospeak.utils.types import WORD_IGNORE_INDEX


# ---------------------------------------------------------------------------
# SupCon
# ---------------------------------------------------------------------------


def test_supcon_finite_nonzero_with_positives() -> None:
    loss_fn = SupConLoss(temperature=0.1)
    z = F.normalize(torch.randn(4, 128), p=2, dim=-1)
    labels = torch.tensor([0, 0, 1, 1])
    loss = loss_fn(z, labels)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_supcon_zero_with_warning_when_all_unique() -> None:
    loss_fn = SupConLoss(temperature=0.1)
    z = F.normalize(torch.randn(4, 128), p=2, dim=-1)
    labels = torch.tensor([0, 1, 2, 3])
    with pytest.warns(UserWarning, match="no positives"):
        loss = loss_fn(z, labels)
    assert loss.item() == pytest.approx(0.0)


def test_supcon_decreases_with_more_aligned_positives() -> None:
    loss_fn = SupConLoss(temperature=0.07)
    torch.manual_seed(42)
    z_bad = F.normalize(torch.randn(8, 128), p=2, dim=-1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    base = z_bad.clone()
    z_good = base.clone()
    for cls in range(4):
        idx = (labels == cls).nonzero(as_tuple=True)[0]
        z_good[idx] = F.normalize(base[idx].mean(0, keepdim=True).expand_as(base[idx]), p=2, dim=-1)

    loss_bad = loss_fn(z_bad, labels)
    loss_good = loss_fn(z_good, labels)
    assert loss_good < loss_bad


def test_supcon_no_nan() -> None:
    loss_fn = SupConLoss()
    z = F.normalize(torch.randn(16, 128), p=2, dim=-1)
    labels = torch.randint(0, 5, (16,))
    loss = loss_fn(z, labels)
    assert not loss.isnan()


# ---------------------------------------------------------------------------
# Orthogonality
# ---------------------------------------------------------------------------


def test_orthogonality_positive_for_correlated() -> None:
    bsz, dim = 16, 128
    z = F.normalize(torch.randn(bsz, dim), p=2, dim=-1)
    loss = orthogonality_loss(z, z)
    assert loss.item() > 0.0


def test_orthogonality_decreases_monotonically() -> None:
    torch.manual_seed(0)
    z_c = torch.nn.Parameter(torch.randn(12, 8))
    z_s = torch.nn.Parameter(z_c.detach().clone() + 0.1 * torch.randn(12, 8))
    opt = torch.optim.SGD([z_c, z_s], lr=0.1)
    prev = float("inf")
    for _ in range(100):
        opt.zero_grad()
        loss = orthogonality_loss(z_c, z_s)
        assert loss.item() <= prev + 1e-6
        prev = loss.item()
        loss.backward()
        opt.step()


# ---------------------------------------------------------------------------
# Gradient reversal
# ---------------------------------------------------------------------------


def test_gradient_reversal_flips_sign_and_scales() -> None:
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = grad_reverse(x, 0.5)
    y.sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.tensor([-0.5, -0.5, -0.5]))


def test_adversarial_lambda_ramp() -> None:
    assert adversarial_lambda(0, 1000, 0.1) == 0.0
    assert adversarial_lambda(500, 1000, 0.1) == pytest.approx(0.05)
    assert adversarial_lambda(1000, 1000, 0.1) == pytest.approx(0.1)
    assert adversarial_lambda(2000, 1000, 0.1) == pytest.approx(0.1)


def test_combined_loss_routes_aux_ce_with_ignore_index() -> None:
    loss_fn = CombinedLoss(stage=2, weights=LossWeights())
    content_labels = torch.tensor([0, WORD_IGNORE_INDEX, 1, 1, 0])
    speaker_labels = torch.tensor([0, 0, 1, 1, 0])
    outputs = {
        "content": {
            "z_c": F.normalize(torch.randn(5, 128), p=2, dim=-1),
            "word_logits": torch.randn(5, 2, requires_grad=True),
        },
        "speaker": {
            "z_s": F.normalize(torch.randn(5, 128), p=2, dim=-1),
            "speaker_logits": torch.randn(5, 2, requires_grad=True),
        },
    }
    batches = {
        "content": {"keyword_label": content_labels},
        "speaker": {"speaker_label": speaker_labels},
    }
    loss = loss_fn(outputs, batches)
    assert torch.isfinite(loss.total)
    loss.total.backward()


def test_combined_loss_stage6_distill_stream() -> None:
    loss_fn = CombinedLoss(stage=6, weights=LossWeights())
    student_z_c = torch.randn(3, 128, requires_grad=True)
    student_z_s = torch.randn(3, 128, requires_grad=True)
    outputs = {
        "distill": {
            "student_z_c": student_z_c,
            "teacher_z_c": torch.randn(3, 128),
            "student_z_s": student_z_s,
            "teacher_z_s": torch.randn(3, 128),
        }
    }
    loss = loss_fn(outputs, {})
    assert torch.isfinite(loss.total)
    assert loss.distill > 0.0
    loss.total.backward()
    assert student_z_c.grad is not None
    assert student_z_s.grad is not None
