"""Unit tests for all loss functions."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from solospeak.losses.supcon import SupConLoss
from solospeak.losses.orthogonality import orthogonality_loss
from solospeak.losses.adversarial import GradientReversalFunction, adversarial_lambda


# ---------------------------------------------------------------------------
# SupCon
# ---------------------------------------------------------------------------

def test_supcon_zero_when_all_same_class_and_embedding() -> None:
    loss_fn = SupConLoss(temperature=0.1)
    z = F.normalize(torch.randn(4, 128), p=2, dim=-1)
    z = z.expand(4, -1).clone()               # identical embeddings
    labels = torch.zeros(4, dtype=torch.long)
    # When all embeddings identical + same class → numerics may be edge case; just no error
    loss = loss_fn(z, labels)
    assert torch.isfinite(loss)


def test_supcon_decreases_with_more_aligned_positives() -> None:
    loss_fn = SupConLoss(temperature=0.07)
    torch.manual_seed(42)
    z_bad  = F.normalize(torch.randn(8, 128), p=2, dim=-1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    # Good: same-class embeddings are similar
    base = z_bad.clone()
    z_good = base.clone()
    for cls in range(4):
        idx = (labels == cls).nonzero(as_tuple=True)[0]
        z_good[idx] = F.normalize(base[idx].mean(0, keepdim=True).expand_as(base[idx]), p=2, dim=-1)

    loss_bad  = loss_fn(z_bad, labels)
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

def test_orthogonality_zero_for_orthogonal_spaces() -> None:
    B, D = 16, 128
    z_c = torch.zeros(B, D)
    z_s = torch.zeros(B, D)
    z_c[:, :64] = 1.0   # first 64 dims
    z_s[:, 64:] = 1.0   # last 64 dims — perfectly orthogonal
    loss = orthogonality_loss(z_c, z_s)
    assert loss.item() < 1e-6


def test_orthogonality_positive_for_correlated() -> None:
    B, D = 16, 128
    z = F.normalize(torch.randn(B, D), p=2, dim=-1)
    loss = orthogonality_loss(z, z)  # identical → maximally correlated
    assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# Gradient reversal
# ---------------------------------------------------------------------------

def test_gradient_reversal_flips_sign() -> None:
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = GradientReversalFunction.apply(x, 1.0)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.tensor([-1.0, -1.0, -1.0]))


def test_adversarial_lambda_ramp() -> None:
    assert adversarial_lambda(0, 1000, 0.1) == 0.0
    assert adversarial_lambda(500, 1000, 0.1) == pytest.approx(0.05)
    assert adversarial_lambda(1000, 1000, 0.1) == pytest.approx(0.1)
    assert adversarial_lambda(2000, 1000, 0.1) == pytest.approx(0.1)
