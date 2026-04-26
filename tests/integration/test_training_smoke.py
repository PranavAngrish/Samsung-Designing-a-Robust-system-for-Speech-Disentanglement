"""Smoke tests for the training pipeline (no real data needed)."""

from __future__ import annotations

import pytest
import torch

from solospeak.losses.supcon import SupConLoss
from solospeak.losses.orthogonality import orthogonality_loss
from solospeak.training.schedulers import cosine_schedule_with_warmup, linear_ramp


def test_supcon_backward_with_mixed_labels() -> None:
    """Full forward+backward through SupCon loss."""
    import torch.nn.functional as F
    loss_fn = SupConLoss(temperature=0.07)
    z_raw = torch.randn(8, 128, requires_grad=True)
    z = F.normalize(z_raw, p=2, dim=-1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
    loss = loss_fn(z, labels)
    loss.backward()
    assert z_raw.grad is not None
    assert not z_raw.grad.isnan().any()


def test_orthogonality_backward() -> None:
    import torch.nn.functional as F
    z_c_raw = torch.randn(8, 128, requires_grad=True)
    z_s_raw = torch.randn(8, 128, requires_grad=True)
    z_c = F.normalize(z_c_raw, p=2, dim=-1)
    z_s = F.normalize(z_s_raw, p=2, dim=-1)
    loss = orthogonality_loss(z_c, z_s)
    loss.backward()
    assert not z_c_raw.grad.isnan().any()
    assert not z_s_raw.grad.isnan().any()


def test_cosine_schedule_reaches_zero() -> None:
    import torch.optim as optim
    model = torch.nn.Linear(4, 4)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    sched = cosine_schedule_with_warmup(opt, num_warmup_steps=10, num_training_steps=100)
    for _ in range(100):
        sched.step()
    lr = opt.param_groups[0]["lr"]
    assert lr < 1e-6


def test_linear_ramp() -> None:
    assert linear_ramp(0, 100, 0.1) == 0.0
    assert linear_ramp(50, 100, 0.1) == pytest.approx(0.05)
    assert linear_ramp(100, 100, 0.1) == pytest.approx(0.1)


@pytest.mark.slow
def test_full_model_forward_backward() -> None:
    """Requires BC-ResNet implementation — skip until Phase 2."""
    pytest.skip("Requires Phase 2 BC-ResNet implementation")
