"""Unit tests for GatedFusionMLP."""

from __future__ import annotations

import pytest
import torch

from solospeak.models.fusion import GatedFusionMLP, naive_fusion_min, naive_fusion_product
from solospeak.utils.config import FusionConfig


@pytest.fixture()
def fusion() -> GatedFusionMLP:
    return GatedFusionMLP(FusionConfig())


def test_output_shape(fusion: GatedFusionMLP) -> None:
    s_c = torch.rand(8)
    s_s = torch.rand(8)
    out = fusion(s_c, s_s)
    assert out.shape == (8,)


def test_output_in_zero_one(fusion: GatedFusionMLP) -> None:
    s_c = torch.rand(16)
    s_s = torch.rand(16)
    out = fusion(s_c, s_s)
    assert (out >= 0).all() and (out <= 1).all()


def test_param_count_under_400() -> None:
    fusion = GatedFusionMLP(FusionConfig())
    n = sum(p.numel() for p in fusion.parameters())
    assert n <= 400, f"Fusion MLP has {n} params, expected <= 400"


def test_gradient_flows_through_fusion(fusion: GatedFusionMLP) -> None:
    """Backward pass must work end-to-end — checks graph is connected."""
    s_c = torch.rand(4, requires_grad=True)
    s_s = torch.rand(4, requires_grad=True)
    out = fusion(s_c, s_s)
    out.sum().backward()
    assert s_c.grad is not None and not s_c.grad.isnan().any()
    assert s_s.grad is not None and not s_s.grad.isnan().any()


def test_naive_min() -> None:
    s_c = torch.tensor([0.8, 0.3])
    s_s = torch.tensor([0.4, 0.9])
    out = naive_fusion_min(s_c, s_s)
    assert torch.allclose(out, torch.tensor([0.4, 0.3]))


def test_naive_product() -> None:
    s_c = torch.tensor([0.5, 0.8])
    s_s = torch.tensor([0.6, 0.5])
    out = naive_fusion_product(s_c, s_s)
    assert torch.allclose(out, torch.tensor([0.3, 0.4]))
