"""Unit tests for content and speaker embedding heads."""

from __future__ import annotations

import torch

from solospeak.models.heads import EmbeddingHead


def test_output_is_l2_normalized() -> None:
    head = EmbeddingHead(input_channels=64).eval()
    x = torch.randn(4, 64, 1, 8)
    with torch.no_grad():
        out = head(x)
    norms = out.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), f"Norms not 1: {norms}"


def test_output_shape() -> None:
    head = EmbeddingHead(input_channels=64, output_dim=128).eval()
    x = torch.randn(4, 64, 1, 8)
    with torch.no_grad():
        out = head(x)
    assert out.shape == (4, 128)


def test_gradient_flows_through_l2_norm() -> None:
    """Backward pass must not produce NaN — zero vectors are clamped."""
    head = EmbeddingHead(input_channels=32)
    x = torch.randn(2, 32, 1, 4, requires_grad=False)
    out = head(x)
    loss = out.sum()
    loss.backward()
    for name, p in head.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
        assert not p.grad.isnan().any(), f"NaN grad in {name}"


def test_two_heads_have_separate_weights() -> None:
    content = EmbeddingHead(input_channels=32)
    speaker = EmbeddingHead(input_channels=32)
    for (n1, p1), (n2, p2) in zip(content.named_parameters(), speaker.named_parameters()):
        assert p1.data_ptr() != p2.data_ptr(), f"Shared param: {n1}"
