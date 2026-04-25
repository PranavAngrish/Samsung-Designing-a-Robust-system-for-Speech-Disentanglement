"""Unit tests for BC-ResNet backbone family."""

from __future__ import annotations

import pytest
import torch

from solospeak.models.backbones.bcresnet import BCResNet, VARIANTS


@pytest.mark.parametrize("variant,expected_range", [
    ("bcresnet1",  (75_000,   90_000)),
    ("bcresnet5",  (450_000,  550_000)),
    ("bcresnet8",  (900_000,  1_100_000)),
    ("bcresnet10", (1_350_000, 1_650_000)),
    ("bcresnet16", (2_700_000, 3_300_000)),
])
@pytest.mark.slow
def test_param_counts(variant: str, expected_range: tuple[int, int]) -> None:
    model = BCResNet(variant)
    n = sum(p.numel() for p in model.parameters())
    lo, hi = expected_range
    assert lo <= n <= hi, f"{variant}: got {n}, expected [{lo}, {hi}]"


@pytest.mark.slow
def test_forward_shape_bcresnet8() -> None:
    """Output must be (B, C, 1, T') with T' ≈ T/8."""
    model = BCResNet("bcresnet8").eval()
    mel = torch.randn(2, 1, 80, 160)
    with torch.no_grad():
        out = model(mel)
    assert out.shape[0] == 2
    assert out.shape[2] == 1            # frequency reduced to 1


@pytest.mark.slow
def test_unknown_variant_raises() -> None:
    with pytest.raises(ValueError, match="Unknown variant"):
        BCResNet("bcresnet99")


def test_variants_dict_complete() -> None:
    expected = {"bcresnet1", "bcresnet5", "bcresnet8", "bcresnet10", "bcresnet16"}
    assert set(VARIANTS.keys()) == expected
