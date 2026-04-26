"""Unit tests for the SoloSpeakResNet backbone family."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from solospeak.models.backbones.bcresnet import BCResNet, SoloSpeakResNet, VARIANTS


EXPECTED_PARAMS = {
    "bcresnet1": 65_032,
    "bcresnet5": 512_720,
    "bcresnet8": 985_080,
    "bcresnet10": 1_583_136,
    "bcresnet16": 3_193_160,
}


@pytest.mark.parametrize("variant,expected", EXPECTED_PARAMS.items())
def test_param_counts(variant: str, expected: int) -> None:
    model = SoloSpeakResNet(variant)
    actual = sum(p.numel() for p in model.parameters())
    lo = int(expected * 0.98)
    hi = int(expected * 1.02)
    assert lo <= actual <= hi, f"{variant}: got {actual}, expected {expected} +/-2%"


@pytest.mark.parametrize("variant", EXPECTED_PARAMS)
def test_all_variants_forward_shape(variant: str) -> None:
    model = SoloSpeakResNet(variant).eval()
    mel = torch.randn(4, 1, 80, 160)
    with torch.no_grad():
        out = model(mel)
    assert out.shape == (4, model.output_channels, 1, 20)


def test_forward_shape_bcresnet8() -> None:
    model = SoloSpeakResNet("bcresnet8")
    mel = torch.randn(4, 1, 80, 160)
    out = model(mel)
    assert out.shape == (4, 96, 1, 20)
    assert model.output_channels == 96


def test_backward() -> None:
    model = SoloSpeakResNet("bcresnet8")
    mel = torch.randn(2, 1, 80, 160, requires_grad=True)
    out = model(mel)
    out.sum().backward()
    assert mel.grad is not None
    assert torch.isfinite(mel.grad).all()


@pytest.mark.onnx
def test_onnx_export(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    model = SoloSpeakResNet("bcresnet8").eval()
    mel = torch.randn(1, 1, 80, 160)
    onnx_path = tmp_path / "bcresnet.onnx"
    torch.onnx.export(
        model,
        mel,
        onnx_path,
        opset_version=17,
        input_names=["mel"],
        output_names=["feat"],
    )
    assert onnx_path.exists()


@pytest.mark.onnx
def test_fuse_model_export(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    model = SoloSpeakResNet("bcresnet8").eval()
    model.fuse_model()
    mel = torch.randn(1, 1, 80, 160)
    out = model(mel)
    assert out.shape == (1, 96, 1, 20)
    onnx_path = tmp_path / "bcresnet_fused.onnx"
    torch.onnx.export(
        model,
        mel,
        onnx_path,
        opset_version=17,
        input_names=["mel"],
        output_names=["feat"],
    )
    assert onnx_path.exists()


def test_unknown_variant_raises() -> None:
    with pytest.raises(ValueError, match="Unknown variant"):
        BCResNet("bcresnet99")


def test_variants_dict_complete() -> None:
    expected = {"bcresnet1", "bcresnet5", "bcresnet8", "bcresnet10", "bcresnet16"}
    assert set(VARIANTS.keys()) == expected
