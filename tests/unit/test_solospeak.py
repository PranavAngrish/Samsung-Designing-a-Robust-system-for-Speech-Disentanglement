"""Unit tests for the assembled SoloSpeak model."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from solospeak.models.solospeak import SoloSpeakModel
from solospeak.models.vad import SileroVAD
from solospeak.utils.config import SoloSpeakConfig


def test_model_class_imports() -> None:
    assert SoloSpeakModel.__name__ == "SoloSpeakModel"


def test_forward_shape_and_unit_norm() -> None:
    cfg = SoloSpeakConfig.from_yaml("configs/defaults.yaml")
    model = SoloSpeakModel(cfg).eval()
    mel = torch.randn(3, 1, 80, 160)
    with torch.no_grad():
        z_c, z_s = model(mel)
    assert z_c.shape == (3, 128)
    assert z_s.shape == (3, 128)
    assert torch.allclose(z_c.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(z_s.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_backward_pass() -> None:
    cfg = SoloSpeakConfig.from_yaml("configs/defaults.yaml")
    model = SoloSpeakModel(cfg)
    mel = torch.randn(2, 1, 80, 160, requires_grad=True)
    z_c, z_s = model(mel)
    (z_c.sum() + z_s.sum()).backward()
    assert mel.grad is not None
    assert torch.isfinite(mel.grad).all()


def test_total_param_count_bcresnet8() -> None:
    cfg = SoloSpeakConfig.from_yaml("configs/defaults.yaml")
    model = SoloSpeakModel(cfg)
    n = sum(p.numel() for p in model.parameters())
    assert 1_080_000 <= n <= 1_125_000, f"Got {n}"


@pytest.mark.onnx
def test_onnx_export(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    cfg = SoloSpeakConfig.from_yaml("configs/defaults.yaml")
    model = SoloSpeakModel(cfg).eval()
    mel = torch.randn(1, 1, 80, 160)
    onnx_path = tmp_path / "solospeak.onnx"
    torch.onnx.export(
        model,
        mel,
        onnx_path,
        opset_version=17,
        input_names=["mel"],
        output_names=["z_c", "z_s"],
    )
    assert onnx_path.exists()


class _FakeOrtOptions:
    inter_op_num_threads: int
    intra_op_num_threads: int


class _FakeOrtInput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeOrtSession:
    def __init__(self, path: str, sess_options: _FakeOrtOptions) -> None:
        self.path = path
        self.sess_options = sess_options

    def get_inputs(self) -> list[_FakeOrtInput]:
        return [_FakeOrtInput("input"), _FakeOrtInput("state"), _FakeOrtInput("sr")]

    def run(
        self,
        _output_names: None,
        _inputs: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        return [
            np.array([[0.1]], dtype=np.float32),
            np.zeros((2, 1, 128), dtype=np.float32),
        ]


def test_silero_vad_import_constants_and_shape_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ort = types.SimpleNamespace(
        SessionOptions=_FakeOrtOptions,
        InferenceSession=_FakeOrtSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    assert SileroVAD.FRAME_SAMPLES == 512
    assert SileroVAD.SAMPLE_RATE == 16000
    assert SileroVAD.CONTEXT_SAMPLES == 64

    vad = SileroVAD(Path("unused.onnx"))
    with pytest.raises(ValueError, match="512 samples"):
        vad.is_speech(np.zeros(511, dtype=np.float32))
    assert vad.is_speech(np.zeros(512, dtype=np.float32)) is False
    vad.reset_state()


@pytest.mark.onnx
def test_silero_vad_wrapper_with_pinned_artifact() -> None:
    onnx_path = Path("artifacts/silero_vad_v4.onnx")
    if not onnx_path.exists():
        pytest.skip("artifacts/silero_vad_v4.onnx is not present")
    vad = SileroVAD(onnx_path)
    assert vad.is_speech(np.zeros(SileroVAD.FRAME_SAMPLES, dtype=np.float32)) is False
    vad.reset_state()
