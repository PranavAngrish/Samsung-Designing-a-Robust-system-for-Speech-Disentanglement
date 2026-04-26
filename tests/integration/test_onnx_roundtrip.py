"""ONNX export round-trip test for the deployable SoloSpeak model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from solospeak.models.solospeak import SoloSpeakModel
from solospeak.utils.config import SoloSpeakConfig


@pytest.mark.slow
@pytest.mark.onnx
def test_onnx_export_and_run(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    cfg = SoloSpeakConfig.from_yaml("configs/defaults.yaml")
    model = SoloSpeakModel(cfg).eval()
    mel = torch.randn(1, 1, 80, 160)
    onnx_path = tmp_path / "solospeak.onnx"

    with torch.no_grad():
        ref_z_c, ref_z_s = model(mel)
    torch.onnx.export(
        model,
        mel,
        onnx_path,
        opset_version=17,
        input_names=["mel"],
        output_names=["z_c", "z_s"],
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out_z_c, out_z_s = session.run(None, {"mel": mel.numpy()})
    assert out_z_c.shape == (1, 128)
    assert out_z_s.shape == (1, 128)
    np.testing.assert_allclose(out_z_c, ref_z_c.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out_z_s, ref_z_s.numpy(), rtol=1e-4, atol=1e-4)
