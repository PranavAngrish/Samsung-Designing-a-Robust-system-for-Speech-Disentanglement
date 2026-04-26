"""ONNX export round-trip test for the deployable SoloSpeak graph."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from solospeak.deployment.export_onnx import ExportWrapper, export_to_onnx
from solospeak.models.solospeak import SoloSpeakModel
from solospeak.utils.config import SoloSpeakConfig


_OUTPUT_NAMES = ["z_c", "z_s", "content_score", "speaker_score", "fusion_score"]


def _save_checkpoint(path: Path) -> None:
    cfg = SoloSpeakConfig.from_yaml("configs/defaults.yaml")
    model = SoloSpeakModel(cfg).eval()
    torch.save(
        {
            "config_snapshot": cfg.model_dump(mode="json"),
            "stage_origin": 6,
            "model_state": model.state_dict(),
        },
        path,
    )


@pytest.mark.slow
@pytest.mark.onnx
def test_export_wrapper_zero_templates_no_nan() -> None:
    cfg = SoloSpeakConfig.from_yaml("configs/defaults.yaml")
    wrapper = ExportWrapper(SoloSpeakModel(cfg).eval()).eval()
    mel = torch.randn(1, 1, 80, 160)
    content_template = torch.zeros(1, 128)
    speaker_template = torch.zeros(1, 128)

    with torch.no_grad():
        outputs = wrapper(mel, content_template, speaker_template)

    assert [tuple(output.shape) for output in outputs] == [
        (1, 128),
        (1, 128),
        (1,),
        (1,),
        (1,),
    ]
    assert all(not torch.isnan(output).any() for output in outputs)


@pytest.mark.slow
@pytest.mark.onnx
def test_onnx_export_and_run(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    checkpoint = tmp_path / "stage6_qat.pt"
    onnx_path = tmp_path / "solospeak.onnx"
    _save_checkpoint(checkpoint)

    export_to_onnx(checkpoint, onnx_path)
    model = onnx.load(str(onnx_path))
    assert [output.name for output in model.graph.output] == _OUTPUT_NAMES

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    mel = np.random.randn(1, 1, 80, 160).astype(np.float32)
    zero = np.zeros((1, 128), dtype=np.float32)
    outputs = session.run(
        None,
        {"mel": mel, "content_template": zero, "speaker_template": zero},
    )
    assert [output.shape for output in outputs] == [(1, 128), (1, 128), (1,), (1,), (1,)]
    assert all(np.isfinite(output).all() for output in outputs)
