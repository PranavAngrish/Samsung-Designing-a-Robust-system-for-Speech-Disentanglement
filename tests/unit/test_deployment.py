"""Unit tests for deployment utilities."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import torch

from solospeak.deployment.ota_package import build_ota_package, _sha256
from solospeak.deployment.validate_artifact import ValidationReport, GateResult
from solospeak.models.solospeak import SoloSpeakModel
from solospeak.training.stages.stage6_qat import copy_float_weights_from_prepared
from solospeak.utils.config import SoloSpeakConfig


def test_sha256_is_deterministic(tmp_path: Path) -> None:
    f = tmp_path / "test.bin"
    f.write_bytes(b"hello world")
    h1 = _sha256(f)
    h2 = _sha256(f)
    assert h1 == h2
    assert len(h1) == 64  # hex digest


def test_build_ota_package_creates_zip(tmp_path: Path) -> None:
    onnx = tmp_path / "solospeak_int8.onnx"
    vad = tmp_path / "silero_vad_v4.onnx"
    onnx.write_bytes(b"\x00" * 100)
    vad.write_bytes(b"\x01" * 50)

    zip_path = build_ota_package(onnx, vad, tmp_path / "out", version="1.0.0")

    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert "MANIFEST.json" in names
        assert "solospeak_int8.onnx" in names
        assert "silero_vad_v4.onnx" in names
        assert "README.txt" in names

    with zipfile.ZipFile(zip_path) as zf:
        manifest = json.loads(zf.read("MANIFEST.json"))
    assert manifest["version"] == "1.0.0"
    assert "solospeak_int8.onnx" in manifest["files"]


def test_validation_report_passes_all() -> None:
    gates = [GateResult("g1", True, 1.0, 5.0, "ok"), GateResult("g2", True, 0.5, 1.0, "ok")]
    report = ValidationReport(gates)
    assert report.passed is True
    assert report.all_passed is True


def test_validation_report_fails_on_one_fail() -> None:
    gates = [GateResult("g1", True, 1.0, 5.0, "ok"), GateResult("g2", False, 6.0, 5.0, "fail")]
    report = ValidationReport(gates)
    assert report.passed is False


def test_copy_float_weights_from_prepared() -> None:
    cfg = SoloSpeakConfig.from_yaml("configs/defaults.yaml")
    prepared_source = SoloSpeakModel(cfg).train()
    prepared_source.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
    prepared = torch.ao.quantization.prepare_qat(prepared_source, inplace=False)

    with torch.no_grad():
        for param in prepared.parameters():
            param.add_(0.001)

    target = SoloSpeakModel(cfg)
    copy_float_weights_from_prepared(prepared, target)

    source_params = dict(prepared.named_parameters())
    for name, param in target.named_parameters():
        assert torch.allclose(param, source_params[name])
    assert not any(
        "FakeQuant" in type(module).__name__ or "Observer" in type(module).__name__
        for module in target.modules()
    )
