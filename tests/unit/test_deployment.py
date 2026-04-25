"""Unit tests for deployment utilities."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from solospeak.deployment.ota_package import build_ota_package, _sha256
from solospeak.deployment.validate_artifact import ValidationReport, GateResult


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

    with zipfile.ZipFile(zip_path) as zf:
        manifest = json.loads(zf.read("MANIFEST.json"))
    assert manifest["version"] == "1.0.0"
    assert "solospeak_int8.onnx" in manifest["files"]


def test_validation_report_passes_all() -> None:
    gates = [GateResult("g1", True, 1.0, 5.0, "ok"), GateResult("g2", True, 0.5, 1.0, "ok")]
    report = ValidationReport(gates)
    assert report.passed is True


def test_validation_report_fails_on_one_fail() -> None:
    gates = [GateResult("g1", True, 1.0, 5.0, "ok"), GateResult("g2", False, 6.0, 5.0, "fail")]
    report = ValidationReport(gates)
    assert report.passed is False
