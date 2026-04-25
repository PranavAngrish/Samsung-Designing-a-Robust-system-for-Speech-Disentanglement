"""ONNX export round-trip test (requires Phase 2 BC-ResNet implementation)."""

from __future__ import annotations

import pytest


@pytest.mark.slow
@pytest.mark.onnx
def test_onnx_export_and_run(tmp_path: "pathlib.Path") -> None:
    pytest.skip("Requires Phase 2 BC-ResNet implementation")
