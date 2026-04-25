"""ONNX INT8 post-training quantization.

Input:  artifacts/solospeak_fp32.onnx  (~18 MB)
Output: artifacts/solospeak_int8.onnx  (~4.5 MB)

Uses onnxruntime.quantization.quantize_static with per-channel quantization.
Calibration data: GSC-v2 validation set (100 batches).
"""

from __future__ import annotations

from pathlib import Path


def quantize_to_int8(
    fp32_onnx_path: Path,
    output_path: Path,
    calibration_manifest: Path,
    per_channel: bool = True,
) -> Path:
    """Quantize FP32 ONNX model to INT8 using static calibration.

    Verifies: TA degradation < 0.3 pp vs FP32 baseline.
    """
    raise NotImplementedError("Implement in Phase 5")
