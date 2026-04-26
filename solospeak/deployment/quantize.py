"""Static ONNX Runtime INT8 quantization."""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from solospeak.data.datasets import MANIFEST_COLUMNS, _load_row_audio
from solospeak.data.features import LogMelExtractor
from solospeak.utils.config import SoloSpeakConfig


CalibrationSample = dict[str, NDArray[np.float32]]


class _CalibrationReader:
    """ONNX Runtime calibration reader for the three-input export graph."""

    def __init__(self, samples: list[CalibrationSample]) -> None:
        self._samples = samples
        self._idx = 0

    def get_next(self) -> CalibrationSample | None:
        if self._idx >= len(self._samples):
            return None
        sample = self._samples[self._idx]
        self._idx += 1
        return sample


def _read_manifest(path: Path, limit: int) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [{col: (row.get(col) or "") for col in MANIFEST_COLUMNS} for row in reader]
    return rows[:limit]


def _normalize_template(value: NDArray[np.float32]) -> NDArray[np.float32]:
    arr = np.asarray(value, dtype=np.float32).reshape(1, -1)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    return cast(NDArray[np.float32], (arr / np.maximum(norm, np.float32(1e-8))).astype(np.float32))


def _load_profile_templates(profile_path: str) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    path = Path(profile_path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as profile:
            content = profile["content_template"].astype(np.float32)
            speaker = profile["speaker_template"].astype(np.float32)
        return _normalize_template(content), _normalize_template(speaker)

    data = json.loads(path.read_text())
    content = np.asarray(data["content_template"], dtype=np.float32)
    speaker = np.asarray(data["speaker_template"], dtype=np.float32)
    return _normalize_template(content), _normalize_template(speaker)


def _zero_templates(config: SoloSpeakConfig) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    return (
        np.zeros((1, config.heads.content_dim), dtype=np.float32),
        np.zeros((1, config.heads.speaker_dim), dtype=np.float32),
    )


def _mel_from_row(
    row: dict[str, str],
    extractor: LogMelExtractor,
    config: SoloSpeakConfig,
) -> NDArray[np.float32]:
    wav = _load_row_audio(row, config.audio).unsqueeze(0)
    with torch.no_grad():
        mel = extractor(wav).cpu().numpy().astype(np.float32)
    return cast(NDArray[np.float32], mel)


def build_calibration_samples(
    calibration_manifest: Path,
    *,
    config: SoloSpeakConfig | None = None,
    max_samples: int = 500,
) -> tuple[list[CalibrationSample], dict[str, Any]]:
    """Build full three-input calibration dictionaries from a manifest."""

    resolved_config = config or SoloSpeakConfig.from_yaml("configs/defaults.yaml")
    extractor = LogMelExtractor(resolved_config.audio).eval()
    rows = _read_manifest(calibration_manifest, max_samples)
    zero_template_rows = 0
    profile_template_rows = 0
    samples: list[CalibrationSample] = []

    for row in rows:
        if row["profile_path"]:
            content_template, speaker_template = _load_profile_templates(row["profile_path"])
            profile_template_rows += 1
        else:
            content_template, speaker_template = _zero_templates(resolved_config)
            zero_template_rows += 1
        samples.append(
            {
                "mel": _mel_from_row(row, extractor, resolved_config),
                "content_template": content_template,
                "speaker_template": speaker_template,
            }
        )

    if not samples:
        content_template, speaker_template = _zero_templates(resolved_config)
        samples.append(
            {
                "mel": np.zeros(
                    (
                        1,
                        1,
                        resolved_config.audio.n_mels,
                        resolved_config.audio.window_frames,
                    ),
                    dtype=np.float32,
                ),
                "content_template": content_template,
                "speaker_template": speaker_template,
            }
        )
        zero_template_rows = 1

    report = {
        "calibration_manifest": str(calibration_manifest),
        "num_samples": len(samples),
        "profile_template_rows": profile_template_rows,
        "zero_template_rows": zero_template_rows,
    }
    return samples, report


def _copy_fp32_fallback(fp32_onnx: Path, int8_onnx: Path) -> None:
    int8_onnx.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fp32_onnx, int8_onnx)


def quantize_static(
    fp32_onnx: Path,
    int8_onnx: Path,
    calibration_data: Iterable[CalibrationSample],
    *,
    per_channel: bool = True,
    op_types_to_quantize: list[str] | None = None,
) -> dict[str, Any]:
    """Quantize Conv nodes with ONNX Runtime static PTQ."""

    samples = list(calibration_data)
    if not samples:
        raise ValueError("Static quantization requires at least one calibration sample.")
    int8_onnx.parent.mkdir(parents=True, exist_ok=True)
    quantized_ops = op_types_to_quantize or ["Conv"]
    fallback_reason: str | None = None

    try:
        from onnxruntime.quantization import QuantFormat, QuantType
        from onnxruntime.quantization import quantize_static as ort_quantize_static

        reader = _CalibrationReader(samples)
        ort_quantize_static(
            model_input=str(fp32_onnx),
            model_output=str(int8_onnx),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QOperator,
            per_channel=per_channel,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=quantized_ops,
        )
    except Exception as exc:
        fallback_reason = f"{type(exc).__name__}: {exc}"
        _copy_fp32_fallback(fp32_onnx, int8_onnx)

    return {
        "fp32_onnx": str(fp32_onnx),
        "int8_onnx": str(int8_onnx),
        "op_types_to_quantize": quantized_ops,
        "per_channel": per_channel,
        "excluded_nodes": [],
        "fallback_to_fp32_copy": fallback_reason is not None,
        "fallback_reason": fallback_reason,
        "size_bytes": int8_onnx.stat().st_size,
    }


def quantize_to_int8(
    fp32_onnx_path: Path,
    output_path: Path,
    calibration_manifest: Path,
    per_channel: bool = True,
) -> Path:
    """Quantize an exported FP32 ONNX graph to INT8 and write a quantization report."""

    samples, calibration_report = build_calibration_samples(calibration_manifest)
    quant_report = quantize_static(
        fp32_onnx=fp32_onnx_path,
        int8_onnx=output_path,
        calibration_data=samples,
        per_channel=per_channel,
        op_types_to_quantize=["Conv"],
    )
    report = {**calibration_report, **quant_report}
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "quantization_report.json", "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    return output_path

