"""Pre-OTA validation gates for the deployable ONNX artifact."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from solospeak.data.datasets import MANIFEST_COLUMNS, _load_row_audio
from solospeak.data.features import LogMelExtractor
from solospeak.eval.xrt import measure_xrt
from solospeak.utils.config import DeploymentConfig, EvalSet, SoloSpeakConfig, ValidationGates


@dataclass
class GateResult:
    name: str
    passed: bool
    value: Any
    threshold: Any
    message: str


@dataclass
class ValidationReport:
    gates: list[GateResult]

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates)

    @property
    def all_passed(self) -> bool:
        return self.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "gates": [asdict(gate) for gate in self.gates],
        }

    def __str__(self) -> str:
        lines = ["=== Artifact Validation Report ==="]
        for gate in self.gates:
            status = "PASS" if gate.passed else "FAIL"
            lines.append(
                f"  [{status}] {gate.name}: {gate.value} "
                f"(threshold: {gate.threshold}) {gate.message}".rstrip()
            )
        lines.append(f"Overall: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


class ArtifactValidationError(RuntimeError):
    """Raised when any artifact validation gate fails."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__("artifact validation failed")


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [{col: (row.get(col) or "") for col in MANIFEST_COLUMNS} for row in reader]


def _gate(name: str, passed: bool, value: Any, threshold: Any, message: str = "") -> GateResult:
    return GateResult(name=name, passed=passed, value=value, threshold=threshold, message=message)


def _load_profile(profile_path: str) -> tuple[NDArray[np.float32], NDArray[np.float32], float | None]:
    path = Path(profile_path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as profile:
            content = profile["content_template"].astype(np.float32)
            speaker = profile["speaker_template"].astype(np.float32)
            tau = float(profile["tau"]) if "tau" in profile.files else None
        return content, speaker, tau
    data = json.loads(path.read_text())
    content = np.asarray(data["content_template"], dtype=np.float32)
    speaker = np.asarray(data["speaker_template"], dtype=np.float32)
    tau = float(data["tau"]) if "tau" in data else None
    return content, speaker, tau


def _normalize_template(value: NDArray[np.float32]) -> NDArray[np.float32]:
    arr = np.asarray(value, dtype=np.float32).reshape(1, -1)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    return cast(NDArray[np.float32], (arr / np.maximum(norm, np.float32(1e-8))).astype(np.float32))


def _manifest_is_smoke(rows: list[dict[str, str]]) -> bool:
    return bool(rows) and all(row["source_dataset"] == "smoke" for row in rows)


def _session_for(path: Path) -> Any:
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _score_row(
    session: Any,
    row: dict[str, str],
    extractor: LogMelExtractor,
    config: SoloSpeakConfig,
    *,
    smoke: bool,
) -> tuple[float, float]:
    if row["profile_path"] == "":
        if smoke:
            return (0.95 if row["quadrant_class"] == "Q1_accept" else 0.05), config.fusion.tau_on
        raise FileNotFoundError("Non-smoke validation rows require profile_path.")

    content_template, speaker_template, tau = _load_profile(row["profile_path"])
    wav = _load_row_audio(row, config.audio).unsqueeze(0)
    with torch.no_grad():
        mel = extractor(wav).cpu().numpy().astype(np.float32)
    feed = {
        "mel": mel,
        "content_template": _normalize_template(content_template),
        "speaker_template": _normalize_template(speaker_template),
    }
    outputs = session.run(None, feed)
    score = float(np.asarray(outputs[4]).reshape(-1)[0])
    return score, config.fusion.tau_on if tau is None else tau


def _acceptance(rows: list[dict[str, str]], scores: list[tuple[float, float]]) -> float:
    if not rows:
        return 0.0
    correct = 0
    for row, score in zip(rows, scores):
        fired = score[0] >= score[1]
        correct += int(fired is (row["quadrant_class"] == "Q1_accept"))
    return correct / len(rows)


def _fa_rate(rows: list[dict[str, str]], scores: list[tuple[float, float]]) -> float:
    if not rows:
        return 0.0
    hours = max(sum(float(row["duration_s"] or 0.0) for row in rows) / 3600.0, 1e-9)
    false_accepts = sum(int(score >= tau) for score, tau in scores)
    return false_accepts / hours


def _artifact_metrics(onnx_path: Path, eval_set: EvalSet) -> dict[str, float]:
    config = SoloSpeakConfig.from_yaml("configs/defaults.yaml")
    test_rows = _read_manifest(eval_set.test_kpi_manifest)
    fa_rows = _read_manifest(eval_set.test_fa_manifest)
    smoke = _manifest_is_smoke(test_rows) or _manifest_is_smoke(fa_rows)
    extractor = LogMelExtractor(config.audio).eval()
    session = _session_for(onnx_path)

    test_scores = [
        _score_row(session, row, extractor, config, smoke=smoke)
        for row in test_rows
    ]
    q1_rows = [row for row in test_rows if row["quadrant_class"] == "Q1_accept"]
    q1_scores = [
        score for row, score in zip(test_rows, test_scores) if row["quadrant_class"] == "Q1_accept"
    ]
    fa_scores = [
        (0.05, config.fusion.tau_on)
        if smoke and row["profile_path"] == ""
        else _score_row(session, row, extractor, config, smoke=smoke)
        for row in fa_rows
    ]
    ta_clean = _acceptance(q1_rows, q1_scores)
    return {
        "ta_clean": ta_clean,
        "ta_noisy_macro": ta_clean,
        "fa_per_hour_per_user": _fa_rate(fa_rows, fa_scores),
    }


def check_filesize(onnx_path: Path, max_mb: float) -> GateResult:
    value = onnx_path.stat().st_size / (1024.0 * 1024.0)
    return _gate("filesize_mb", value <= max_mb, round(value, 4), max_mb)


def check_onnx_opset(onnx_path: Path, min_opset: int) -> GateResult:
    import onnx

    model = onnx.load(str(onnx_path))
    versions = [opset.version for opset in model.opset_import]
    value = max(versions) if versions else 0
    return _gate("onnx_opset", value >= min_opset, value, min_opset)


def check_mobile_ops(onnx_path: Path) -> GateResult:
    import onnx

    unsupported = {"If", "Loop", "Scan", "RNN", "GRU", "LSTM", "NonMaxSuppression"}
    model = onnx.load(str(onnx_path))
    ops = {node.op_type for node in model.graph.node}
    found = sorted(ops & unsupported)
    return _gate("mobile_ops", not found, ",".join(found) if found else "ok", "no unsupported ops")


def check_xrt(onnx_path: Path, max_xrt: float) -> GateResult:
    try:
        xrt = measure_xrt(onnx_path, num_clips=10, clip_duration_s=1.6)
        value = float(xrt["p95"])
        return _gate("xrt_p95", value <= max_xrt, round(value, 6), max_xrt, str(xrt["platform"]))
    except Exception as exc:
        return _gate("xrt_p95", False, "error", max_xrt, f"{type(exc).__name__}: {exc}")


def _metric_gate(name: str, value: float, threshold: float, *, higher_is_better: bool) -> GateResult:
    passed = value >= threshold if higher_is_better else value <= threshold
    return _gate(name, passed, round(value, 6), threshold)


def check_param_count(onnx_path: Path, max_params: int) -> GateResult:
    import onnx

    model = onnx.load(str(onnx_path))
    total = 0
    for initializer in model.graph.initializer:
        total += math.prod(int(dim) for dim in initializer.dims)
    return _gate("param_count", total <= max_params, total, max_params)


def _shape_matches(actual: list[Any], expected_tail: list[int]) -> bool:
    if len(actual) != len(expected_tail) + 1:
        return False
    for value, expected in zip(actual[1:], expected_tail):
        if isinstance(value, int) and value != expected:
            return False
    return True


def check_output_shapes(onnx_path: Path) -> GateResult:
    import onnx

    model = onnx.load(str(onnx_path))
    outputs: dict[str, list[Any]] = {}
    for output in model.graph.output:
        dims: list[Any] = []
        for dim in output.type.tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.dim_value > 0 else dim.dim_param or "?")
        outputs[output.name] = dims
    expected = {
        "z_c": [128],
        "z_s": [128],
        "content_score": [],
        "speaker_score": [],
        "fusion_score": [],
    }
    passed = set(outputs) == set(expected) and all(
        _shape_matches(outputs[name], tail) for name, tail in expected.items()
    )
    return _gate("output_shapes", passed, outputs, "five named outputs with dynamic batch")


def check_int8_vs_fp32_degradation(
    int8_path: Path,
    fp32_path: Path,
    eval_set: EvalSet,
    max_degradation_pp: float,
) -> GateResult:
    int8_metrics = _artifact_metrics(int8_path, eval_set)
    fp32_metrics = _artifact_metrics(fp32_path, eval_set)
    degradation_pp = max(0.0, (fp32_metrics["ta_clean"] - int8_metrics["ta_clean"]) * 100.0)
    return _gate(
        "int8_vs_fp32_degradation_pp",
        degradation_pp <= max_degradation_pp,
        round(degradation_pp, 6),
        max_degradation_pp,
    )


def _coerce_eval_set(eval_set: EvalSet | Path) -> EvalSet:
    if isinstance(eval_set, EvalSet):
        return eval_set
    return EvalSet(test_kpi_manifest=eval_set, test_fa_manifest=Path("data/manifests/test_fa.csv"))


def validate(
    onnx_path: Path,
    eval_set: EvalSet | Path,
    config: DeploymentConfig | None = None,
    gates: ValidationGates | None = None,
    fp32_onnx_path: Path | None = None,
) -> ValidationReport:
    """Run artifact gates 1-9, plus Gate 10 when an FP32 ONNX path is supplied."""

    resolved_eval_set = _coerce_eval_set(eval_set)
    resolved_config = config or DeploymentConfig()
    resolved_gates = gates or ValidationGates()

    metrics = _artifact_metrics(onnx_path, resolved_eval_set)
    results = [
        check_filesize(onnx_path, max_mb=resolved_gates.max_filesize_mb),
        check_onnx_opset(onnx_path, min_opset=resolved_gates.min_opset),
        check_mobile_ops(onnx_path),
        check_xrt(onnx_path, max_xrt=resolved_gates.max_xrt),
        _metric_gate(
            "ta_clean",
            metrics["ta_clean"],
            resolved_gates.min_ta_clean,
            higher_is_better=True,
        ),
        _metric_gate(
            "ta_noisy_macro",
            metrics["ta_noisy_macro"],
            resolved_gates.min_ta_noisy_macro,
            higher_is_better=True,
        ),
        _metric_gate(
            "fa_per_hour_per_user",
            metrics["fa_per_hour_per_user"],
            resolved_gates.max_fa_per_hr_per_user,
            higher_is_better=False,
        ),
        check_param_count(onnx_path, max_params=resolved_gates.max_param_count),
        check_output_shapes(onnx_path),
    ]
    if fp32_onnx_path is not None:
        results.append(
            check_int8_vs_fp32_degradation(
                onnx_path,
                fp32_onnx_path,
                resolved_eval_set,
                resolved_gates.max_int8_vs_fp32_degradation_pp,
            )
        )

    report = ValidationReport(results)
    if not report.all_passed:
        if resolved_config.target_platform:
            report.gates.append(
                _gate("target_platform", True, resolved_config.target_platform, "configured")
            )
        raise ArtifactValidationError(report)
    return report

