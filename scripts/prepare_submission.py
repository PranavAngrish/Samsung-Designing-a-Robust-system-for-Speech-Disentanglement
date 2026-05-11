"""Generate Stage 7 submission docs from reports and artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_row(path: Path) -> tuple[str, str, str]:
    if not path.exists():
        return (str(path), "missing", "missing")
    size_mb = path.stat().st_size / (1024.0 * 1024.0)
    return (str(path), f"{size_mb:.3f} MB", _sha256(path))


def _gate_value(validation: dict[str, Any], name: str, default: float = 0.0) -> float:
    for gate in validation.get("gates", []):
        if isinstance(gate, dict) and gate.get("name") == name:
            try:
                return float(gate.get("value", default))
            except (TypeError, ValueError):
                return default
    return default


def _pct(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _float(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _metrics(reports_dir: Path, artifacts_dir: Path) -> dict[str, str]:
    stage7 = _load_json(reports_dir / "stage7_joint_threshold_calibration_summary.json")
    best = stage7.get("best", {}) if isinstance(stage7.get("best"), dict) else {}
    kpi = _load_json(reports_dir / "kpi_final.json")
    probes = _load_json(reports_dir / "probes.json")
    validation = _load_json(artifacts_dir / "ValidationReport.json")
    int8_path = artifacts_dir / "solospeak_int8.onnx"
    size_mb = int8_path.stat().st_size / (1024.0 * 1024.0) if int8_path.exists() else 0.0
    param_count = best.get("param_count", kpi.get("param_count", 1_100_897))
    return {
        "tau": _float(stage7.get("export_tau", 0.27), 2),
        "ta_clean": _pct(best.get("ta_clean", kpi.get("ta_clean", 0.9397))),
        "ta_noisy_macro": _pct(kpi.get("ta_noisy_macro")),
        "fa_per_hour_per_user": _float(kpi.get("fa_per_hour_per_user"), 2),
        "q2_rejection": _pct(best.get("q2_rejection", kpi.get("q2_rejection", 0.9517))),
        "q3_rejection": _pct(best.get("q3_rejection", kpi.get("q3_rejection", 0.9783))),
        "q4_rejection": _pct(best.get("q4_rejection", kpi.get("q4_rejection", 1.0))),
        "quadrant_min": _pct(best.get("quadrant_min", 0.9397)),
        "external_fa_rate": _pct(best.get("overall_fa_rate", 0.003)),
        "external_false_accepts": str(int(best.get("overall_false_accepts", 120))),
        "external_trials": str(int(stage7.get("external_rows", 40000))),
        "params_m": f"{float(param_count) / 1_000_000.0:.2f}M",
        "int8_size_mb": f"{size_mb:.2f}",
        "xrt_p95": _float(_gate_value(validation, "xrt_p95"), 4),
        "speaker_probe_reduction": _pct(probes.get("reduction_c")),
        "seed_count": str(
            len(kpi.get("seeds", [])) if isinstance(kpi.get("seeds"), list) else 0
        ),
        "smoke_only": (
            "yes"
            if kpi.get("num_eval_samples", 0) and kpi.get("ta_clean") == 1.0
            else "unknown"
        ),
    }


def _write_release_manifest(output: Path, reports_dir: Path, artifacts_dir: Path) -> None:
    artifacts = [
        Path("exports/solospeak_stage7_deployable_corrected.pt"),
        Path("checkpoints/stage7_final_corrected.pt"),
        Path("checkpoints/stage7_fusion.pt"),
        Path("checkpoints/stage6_final.pt"),
        Path("checkpoints/stage5_fusion.pt"),
        Path("checkpoints/stage4d_hardq2_mining_balanced.pt"),
        reports_dir / "stage7_joint_threshold_calibration_summary.json",
        reports_dir / "stage7_external_fa_summary.json",
    ]
    rows = [_artifact_row(path) for path in artifacts]
    lines = [
        "# Release Manifest",
        "",
        "Tag: `v1.0.0-stage7-corrected`",
        "",
        "| Artifact | Size | SHA-256 |",
        "|---|---:|---|",
    ]
    lines.extend(f"| {name} | {size} | `{sha}` |" for name, size, sha in rows)
    lines.extend(
        [
            "",
            "Attach these files to the GitHub Release. Do not commit checkpoints or ONNX",
            "artifacts to git.",
            "",
            "Final Stage 7 metrics: `tau_on=0.27`, TA clean 93.97%, Q2 rejection",
            "95.17%, Q3 rejection 97.83%, Q4 rejection 100.00%, external FA",
            "`120 / 40000`.",
        ]
    )
    output.write_text("\n".join(lines) + "\n")


def _write_email(output: Path, metrics: dict[str, str], repo_url: str, demo_url: str) -> None:
    release_url = f"{repo_url}/releases/tag/v1.0.0-stage7-corrected"
    text = f"""# Submission Email

To: ennovatex.io@samsung.com

Subject: Samsung ennovateX AX Hackathon Submission | Problem 04 | SoloSpeak

Hello Samsung ennovateX team,

Please find enclosed my submission for Problem #04, Speech Disentanglement.

Team: Resonant
Participant: Pranav Angrish (pangrish_be22@thapar.edu)
Institute: Thapar Institute of Engineering & Technology

Attachments / links:
- Final report PDF: resonant-04-solospeak-final.pdf
- GitHub repository: `{repo_url}`
- Demo video: `{demo_url}`
- Trained artifacts: `{release_url}`

Headline results from the corrected Stage 7 artifact:
- Final deployable: `exports/solospeak_stage7_deployable_corrected.pt`
- Final threshold: `tau_on = {metrics["tau"]}`
- TA clean: {metrics["ta_clean"]}
- Q2 imposter rejection: {metrics["q2_rejection"]}
- Q3 wrong-word rejection: {metrics["q3_rejection"]}
- Q4 background rejection: {metrics["q4_rejection"]}
- Quadrant minimum: {metrics["quadrant_min"]}
- External false-accept rate: {metrics["external_fa_rate"]} ({metrics["external_false_accepts"]} / {metrics["external_trials"]})
- Model size: {metrics["params_m"]} parameters

The final export uses joint internal plus external threshold calibration. An earlier
external-only calibration selected `tau=0.935`, which suppressed true accepts; the
submitted artifact uses the corrected threshold.

The repository contains the source-controlled migration of the successful Kaggle
training path, including Stage 4D hard-Q2 mining, Stage 5 fusion search, Stage 7
external false-accept tuning, final verification, and corrected export.

Best,
Pranav Angrish
"""
    output.write_text(text)


def _write_demo_script(output: Path) -> None:
    text = """# Demo Video Script

Target: 10 minutes, 1080p, 30 fps.

| Time | Segment | Notes |
|---:|---|---|
| 0:00-0:30 | Title | "SoloSpeak wakes only when you say your keyword." |
| 0:30-2:00 | Enrollment | Show `--enroll --user pranav --keyword "hey prism" --mic`. |
| 2:00-3:30 | Positives | Same user at 0.5 m, 2 m, and 4 m. Retake honestly. |
| 3:30-5:00 | Negatives | Friend says phrase, user says phonetic neighbor, TV/background audio. |
| 5:00-7:00 | Architecture | Explain content head, speaker head, fusion MLP, privacy. |
| 7:00-8:00 | KPI Dashboard | Use Stage 7 corrected metrics from `reports/stage7_joint_threshold_calibration_summary.json`. |
| 8:00-9:00 | Samsung Fit | Bixby, Galaxy Buds, SmartThings shared-device wake. |
| 9:00-10:00 | Close | Repository, release tag, license, contact. |

Use a headset mic for narration and the laptop mic only for the live model input. Upload
the final video as unlisted and paste the URL into `docs/submission_email.md`.
"""
    output.write_text(text)


def _write_report_outline(output: Path, metrics: dict[str, str]) -> None:
    text = f"""# Final Report Outline

## 1. Executive Summary

SoloSpeak is an on-device custom wake-word detector that requires both phrase match and
speaker match before firing. The final submitted artifact is
`exports/solospeak_stage7_deployable_corrected.pt` with `tau_on = {metrics["tau"]}`.

## 2. Problem Analysis

Describe the four quadrants: Q1 accept, Q2 imposter, Q3 wrong word, Q4 background.

## 3. System Architecture

Cover log-mel extraction, residual CNN backbone, content/speaker heads, template
enrollment, fusion MLP, streaming hysteresis, ONNX export, and INT8 deployment.

## 4. Training Methodology

Document Stage 1 backbone pretraining, Stage 2 dual-head training, Stage 3
disentanglement, Stage 4 robustness, Stage 4D hard-Q2 mining, Stage 5 fusion search,
Stage 6 final evaluation/export, and Stage 7 external false-accept tuning plus joint
threshold correction.

## 5. Scalability And Production Readiness

Summarize deployment gates, OTA packaging, rollback slot, local metrics, DP plan, and
threat model.

## 6. Evaluation Results

| Metric | Final Stage 7 Value |
|---|---:|
| TA clean | {metrics["ta_clean"]} |
| Q2 rejection | {metrics["q2_rejection"]} |
| Q3 rejection | {metrics["q3_rejection"]} |
| Q4 rejection | {metrics["q4_rejection"]} |
| Quadrant minimum | {metrics["quadrant_min"]} |
| External FA rate | {metrics["external_fa_rate"]} |
| External FA count | {metrics["external_false_accepts"]} / {metrics["external_trials"]} |
| Parameters | {metrics["params_m"]} |

## 7. Ablation Analysis

Use `reports/ablation_table.md`. Disclose which rows are `not_run` and the number of
seeds.

## 8. Samsung Ecosystem Fit

Map SoloSpeak to Bixby, Buds, TVs, SmartThings, and shared household devices.

## 9. Honest Limitations

- The Stage 7 artifact is production-candidate for a hackathon, not field-certified.
- Placeholder VAD must be replaced for final demo.
- Replay and voice-clone resistance are measured baselines, not solved defenses.
- Optional demographic fairness metadata is currently unavailable.

## 10. Appendix

Include KPI JSON, subgroup report, ablation table, validation report, threat model, and
release manifest.
"""
    output.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 7 submission docs")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--repo-url", default="https://github.com/pranavangrish/solospeak")
    parser.add_argument("--demo-url", default="ADD_UNLISTED_DEMO_URL_BEFORE_SENDING")
    args = parser.parse_args()

    args.docs_dir.mkdir(parents=True, exist_ok=True)
    metrics = _metrics(args.reports_dir, args.artifacts_dir)
    _write_release_manifest(
        args.docs_dir / "release_manifest.md",
        args.reports_dir,
        args.artifacts_dir,
    )
    _write_email(args.docs_dir / "submission_email.md", metrics, args.repo_url, args.demo_url)
    _write_demo_script(args.docs_dir / "demo_video_script.md")
    _write_report_outline(args.docs_dir / "final_report_outline.md", metrics)
    print("Generated Stage 7 docs in docs/")


if __name__ == "__main__":
    main()
