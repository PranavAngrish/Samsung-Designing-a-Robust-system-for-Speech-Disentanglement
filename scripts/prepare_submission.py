"""Generate Phase-7 submission docs from reports and artifacts."""

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
        return (path.name, "missing", "missing")
    size_mb = path.stat().st_size / (1024.0 * 1024.0)
    return (path.name, f"{size_mb:.3f} MB", _sha256(path))


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
    kpi = _load_json(reports_dir / "kpi_final.json")
    probes = _load_json(reports_dir / "probes.json")
    validation = _load_json(artifacts_dir / "ValidationReport.json")
    int8_path = artifacts_dir / "solospeak_int8.onnx"
    size_mb = int8_path.stat().st_size / (1024.0 * 1024.0) if int8_path.exists() else 0.0
    return {
        "ta_clean": _pct(kpi.get("ta_clean")),
        "ta_noisy_macro": _pct(kpi.get("ta_noisy_macro")),
        "fa_per_hour_per_user": _float(kpi.get("fa_per_hour_per_user"), 2),
        "q2_rejection": _pct(kpi.get("q2_rejection")),
        "q3_rejection": _pct(kpi.get("q3_rejection")),
        "q4_rejection": _pct(kpi.get("q4_rejection")),
        "params_m": f"{float(kpi.get('param_count', 0.0)) / 1_000_000.0:.2f}M",
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
        reports_dir / "production_pipeline_summary.json",
        reports_dir / "stage7_joint_threshold_calibration_summary.json",
        reports_dir / "stage7_final_kpi_verification.json",
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
        ]
    )
    output.write_text("\n".join(lines) + "\n")


def _write_email(output: Path, metrics: dict[str, str], repo_url: str, demo_url: str) -> None:
    release_url = f"{repo_url}/releases/tag/v1.0.0-phase2"
    text = f"""# Submission Email

To: ennovatex.io@samsung.com

Subject: AX Hackathon Phase 2 Submission | 04 | Resonant

Hello Samsung ennovateX team,

Please find enclosed our Phase 2 submission for Problem #04 - Speech Disentanglement.

Team: Resonant
Participant: Pranav Angrish (pangrish_be22@thapar.edu)
Institute: Thapar Institute of Engineering & Technology

Attachments / links:
- Final report PDF: resonant-04-solospeak-final.pdf
- GitHub repository: `{repo_url}`
- Demo video: `{demo_url}`
- Trained artifacts: `{release_url}`

Headline results, with full tables in the report:
- TA Clean: {metrics["ta_clean"]} (MIN gate 0.92, TARGET 0.96)
- TA Noisy macro: {metrics["ta_noisy_macro"]} (MIN gate 0.80, TARGET 0.88)
- FA per hour per user: {metrics["fa_per_hour_per_user"]}
- Q2 imposter rejection: {metrics["q2_rejection"]}
- Q3 phonetic-neighbor rejection: {metrics["q3_rejection"]}
- Model size: {metrics["params_m"]} params, {metrics["int8_size_mb"]} MB INT8
- xRT local p95: {metrics["xrt_p95"]} (HARD gate 0.20, STRETCH 0.08)
- Disentanglement: speaker probe on z_c reduced by {metrics["speaker_probe_reduction"]}
- All 4 quadrants (Q1/Q2/Q3/Q4) explicitly evaluated

Important caveat: current generated numbers are smoke-run numbers unless replaced by a
full-data training and evaluation run before submission.

Looking forward to Phase 3.

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
| 7:00-8:00 | KPI Dashboard | Use numbers from `reports/kpi_final.json`; do not boost them. |
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
speaker match before firing. Current generated metrics are smoke-only unless a full-data
run replaces them.

## 2. Problem Analysis

Describe the four quadrants: Q1 accept, Q2 imposter, Q3 wrong word, Q4 background.

## 3. System Architecture

Cover log-mel extraction, residual CNN backbone, content/speaker heads, template
enrollment, fusion MLP, streaming hysteresis, ONNX export, and INT8 deployment.

## 4. Training Methodology

Document stages 1-6, loss terms, augmentations, smoke data vs full data, and seed count
({metrics["seed_count"]} in the current reports).

## 5. Scalability And Production Readiness

Summarize deployment gates, OTA packaging, rollback slot, local metrics, DP plan, and
threat model.

## 6. Evaluation Results

| Metric | Current Reported Value |
|---|---:|
| TA clean | {metrics["ta_clean"]} |
| TA noisy macro | {metrics["ta_noisy_macro"]} |
| FA/hr/user | {metrics["fa_per_hour_per_user"]} |
| Q2 rejection | {metrics["q2_rejection"]} |
| Q3 rejection | {metrics["q3_rejection"]} |
| Q4 rejection | {metrics["q4_rejection"]} |
| INT8 size | {metrics["int8_size_mb"]} MB |
| xRT local p95 | {metrics["xrt_p95"]} |

## 7. Ablation Analysis

Use `reports/ablation_table.md`. Disclose which rows are `not_run` and the number of
seeds.

## 8. Samsung Ecosystem Fit

Map SoloSpeak to Bixby, Buds, TVs, SmartThings, and shared household devices.

## 9. Honest Limitations

- Smoke metrics do not prove final accuracy.
- Placeholder VAD must be replaced for final demo.
- Replay and voice-clone resistance are measured baselines, not solved defenses.
- Optional demographic fairness metadata is currently unavailable in smoke manifests.

## 10. Appendix

Include KPI JSON, subgroup report, ablation table, validation report, threat model, and
release manifest.
"""
    output.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase-7 submission docs")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--repo-url", default="https://github.com/pranavangrish/solospeak")
    parser.add_argument("--demo-url", default="PASTE_UNLISTED_YOUTUBE_URL_HERE")
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
    print("Generated Phase-7 docs in docs/")


if __name__ == "__main__":
    main()
