"""Notebook-authoritative threshold calibration for Stage 7.

The final Kaggle notebook discovered that calibrating the deployable threshold from
external false-accept scores alone selected ``tau=0.935`` and destroyed true accepts.
The corrected export uses a joint sweep over internal Q1/Q2/Q3/Q4 scores and external
FA scores. This module is the source version of ``ta-test.ipynb`` cell 21.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _frac(mask: np.ndarray, values: np.ndarray) -> float:
    denom = int(mask.sum())
    if denom <= 0:
        return 0.0
    return float(values[mask].mean())


def calibrate_joint_threshold(
    internal_scores: Path,
    external_fa_scores: Path,
    *,
    calibration_csv: Path = Path("reports/stage7_joint_threshold_calibration.csv"),
    summary_json: Path = Path("reports/stage7_joint_threshold_calibration_summary.json"),
    tau_grid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Select Stage-7 ``tau`` using both internal quadrants and external FA scores.

    ``internal_scores`` must contain at least ``quadrant`` and ``prob`` columns.
    ``external_fa_scores`` must contain at least ``source_dataset`` and ``prob`` columns.
    The output schema and ranking match the corrected final Kaggle cell.
    """

    if not internal_scores.exists():
        raise FileNotFoundError(f"Missing internal score file: {internal_scores}")
    if not external_fa_scores.exists():
        raise FileNotFoundError(f"Missing external FA score file: {external_fa_scores}")

    internal_df = pd.read_csv(internal_scores)
    fa_df = pd.read_csv(external_fa_scores)

    required_internal = {"quadrant", "prob"}
    required_external = {"source_dataset", "prob"}
    missing_internal = required_internal - set(internal_df.columns)
    missing_external = required_external - set(fa_df.columns)
    if missing_internal:
        raise ValueError(f"Internal scores missing columns: {sorted(missing_internal)}")
    if missing_external:
        raise ValueError(f"External FA scores missing columns: {sorted(missing_external)}")

    internal_prob = internal_df["prob"].to_numpy(dtype=np.float32)
    fa_prob = fa_df["prob"].to_numpy(dtype=np.float32)
    quadrants = internal_df["quadrant"].astype(str).to_numpy()
    fa_sources = fa_df["source_dataset"].astype(str).to_numpy()

    q1 = quadrants == "Q1_accept"
    q2 = quadrants == "Q2_imposter"
    q3 = quadrants == "Q3_wrong_word"
    q4 = quadrants == "Q4_background"
    grid = tau_grid if tau_grid is not None else np.linspace(0.05, 0.95, 181)

    rows: list[dict[str, Any]] = []
    for tau_value in grid:
        tau = float(tau_value)
        internal_accept = internal_prob >= tau
        fa_accept = fa_prob >= tau

        ta_clean = _frac(q1, internal_accept.astype(np.float32))
        q2_rejection = _frac(q2, (~internal_accept).astype(np.float32))
        q3_rejection = _frac(q3, (~internal_accept).astype(np.float32))
        q4_rejection = _frac(q4, (~internal_accept).astype(np.float32))

        quadrant_min = min(ta_clean, q2_rejection, q3_rejection, q4_rejection)
        quadrant_mean = (ta_clean + q2_rejection + q3_rejection + q4_rejection) / 4.0
        overall_false_accepts = int(fa_accept.sum())
        overall_fa_rate = float(fa_accept.mean()) if len(fa_accept) else 0.0

        source_metrics: dict[str, Any] = {}
        for source in sorted(set(fa_sources.tolist())):
            src_mask = fa_sources == source
            src_accept = fa_accept[src_mask]
            source_metrics[f"{source}_rows"] = int(src_mask.sum())
            source_metrics[f"{source}_false_accepts"] = int(src_accept.sum())
            source_metrics[f"{source}_fa_rate"] = (
                float(src_accept.mean()) if len(src_accept) else 0.0
            )

        production_candidate = (
            ta_clean >= 0.92
            and q2_rejection >= 0.95
            and q3_rejection >= 0.90
            and q4_rejection >= 0.99
            and overall_fa_rate <= 0.005
            and source_metrics.get("common_voice_fa_rate", 1.0) <= 0.005
            and source_metrics.get("librispeech_fa_rate", 1.0) <= 0.005
            and source_metrics.get("urbansound8k_fa_rate", 1.0) <= 0.005
            and source_metrics.get("background_noise_fa_rate", 0.0) <= 0.005
        )

        demo_candidate = (
            ta_clean >= 0.90
            and q2_rejection >= 0.90
            and q3_rejection >= 0.90
            and q4_rejection >= 0.95
            and quadrant_min >= 0.90
            and overall_fa_rate <= 0.01
        )

        selection_score = (
            10000.0 * float(production_candidate)
            + 1000.0 * float(demo_candidate)
            + 100.0 * quadrant_min
            + 20.0 * ta_clean
            + 10.0 * q2_rejection
            + 5.0 * q3_rejection
            + 5.0 * q4_rejection
            - 500.0 * overall_fa_rate
        )

        rows.append(
            {
                "tau": tau,
                "ta_clean": float(ta_clean),
                "q2_rejection": float(q2_rejection),
                "q3_rejection": float(q3_rejection),
                "q4_rejection": float(q4_rejection),
                "quadrant_min": float(quadrant_min),
                "quadrant_mean": float(quadrant_mean),
                "overall_false_accepts": overall_false_accepts,
                "overall_fa_rate": float(overall_fa_rate),
                "production_candidate": bool(production_candidate),
                "demo_candidate": bool(demo_candidate),
                "selection_score": float(selection_score),
                **source_metrics,
            }
        )

    calib_df = pd.DataFrame(rows)
    calibration_csv.parent.mkdir(parents=True, exist_ok=True)
    calib_df.to_csv(calibration_csv, index=False)

    production_candidates = calib_df[calib_df["production_candidate"] == True]
    demo_candidates = calib_df[calib_df["demo_candidate"] == True]
    sort_by = ["quadrant_min", "ta_clean", "q2_rejection", "overall_fa_rate", "tau"]
    sort_ascending = [False, False, False, True, True]

    if len(production_candidates):
        best = production_candidates.sort_values(
            by=sort_by,
            ascending=sort_ascending,
        ).iloc[0].to_dict()
        export_mode = "production_candidate"
        best_production_candidate = _to_jsonable(best)
    elif len(demo_candidates):
        best = demo_candidates.sort_values(
            by=sort_by,
            ascending=sort_ascending,
        ).iloc[0].to_dict()
        export_mode = "demo_candidate"
        best_production_candidate = None
    else:
        best = calib_df.sort_values(by=["selection_score"], ascending=[False]).iloc[0].to_dict()
        export_mode = "best_available_not_passing"
        best_production_candidate = None

    summary = {
        "export_tau": float(best["tau"]),
        "export_mode": export_mode,
        "best": _to_jsonable(best),
        "best_production_candidate": best_production_candidate,
        "production_gates": {
            "ta_clean_min": 0.92,
            "q2_rejection_min": 0.95,
            "q3_rejection_min": 0.90,
            "q4_rejection_min": 0.99,
            "overall_external_fa_rate_max": 0.005,
            "common_voice_fa_rate_max": 0.005,
            "librispeech_fa_rate_max": 0.005,
            "background_noise_fa_rate_max": 0.005,
            "urbansound8k_fa_rate_max": 0.005,
        },
        "n_production_candidates": int(len(production_candidates)),
        "n_demo_candidates": int(len(demo_candidates)),
        "calibration_csv": str(calibration_csv),
        "internal_scores": str(internal_scores),
        "external_fa_scores": str(external_fa_scores),
        "internal_rows": int(len(internal_df)),
        "internal_quadrants": dict(Counter(internal_df["quadrant"].astype(str))),
        "external_rows": int(len(fa_df)),
        "external_sources": dict(Counter(fa_df["source_dataset"].astype(str))),
        "note": (
            "This tau was selected jointly using internal Q1/Q2/Q3/Q4 scores and "
            "external FA scores. Unlike the previous tau=0.935 export, this preserves "
            "true accepts."
        ),
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(_to_jsonable(summary), indent=2, sort_keys=True) + "\n")
    return _to_jsonable(summary)
