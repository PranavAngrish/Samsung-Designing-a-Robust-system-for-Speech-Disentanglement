"""Seven-configuration ablation result runner/formatter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from solospeak.utils.config import AblationConfig
from solospeak.utils.types import AblationResult

ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    "full": {},
    "no_ortho": {"losses": {"orthogonality": 0.0}},
    "no_adv": {"losses": {"adversarial": 0.0}},
    "no_disent": {"losses": {"orthogonality": 0.0, "adversarial": 0.0}},
    "no_tts_enroll": {"enrollment": {"tts_n_variants": 0}},
    "no_gated_fusion": {"fusion": {"mode": "min"}},
    "no_curriculum": {"data": {"curriculum": False}},
}


def _load_full_metrics(output_dir: Path) -> dict[str, float]:
    path = output_dir / "kpi_final.json"
    if not path.exists():
        return {"ta_clean": 0.0, "ta_noisy_macro": 0.0, "fa_per_hour_per_user": 0.0}
    data = json.loads(path.read_text())
    if isinstance(data, list) and data:
        data = data[0]
    if "summary" in data:
        data = data["summary"]
    return {
        "ta_clean": float(data.get("ta_clean", 0.0)),
        "ta_noisy_macro": float(data.get("ta_noisy_macro", 0.0)),
        "fa_per_hour_per_user": float(data.get("fa_per_hour_per_user", 0.0)),
    }


def run_ablations(config: AblationConfig) -> AblationResult:
    """Return ablation rows, using existing result files when available.

    This function does not launch multi-day retraining. Missing ablation result files are
    marked ``not_run`` so reports are honest about compute that has not happened yet.
    """
    output_dir = Path("reports")
    full_metrics = _load_full_metrics(output_dir)
    results: AblationResult = {}
    for name in config.ablations:
        result_path = output_dir / "ablations" / f"{name}.json"
        if result_path.exists():
            row = json.loads(result_path.read_text())
            status = "complete"
        elif name == "full":
            row = dict(full_metrics)
            status = "baseline_from_kpi" if output_dir.joinpath("kpi_final.json").exists() else "not_run"
        else:
            row = {"ta_clean": 0.0, "ta_noisy_macro": 0.0, "fa_per_hour_per_user": 0.0}
            status = "not_run"
        row["seeds"] = len(config.seeds)
        row["status"] = status
        results[name] = row
    return results


def run_all_ablations(
    base_checkpoint: Path,
    output_dir: Path,
    seeds: list[int] | None = None,
) -> dict[str, dict[str, float | int | str]]:
    """Compatibility wrapper used by ``scripts/run_ablation.py``."""
    config = AblationConfig(base_checkpoint=base_checkpoint, seeds=seeds or [42])
    results = run_ablations(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    return results


def format_ablation_table(results: dict[str, dict[str, float | int | str]]) -> str:
    """Render ablation rows as the Phase-4 Markdown table."""
    full = results.get("full", {})
    full_clean = float(full.get("ta_clean", 0.0))
    full_noisy = float(full.get("ta_noisy_macro", 0.0))
    full_fa = float(full.get("fa_per_hour_per_user", 0.0))
    lines = [
        "| Ablation | Clean TA | Noisy Macro TA | FA/hr/user | Delta Clean | Delta Noisy | Delta FA | Seeds | Status |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|\n",
    ]
    for name in ABLATION_CONFIGS:
        row = results.get(name, {})
        clean = float(row.get("ta_clean", 0.0))
        noisy = float(row.get("ta_noisy_macro", 0.0))
        fa = float(row.get("fa_per_hour_per_user", 0.0))
        seeds = int(row.get("seeds", 0))
        status = str(row.get("status", "not_run"))
        lines.append(
            f"| {name} | {clean:.4f} | {noisy:.4f} | {fa:.4f} | "
            f"{clean - full_clean:+.4f} | {noisy - full_noisy:+.4f} | "
            f"{fa - full_fa:+.4f} | {seeds} | {status} |\n"
        )
    return "".join(lines)

