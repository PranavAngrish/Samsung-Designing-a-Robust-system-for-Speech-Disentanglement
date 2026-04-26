"""CLI entry point for Phase-4 KPI, probe, and subgroup evaluation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from solospeak.eval.kpi_suite import result_to_jsonable, run_kpi_suite
from solospeak.eval.probes import run_probe_eval
from solospeak.eval.subgroup import format_subgroup_report, run_subgroup_eval
from solospeak.utils.config import EvalConfig, SoloSpeakConfig
from solospeak.utils.seeding import seed_everything


def _mean_numeric(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _aggregate_dicts(runs: list[dict[str, Any]]) -> dict[str, Any]:
    first = runs[0]
    summary: dict[str, Any] = {}
    for key, value in first.items():
        if key == "seed":
            continue
        if isinstance(value, int | float):
            summary[key] = _mean_numeric([float(run[key]) for run in runs])
        elif isinstance(value, Mapping):
            nested: dict[str, Any] = {}
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, int | float):
                    nested[nested_key] = _mean_numeric(
                        [float(run[key][nested_key]) for run in runs]
                    )
                else:
                    nested[nested_key] = nested_value
            summary[key] = nested
        else:
            summary[key] = value
    summary["seed_runs"] = runs
    summary["seeds"] = [run["seed"] for run in runs]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SoloSpeak Phase-4 evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/defaults.yaml"))
    parser.add_argument("--eval-config", type=Path, default=Path("configs/eval/full_kpi_suite.yaml"))
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("reports/kpi_final.json"))
    parser.add_argument("--probes-output", type=Path, default=Path("reports/probes.json"))
    parser.add_argument("--subgroup-output", type=Path, default=Path("reports/subgroup_report.md"))
    args = parser.parse_args()

    config = SoloSpeakConfig.from_yaml(args.config)
    eval_config = EvalConfig.from_yaml(args.eval_config)
    seeds = args.seeds or [config.training.seed]
    test_manifest = eval_config.test_manifests[0]
    fa_manifest = eval_config.test_manifests[1]

    runs: list[dict[str, Any]] = []
    for seed in seeds:
        seed_everything(seed)
        result = run_kpi_suite(
            checkpoint_path=args.checkpoint,
            config=config,
            test_manifest=test_manifest,
            fa_manifest=fa_manifest,
            seed=seed,
        )
        runs.append(result_to_jsonable(result))

    subgroup = run_subgroup_eval(args.checkpoint, test_manifest)
    kpi_payload = _aggregate_dicts(runs)
    kpi_payload["per_demographic"] = subgroup
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(kpi_payload, indent=2, sort_keys=True) + "\n")

    probes = run_probe_eval(
        args.checkpoint,
        config.data.manifests_dir / "dev_content.csv",
        config_path=args.config,
    )
    args.probes_output.parent.mkdir(parents=True, exist_ok=True)
    args.probes_output.write_text(json.dumps(probes, indent=2, sort_keys=True) + "\n")

    args.subgroup_output.parent.mkdir(parents=True, exist_ok=True)
    args.subgroup_output.write_text(format_subgroup_report(subgroup))

    print(f"KPI results written to {args.output}")
    print(f"Probe report written to {args.probes_output}")
    print(f"Subgroup report written to {args.subgroup_output}")


if __name__ == "__main__":
    main()
