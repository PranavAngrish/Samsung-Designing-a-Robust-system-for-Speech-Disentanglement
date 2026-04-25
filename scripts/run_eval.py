"""CLI entry point for running the full KPI evaluation suite.

Usage:
    python -m scripts.run_eval --checkpoint checkpoints/latest.pt
    python -m scripts.run_eval --checkpoint checkpoints/stage5_fusion.pt --seeds 42 1337 2024
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from solospeak.utils.config import SoloSpeakConfig
from solospeak.utils.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SoloSpeak KPI evaluation suite")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/defaults.yaml"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    parser.add_argument("--output", type=Path, default=Path("reports/kpi_final.json"))
    args = parser.parse_args()

    config = SoloSpeakConfig.from_yaml(args.config)

    from solospeak.eval.kpi_suite import run_kpi_suite
    results = []
    for seed in args.seeds:
        seed_everything(seed)
        result = run_kpi_suite(
            checkpoint_path=args.checkpoint,
            config=config,
            test_manifest=Path("data/manifests/test_kpi.csv"),
            fa_manifest=Path("data/manifests/test_fa.csv"),
            seed=seed,
        )
        results.append(result.__dict__)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"KPI results written to {args.output}")


if __name__ == "__main__":
    main()
