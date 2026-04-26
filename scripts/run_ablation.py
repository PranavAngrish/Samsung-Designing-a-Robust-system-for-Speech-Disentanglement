"""CLI entry point for the Phase-4 ablation table."""

from __future__ import annotations

import argparse
from pathlib import Path

from solospeak.eval.ablations import format_ablation_table, run_all_ablations
from solospeak.utils.config import AblationConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or summarize SoloSpeak ablations")
    parser.add_argument("--config", type=Path, default=Path("configs/eval/ablations.yaml"))
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()

    config = AblationConfig.from_yaml(args.config)
    base_checkpoint = args.base_checkpoint if args.base_checkpoint else config.base_checkpoint
    seeds = args.seeds if args.seeds is not None else config.seeds
    results = run_all_ablations(
        base_checkpoint=base_checkpoint,
        output_dir=args.output_dir,
        seeds=seeds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "ablation_table.md"
    out.write_text(format_ablation_table(results))
    print(f"Ablation table written to {out}")


if __name__ == "__main__":
    main()
