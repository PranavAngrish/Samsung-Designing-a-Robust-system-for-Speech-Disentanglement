"""CLI entry point for the 7-configuration ablation study.

Usage:
    python -m scripts.run_ablation
    python -m scripts.run_ablation --base-checkpoint checkpoints/stage2_dualhead.pt
    python -m scripts.run_ablation --seeds 42 1337
"""

from __future__ import annotations

import argparse
from pathlib import Path

from solospeak.eval.ablations import run_all_ablations, format_ablation_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SoloSpeak ablation study")
    parser.add_argument("--base-checkpoint", type=Path,
                        default=Path("checkpoints/stage2_dualhead.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()

    results = run_all_ablations(
        base_checkpoint=args.base_checkpoint,
        output_dir=args.output_dir,
        seeds=args.seeds,
    )
    table = format_ablation_table(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "ablation_table.md"
    out.write_text(table)
    print(f"Ablation table written to {out}")


if __name__ == "__main__":
    main()
