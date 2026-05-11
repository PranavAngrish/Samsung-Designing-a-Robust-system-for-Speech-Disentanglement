"""Export, jointly calibrate, and correct the Stage-7 deployable."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize Stage7 deployable artifacts")
    parser.add_argument(
        "--stage7-checkpoint",
        type=Path,
        default=Path("checkpoints/stage7_fusion.pt"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/stage7_external_fa.yaml"),
    )
    parser.add_argument(
        "--final-checkpoint",
        type=Path,
        default=Path("checkpoints/stage7_final.pt"),
    )
    parser.add_argument(
        "--deployable",
        type=Path,
        default=Path("exports/solospeak_stage7_deployable.pt"),
    )
    parser.add_argument(
        "--internal-scores",
        type=Path,
        default=Path("reports/stage7_final_internal_scores.csv"),
    )
    parser.add_argument(
        "--external-scores",
        type=Path,
        default=Path("reports/stage7_external_fa_scores.csv"),
    )
    parser.add_argument(
        "--calibration-csv",
        type=Path,
        default=Path("reports/stage7_joint_threshold_calibration.csv"),
    )
    parser.add_argument(
        "--calibration-summary",
        type=Path,
        default=Path("reports/stage7_joint_threshold_calibration_summary.json"),
    )
    parser.add_argument(
        "--corrected-checkpoint",
        type=Path,
        default=Path("checkpoints/stage7_final_corrected.pt"),
    )
    parser.add_argument(
        "--corrected-deployable",
        type=Path,
        default=Path("exports/solospeak_stage7_deployable_corrected.pt"),
    )
    parser.add_argument("--skip-initial-export", action="store_true")
    args = parser.parse_args()

    from solospeak.deployment.export_stage7 import (
        export_corrected_stage7,
        export_stage7_deployable,
    )
    from solospeak.eval.threshold_calibration import calibrate_joint_threshold

    if not args.skip_initial_export:
        exported = export_stage7_deployable(
            stage7_checkpoint=args.stage7_checkpoint,
            final_checkpoint=args.final_checkpoint,
            deployable=args.deployable,
            config_path=args.config,
        )
        print(f"Stage7 deployable: {exported}")

    summary = calibrate_joint_threshold(
        args.internal_scores,
        args.external_scores,
        calibration_csv=args.calibration_csv,
        summary_json=args.calibration_summary,
    )
    corrected = export_corrected_stage7(
        old_deployable=args.deployable,
        corrected_checkpoint=args.corrected_checkpoint,
        corrected_deployable=args.corrected_deployable,
        calibration_summary=summary,
        config_path=args.config,
    )
    print(f"Joint calibration summary: {args.calibration_summary}")
    print(f"Corrected Stage7 deployable: {corrected}")


if __name__ == "__main__":
    main()
