"""CLI entry point for running any single training stage.

Usage:
    python -m scripts.run_stage --stage 1
    python -m scripts.run_stage --stage 3 --resume-from checkpoints/stage2_dualhead.pt
    python -m scripts.run_stage --stage 4d
    python -m scripts.run_stage --stage 7 --resume-from checkpoints/stage6_final.pt
    python -m scripts.run_stage --production
    python -m scripts.run_stage --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

from solospeak.utils.config import SoloSpeakConfig
from solospeak.utils.seeding import seed_everything

STAGE_CONFIGS: dict[str, Path] = {
    "1": Path("configs/training/stage1_backbone_pretrain.yaml"),
    "2": Path("configs/training/stage2_dual_head.yaml"),
    "3": Path("configs/training/stage3_disentangle.yaml"),
    "4": Path("configs/training/stage4_robustness.yaml"),
    "4d": Path("configs/training/stage4d_hardq2_mining.yaml"),
    "5": Path("configs/training/stage5_fusion.yaml"),
    "6": Path("configs/training/stage6_final_eval.yaml"),
    "7": Path("configs/training/stage7_external_fa.yaml"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a SoloSpeak training stage")
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_CONFIGS.keys()),
        help="Stage to run: 1, 2, 3, 4, 4d, 5, 6, or 7",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run stages 1, 2, 3, 4, 4d, 5, 6, and 7 sequentially",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Run the full production pipeline including verification and corrected export",
    )
    parser.add_argument("--config", type=Path, help="Override config YAML path")
    parser.add_argument(
        "--resume-from",
        type=Path,
        dest="resume_from",
        help="Resume from a specific checkpoint",
    )
    parser.add_argument(
        "--external-common-voice-root",
        action="append",
        type=Path,
        default=[],
        help="Override Stage 7 Common Voice root. May be repeated.",
    )
    parser.add_argument(
        "--external-librispeech-root",
        action="append",
        type=Path,
        default=[],
        help="Override Stage 7 LibriSpeech root. May be repeated.",
    )
    parser.add_argument(
        "--external-background-noise-root",
        action="append",
        type=Path,
        default=[],
        help="Override Stage 7 background-noise root. May be repeated.",
    )
    parser.add_argument(
        "--external-urbansound-root",
        action="append",
        type=Path,
        default=[],
        help="Override Stage 7 UrbanSound8K root. May be repeated.",
    )
    args = parser.parse_args()

    if args.production:
        from scripts.run_production_pipeline import run_production_pipeline

        run_production_pipeline(
            config_path=args.config or Path("configs/training/production.yaml"),
            common_voice_roots=args.external_common_voice_root,
            librispeech_roots=args.external_librispeech_root,
            background_noise_roots=args.external_background_noise_root,
            urbansound_roots=args.external_urbansound_root,
        )
        return

    if not args.all and args.stage is None:
        parser.error("Specify --stage N, --all, or --production")

    stages = ["1", "2", "3", "4", "4d", "5", "6", "7"] if args.all else [args.stage]

    for stage_id in stages:
        if stage_id is None:
            continue
        config_path = args.config or STAGE_CONFIGS[stage_id]
        config = SoloSpeakConfig.from_yaml(config_path)
        if args.resume_from:
            config.training.resume_from = args.resume_from
        if args.external_common_voice_root:
            config.data.external_common_voice_roots = args.external_common_voice_root
        if args.external_librispeech_root:
            config.data.external_librispeech_roots = args.external_librispeech_root
        if args.external_background_noise_root:
            config.data.external_background_noise_roots = args.external_background_noise_root
        if args.external_urbansound_root:
            config.data.external_urbansound_roots = args.external_urbansound_root

        seed_everything(config.training.seed)

        if stage_id == "4d":
            from solospeak.training.stages.stage4d_hardq2_mining import Stage4D

            ckpt = Stage4D(config).run()
            print(f"Stage 4D complete → {ckpt}")
            continue

        from solospeak.training.trainer import Trainer

        trainer = Trainer(config)
        ckpt = trainer.run_stage(int(stage_id))
        print(f"Stage {stage_id} complete → {ckpt}")


if __name__ == "__main__":
    main()
