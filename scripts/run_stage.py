"""CLI entry point for running any single training stage.

Usage:
    python -m scripts.run_stage --stage 1
    python -m scripts.run_stage --stage 3 --resume-from checkpoints/stage2_dualhead.pt
    python -m scripts.run_stage --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

from solospeak.utils.config import SoloSpeakConfig
from solospeak.utils.seeding import seed_everything

STAGE_CONFIGS = {
    1: Path("configs/training/stage1_backbone_pretrain.yaml"),
    2: Path("configs/training/stage2_dual_head.yaml"),
    3: Path("configs/training/stage3_disentangle.yaml"),
    4: Path("configs/training/stage4_robustness.yaml"),
    5: Path("configs/training/stage5_fusion.yaml"),
    6: Path("configs/training/stage6_qat.yaml"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a SoloSpeak training stage")
    parser.add_argument("--stage", type=int, choices=range(1, 7),
                        help="Stage number to run (1–6)")
    parser.add_argument("--all", action="store_true", help="Run all stages 1–6 sequentially")
    parser.add_argument("--config", type=Path, help="Override config YAML path")
    parser.add_argument("--resume-from", type=Path, dest="resume_from",
                        help="Resume from a specific checkpoint")
    args = parser.parse_args()

    if not args.all and args.stage is None:
        parser.error("Specify --stage N or --all")

    stages = list(range(1, 7)) if args.all else [args.stage]

    for stage_id in stages:
        config_path = args.config or STAGE_CONFIGS[stage_id]
        config = SoloSpeakConfig.from_yaml(config_path)
        if args.resume_from:
            config.training.resume_from = args.resume_from

        seed_everything(config.training.seed)

        from solospeak.training.trainer import Trainer
        trainer = Trainer(config)
        ckpt = trainer.run_stage(stage_id)
        print(f"Stage {stage_id} complete → {ckpt}")


if __name__ == "__main__":
    main()
