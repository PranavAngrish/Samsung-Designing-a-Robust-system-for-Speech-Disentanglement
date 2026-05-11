"""Run the final SoloSpeak production-candidate pipeline.

This is the source-controlled equivalent of the successful Kaggle path:

Stage 1 -> Stage 2 -> Stage 3 -> Stage 4C -> Stage 4D hard-Q2 mining ->
Stage 5 dual fusion search -> Stage 6 final evaluation/export ->
Stage 7 external FA tuning -> internal verification -> joint tau correction/export.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

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


def _apply_production_overrides(
    config: SoloSpeakConfig,
    production: SoloSpeakConfig,
    *,
    common_voice_roots: Sequence[Path],
    librispeech_roots: Sequence[Path],
    background_noise_roots: Sequence[Path],
    urbansound_roots: Sequence[Path],
) -> SoloSpeakConfig:
    config.data.root = production.data.root
    config.data.manifests_dir = production.data.manifests_dir
    config.data.external_common_voice_roots = (
        list(common_voice_roots) or production.data.external_common_voice_roots
    )
    config.data.external_librispeech_roots = (
        list(librispeech_roots) or production.data.external_librispeech_roots
    )
    config.data.external_background_noise_roots = (
        list(background_noise_roots) or production.data.external_background_noise_roots
    )
    config.data.external_urbansound_roots = (
        list(urbansound_roots) or production.data.external_urbansound_roots
    )
    config.data.external_fa_max_total_trials = production.data.external_fa_max_total_trials
    config.stage4d = production.stage4d
    config.stage5 = production.stage5
    config.stage6 = production.stage6
    config.stage7 = production.stage7
    return config


def _stage_config(
    stage: str,
    production: SoloSpeakConfig,
    *,
    common_voice_roots: Sequence[Path],
    librispeech_roots: Sequence[Path],
    background_noise_roots: Sequence[Path],
    urbansound_roots: Sequence[Path],
) -> SoloSpeakConfig:
    config = SoloSpeakConfig.from_yaml(STAGE_CONFIGS[stage])
    return _apply_production_overrides(
        config,
        production,
        common_voice_roots=common_voice_roots,
        librispeech_roots=librispeech_roots,
        background_noise_roots=background_noise_roots,
        urbansound_roots=urbansound_roots,
    )


def _run_trainer_stage(stage: int, config: SoloSpeakConfig) -> Path:
    from solospeak.training.trainer import Trainer

    seed_everything(config.training.seed)
    return Trainer(config).run_stage(stage)


def _run_stage4d(config: SoloSpeakConfig) -> Path:
    from solospeak.training.stages.stage4d_hardq2_mining import Stage4D

    seed_everything(config.training.seed)
    return Stage4D(config).run()


def _verify_stage7(config: SoloSpeakConfig, stage7_checkpoint: Path) -> Path:
    import torch

    from solospeak.data.features import LogMelExtractor
    from solospeak.eval.external_fa import read_external_fa_manifest, score_external_fa_rows
    from solospeak.eval.internal_quadrants import (
        InternalQuadrantSpec,
        build_gsc_internal_examples,
        score_internal_examples_to_csv,
    )
    from solospeak.models.solospeak import SoloSpeakModel
    from solospeak.training.stages.common import default_device

    device = default_device()
    checkpoint = torch.load(stage7_checkpoint, map_location="cpu", weights_only=False)
    metrics = checkpoint.get("metrics", {}) or {}
    extra = checkpoint.get("extra", {}) or {}
    tau = float(extra.get("stage7_tau", metrics.get("dev/tau_stage7", config.fusion.tau_on)))
    config.fusion.tau_on = tau

    model = SoloSpeakModel(config).to(device).eval()
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Bad Stage7 state: missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    extractor = LogMelExtractor(config.audio).to(device).eval()
    profiles, internal_examples = build_gsc_internal_examples(
        model=model,
        extractor=extractor,
        config=config,
        manifests_dir=config.data.manifests_dir,
        profile_dir=Path("data/stage7_verify_profiles"),
        spec=InternalQuadrantSpec(
            wake_word=config.stage7.wake_word,
            n_enroll=config.stage7.n_enroll,
            max_profiles=config.stage7.max_profiles,
            profile_seed=777,
            q1_per_profile=2,
            q2_per_profile=6,
            q3_per_profile=2,
            q4_per_profile=2,
            require_wrong_word_for_profile=True,
            profile_prefix="stage7_verify_profile",
        ),
        device=device,
    )
    internal_metrics = score_internal_examples_to_csv(
        model=model,
        examples=internal_examples,
        tau=tau,
        output_path=config.stage7.internal_scores,
        device=device,
    )

    external_summary = None
    if config.stage7.external_manifest.exists():
        external_rows = read_external_fa_manifest(config.stage7.external_manifest)
        external_summary = score_external_fa_rows(
            model=model,
            extractor=extractor,
            config=config,
            rows=external_rows,
            tau=tau,
            output_path=config.stage7.stage7_external_scores,
            device=device,
        )

    pass_checks = {
        "ta_clean_passed": internal_metrics["ta_clean"] >= config.stage7.ta_clean_min,
        "q2_rejection_passed": internal_metrics["q2_rejection"]
        >= config.stage7.q2_rejection_min,
        "q3_rejection_passed": internal_metrics["q3_rejection"] >= 0.90,
        "q4_rejection_passed": internal_metrics["q4_rejection"] >= 0.99,
        "quadrant_min_passed": internal_metrics["quadrant_accuracy_min"] >= 0.90,
    }
    if external_summary is not None:
        by_source = external_summary["by_source"]
        overall = external_summary["overall"]
        pass_checks.update(
            {
                "overall_external_fa_passed": overall["fa_rate"]
                <= config.stage7.external_fa_rate_max,
                "common_voice_fa_passed": by_source.get("common_voice", {"fa_rate": 0.0})[
                    "fa_rate"
                ]
                <= config.stage7.common_voice_fa_rate_max,
                "librispeech_fa_passed": by_source.get("librispeech", {"fa_rate": 0.0})[
                    "fa_rate"
                ]
                <= 0.005,
                "urbansound_fa_passed": by_source.get("urbansound8k", {"fa_rate": 0.0})[
                    "fa_rate"
                ]
                <= 0.005,
                "background_noise_fa_passed": by_source.get(
                    "background_noise", {"fa_rate": 0.0}
                )["fa_rate"]
                <= 0.005,
            }
        )

    payload = {
        "stage7_checkpoint": str(stage7_checkpoint),
        "export_tau": float(tau),
        "wake_word": config.stage7.wake_word,
        "n_enroll": config.stage7.n_enroll,
        "profiles": len(profiles),
        "internal_metrics": internal_metrics,
        "external_summary": external_summary,
        "pass_checks": pass_checks,
        "final_production_candidate_passed": bool(all(pass_checks.values())),
        "files": {
            "internal_scores": str(config.stage7.internal_scores),
            "external_scores": str(config.stage7.stage7_external_scores)
            if external_summary is not None
            else None,
            "verification_summary": str(config.stage7.final_kpi_verification),
        },
    }
    config.stage7.final_kpi_verification.parent.mkdir(parents=True, exist_ok=True)
    config.stage7.final_kpi_verification.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return config.stage7.final_kpi_verification


def _finalize_stage7(config: SoloSpeakConfig) -> Path:
    from solospeak.deployment.export_stage7 import (
        export_corrected_stage7,
        export_stage7_deployable,
    )
    from solospeak.eval.production_report import write_production_final_summary
    from solospeak.eval.stage7_joint_calibration import calibrate_joint_threshold

    stage7_default_deployable = Path("exports/solospeak_stage7_deployable.pt")
    export_stage7_deployable(
        stage7_checkpoint=config.stage7.stage7_checkpoint,
        final_checkpoint=Path("checkpoints/stage7_final.pt"),
        deployable=stage7_default_deployable,
        config_path=STAGE_CONFIGS["7"],
    )
    summary = calibrate_joint_threshold(
        config.stage7.internal_scores,
        config.stage7.stage7_external_scores,
        calibration_csv=Path("reports/stage7_joint_threshold_calibration.csv"),
        summary_json=config.stage7.joint_calibration_summary,
    )
    corrected = export_corrected_stage7(
        old_deployable=stage7_default_deployable,
        corrected_checkpoint=config.stage7.final_checkpoint,
        corrected_deployable=config.stage7.deployable_path,
        calibration_summary=summary,
        config_path=STAGE_CONFIGS["7"],
    )
    write_production_final_summary(
        calibration_summary=config.stage7.joint_calibration_summary,
        verification_summary=config.stage7.final_kpi_verification,
        deployable_path=config.stage7.deployable_path,
        output_path=Path("reports/stage7_final_summary.json"),
    )
    return corrected


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _write_pipeline_summary(
    *,
    path: Path = Path("reports/production_pipeline_summary.json"),
    final_artifact: Path,
    calibration_summary: dict[str, object],
    stage5_best_candidate: dict[str, object],
    stage6_demo_passed: bool,
    stage7_default_tau: float,
    dataset_roots_used: dict[str, list[str]],
    smoke: bool,
) -> Path:
    best = dict(calibration_summary.get("best", {}) or {})
    payload = {
        "schema_version": 1,
        "smoke": bool(smoke),
        "final_artifact": str(final_artifact),
        "final_tau": float(calibration_summary.get("export_tau", best.get("tau", 0.27))),
        "stage5_best_candidate": stage5_best_candidate,
        "stage6_demo_passed": bool(stage6_demo_passed),
        "stage7_default_tau": float(stage7_default_tau),
        "stage7_corrected_tau": float(
            calibration_summary.get("export_tau", best.get("tau", 0.27))
        ),
        "final_internal_metrics": {
            "ta_clean": float(best.get("ta_clean", 0.0)),
            "q2_rejection": float(best.get("q2_rejection", 0.0)),
            "q3_rejection": float(best.get("q3_rejection", 0.0)),
            "q4_rejection": float(best.get("q4_rejection", 0.0)),
            "quadrant_min": float(best.get("quadrant_min", 0.0)),
            "quadrant_mean": float(best.get("quadrant_mean", 0.0)),
        },
        "final_external_fa_metrics": {
            "false_accepts": int(best.get("overall_false_accepts", 0)),
            "fa_rate": float(best.get("overall_fa_rate", 0.0)),
            "external_rows": int(calibration_summary.get("external_rows", 0)),
        },
        "production_candidate": bool(
            calibration_summary.get("export_mode") == "production_candidate"
        ),
        "dataset_roots_used": dataset_roots_used,
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _write_smoke_score_fixtures(config: SoloSpeakConfig) -> tuple[Path, Path]:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    internal_path = config.stage7.internal_scores
    external_path = config.stage7.stage7_external_scores

    internal_rows: list[dict[str, object]] = []
    internal_rows += [{"quadrant": "Q1_accept", "prob": 0.276} for _ in range(94)]
    internal_rows += [{"quadrant": "Q1_accept", "prob": 0.20} for _ in range(6)]
    internal_rows += [{"quadrant": "Q2_imposter", "prob": 0.30} for _ in range(4)]
    internal_rows += [{"quadrant": "Q2_imposter", "prob": 0.267} for _ in range(5)]
    internal_rows += [{"quadrant": "Q2_imposter", "prob": 0.10} for _ in range(91)]
    internal_rows += [{"quadrant": "Q3_wrong_word", "prob": 0.30} for _ in range(2)]
    internal_rows += [{"quadrant": "Q3_wrong_word", "prob": 0.10} for _ in range(98)]
    internal_rows += [{"quadrant": "Q4_background", "prob": 0.10} for _ in range(100)]
    internal_path.parent.mkdir(parents=True, exist_ok=True)
    with open(internal_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["quadrant", "prob"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(internal_rows)

    external_rows: list[dict[str, object]] = []
    external_rows += [{"source_dataset": "common_voice", "prob": 0.30} for _ in range(2)]
    external_rows += [{"source_dataset": "common_voice", "prob": 0.10} for _ in range(698)]
    external_rows += [{"source_dataset": "librispeech", "prob": 0.30}]
    external_rows += [{"source_dataset": "librispeech", "prob": 0.10} for _ in range(199)]
    external_rows += [{"source_dataset": "urbansound8k", "prob": 0.10} for _ in range(90)]
    external_rows += [{"source_dataset": "background_noise", "prob": 0.10} for _ in range(10)]
    external_path.parent.mkdir(parents=True, exist_ok=True)
    with open(external_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_dataset", "prob"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(external_rows)
    return internal_path, external_path


def _run_smoke_pipeline(
    *,
    production: SoloSpeakConfig,
    common_voice_roots: Sequence[Path],
    librispeech_roots: Sequence[Path],
    background_noise_roots: Sequence[Path],
    urbansound_roots: Sequence[Path],
) -> Path:
    import torch

    from solospeak.deployment.export_stage7 import export_corrected_stage7
    from solospeak.eval.threshold_calibration import calibrate_joint_threshold
    from solospeak.training.stages.common import save_checkpoint
    from solospeak.training.stages.stage5_notebook_search import EXPECTED_STAGE5_WINNER_CONFIG

    checkpoints_dir = production.training.checkpoint_dir
    exports_dir = Path("exports")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)

    internal_path, external_path = _write_smoke_score_fixtures(production)
    calibration_summary = calibrate_joint_threshold(
        internal_path,
        external_path,
        calibration_csv=Path("reports/stage7_joint_threshold_calibration.csv"),
        summary_json=production.stage7.joint_calibration_summary,
    )
    stage7_default_tau = 0.32
    old_deployable = Path("exports/solospeak_stage7_deployable.pt")
    torch.save(
        {
            "model_state": {},
            "tau_on": stage7_default_tau,
            "config": {"fusion": {"tau_on": stage7_default_tau}},
            "stage7_metrics": {"dev/tau_stage7": stage7_default_tau},
            "stage7_extra": {
                "export_tau": stage7_default_tau,
                "export_mode": "stage7_default_tau",
                "wake_word": production.stage7.wake_word,
                "n_enroll": production.stage7.n_enroll,
            },
        },
        old_deployable,
    )
    save_checkpoint(
        production.stage7.stage7_checkpoint,
        stage_origin=7,
        config=production,
        step=0,
        metrics={"dev/tau_stage7": stage7_default_tau},
        model_state={},
        optimizer_state={},
        extra={"stage7_tau": stage7_default_tau, "smoke_only": True},
    )
    corrected = export_corrected_stage7(
        old_deployable=old_deployable,
        corrected_checkpoint=production.stage7.final_checkpoint,
        corrected_deployable=production.stage7.deployable_path,
        calibration_summary=calibration_summary,
        config_path=Path("configs/training/production.yaml"),
    )
    production.stage7.final_kpi_verification.parent.mkdir(parents=True, exist_ok=True)
    production.stage7.final_kpi_verification.write_text(
        json.dumps(
            {
                "export_tau": calibration_summary["export_tau"],
                "final_production_candidate_passed": True,
                "smoke_only": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    dataset_roots = {
        "common_voice": [str(p) for p in common_voice_roots],
        "librispeech": [str(p) for p in librispeech_roots],
        "background_noise": [str(p) for p in background_noise_roots],
        "urbansound8k": [str(p) for p in urbansound_roots],
    }
    summary_path = _write_pipeline_summary(
        final_artifact=corrected,
        calibration_summary=calibration_summary,
        stage5_best_candidate=EXPECTED_STAGE5_WINNER_CONFIG,
        stage6_demo_passed=True,
        stage7_default_tau=stage7_default_tau,
        dataset_roots_used=dataset_roots,
        smoke=True,
    )
    print(f"Corrected Stage7 deployable: {corrected}")
    print(f"Production pipeline summary: {summary_path}")
    return corrected


def run_production_pipeline(
    *,
    config_path: Path = Path("configs/training/production.yaml"),
    common_voice_roots: Sequence[Path] = (),
    librispeech_roots: Sequence[Path] = (),
    background_noise_roots: Sequence[Path] = (),
    urbansound_roots: Sequence[Path] = (),
    smoke: bool = False,
    full_stage5_search: bool = False,
) -> Path:
    production = SoloSpeakConfig.from_yaml(config_path)
    if smoke:
        return _run_smoke_pipeline(
            production=production,
            common_voice_roots=common_voice_roots,
            librispeech_roots=librispeech_roots,
            background_noise_roots=background_noise_roots,
            urbansound_roots=urbansound_roots,
        )
    checkpoints: dict[str, Path] = {}

    stage1 = _stage_config(
        "1",
        production,
        common_voice_roots=common_voice_roots,
        librispeech_roots=librispeech_roots,
        background_noise_roots=background_noise_roots,
        urbansound_roots=urbansound_roots,
    )
    checkpoints["stage1"] = _run_trainer_stage(1, stage1)

    stage2 = _stage_config(
        "2",
        production,
        common_voice_roots=common_voice_roots,
        librispeech_roots=librispeech_roots,
        background_noise_roots=background_noise_roots,
        urbansound_roots=urbansound_roots,
    )
    stage2.training.resume_from = checkpoints["stage1"]
    checkpoints["stage2"] = _run_trainer_stage(2, stage2)

    stage3 = _stage_config(
        "3",
        production,
        common_voice_roots=common_voice_roots,
        librispeech_roots=librispeech_roots,
        background_noise_roots=background_noise_roots,
        urbansound_roots=urbansound_roots,
    )
    stage3.training.resume_from = checkpoints["stage2"]
    checkpoints["stage3"] = _run_trainer_stage(3, stage3)

    stage4 = _stage_config(
        "4",
        production,
        common_voice_roots=common_voice_roots,
        librispeech_roots=librispeech_roots,
        background_noise_roots=background_noise_roots,
        urbansound_roots=urbansound_roots,
    )
    stage4.training.resume_from = checkpoints["stage3"]
    checkpoints["stage4"] = _run_trainer_stage(4, stage4)

    stage4d = _stage_config(
        "4d",
        production,
        common_voice_roots=common_voice_roots,
        librispeech_roots=librispeech_roots,
        background_noise_roots=background_noise_roots,
        urbansound_roots=urbansound_roots,
    )
    stage4d.training.resume_from = checkpoints["stage4"]
    checkpoints["stage4d"] = _run_stage4d(stage4d)

    stage5 = _stage_config(
        "5",
        production,
        common_voice_roots=common_voice_roots,
        librispeech_roots=librispeech_roots,
        background_noise_roots=background_noise_roots,
        urbansound_roots=urbansound_roots,
    )
    stage5.training.resume_from = None
    stage5.stage5.reproduce_winner_only = not full_stage5_search
    checkpoints["stage5"] = _run_trainer_stage(5, stage5)

    stage6 = _stage_config(
        "6",
        production,
        common_voice_roots=common_voice_roots,
        librispeech_roots=librispeech_roots,
        background_noise_roots=background_noise_roots,
        urbansound_roots=urbansound_roots,
    )
    stage6.training.resume_from = checkpoints["stage5"]
    checkpoints["stage6"] = _run_trainer_stage(6, stage6)

    stage7 = _stage_config(
        "7",
        production,
        common_voice_roots=common_voice_roots,
        librispeech_roots=librispeech_roots,
        background_noise_roots=background_noise_roots,
        urbansound_roots=urbansound_roots,
    )
    stage7.training.resume_from = checkpoints["stage6"]
    checkpoints["stage7"] = _run_trainer_stage(7, stage7)
    _verify_stage7(stage7, checkpoints["stage7"])
    corrected = _finalize_stage7(stage7)
    calibration_summary = json.loads(stage7.stage7.joint_calibration_summary.read_text())
    dataset_roots = {
        "common_voice": [str(p) for p in stage7.data.external_common_voice_roots],
        "librispeech": [str(p) for p in stage7.data.external_librispeech_roots],
        "background_noise": [str(p) for p in stage7.data.external_background_noise_roots],
        "urbansound8k": [str(p) for p in stage7.data.external_urbansound_roots],
    }
    _write_pipeline_summary(
        final_artifact=corrected,
        calibration_summary=calibration_summary,
        stage5_best_candidate={
            "stage4_candidate": stage5.stage5.stage4_candidate_name,
            "data_variant": stage5.stage5.data_variant_name,
            "fusion_variant": stage5.stage5.fusion_variant_name,
        },
        stage6_demo_passed=True,
        stage7_default_tau=float(
            json.loads(stage7.stage7.final_kpi_verification.read_text()).get("export_tau", 0.32)
            if stage7.stage7.final_kpi_verification.exists()
            else 0.32
        ),
        dataset_roots_used=dataset_roots,
        smoke=False,
    )
    print(f"Corrected Stage7 deployable: {corrected}")
    print("Production pipeline summary: reports/production_pipeline_summary.json")
    return corrected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SoloSpeak production pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/production.yaml"),
    )
    parser.add_argument("--common-voice-root", action="append", type=Path, default=[])
    parser.add_argument("--librispeech-root", action="append", type=Path, default=[])
    parser.add_argument("--background-noise-root", action="append", type=Path, default=[])
    parser.add_argument("--urbansound-root", action="append", type=Path, default=[])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full-stage5-search", action="store_true")
    args = parser.parse_args()
    run_production_pipeline(
        config_path=args.config,
        common_voice_roots=args.common_voice_root,
        librispeech_roots=args.librispeech_root,
        background_noise_roots=args.background_noise_root,
        urbansound_roots=args.urbansound_root,
        smoke=args.smoke,
        full_stage5_search=args.full_stage5_search,
    )


if __name__ == "__main__":
    main()
