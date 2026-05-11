"""Stage-7 corrected deployable export.

This is the source-code version of the final ``ta-test.ipynb`` corrected export cell.
It applies the joint internal+external threshold calibration summary to an existing
Stage-7 deployable artifact and writes both a full checkpoint and lightweight deployable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from solospeak.training.stages.common import save_checkpoint
from solospeak.utils.config import SoloSpeakConfig


def _load_summary(summary: Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(summary, Path):
        return cast(dict[str, Any], json.loads(summary.read_text()))
    return dict(summary)


def _deployable_config(config: SoloSpeakConfig, tau: float) -> dict[str, Any]:
    return {
        "audio": {
            "sample_rate": config.audio.sample_rate,
            "window_samples": config.audio.window_samples,
            "n_mels": config.audio.n_mels,
            "n_fft": config.audio.n_fft,
            "hop_length": config.audio.hop_length,
            "win_length": config.audio.win_length,
        },
        "fusion": {
            "tau_on": float(tau),
            "tau_off": float(config.fusion.tau_off),
        },
        "heads": {
            "content_dim": config.heads.content_dim,
            "speaker_dim": config.heads.speaker_dim,
            "hidden_dim": config.heads.hidden_dim,
        },
        "backbone": {
            "variant": config.backbone.variant,
        },
    }


def export_stage7_deployable(
    *,
    stage7_checkpoint: Path = Path("checkpoints/stage7_fusion.pt"),
    final_checkpoint: Path = Path("checkpoints/stage7_final.pt"),
    deployable: Path = Path("exports/solospeak_stage7_deployable.pt"),
    config_path: Path = Path("configs/training/stage7_external_fa.yaml"),
    calibration_summary: Path | dict[str, Any] | None = None,
    external_fa_summary: Path | dict[str, Any] | None = None,
) -> Path:
    """Export the notebook-style Stage-7 lightweight deployable artifact."""

    import torch

    if not stage7_checkpoint.exists():
        raise FileNotFoundError(f"Missing Stage7 checkpoint: {stage7_checkpoint}")
    stage7_obj = torch.load(stage7_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(stage7_obj, dict) or "model_state" not in stage7_obj:
        raise ValueError(f"{stage7_checkpoint} must contain a dict with 'model_state'.")
    if stage7_obj.get("stage_origin") != 7:
        raise ValueError(f"Expected stage_origin=7, got {stage7_obj.get('stage_origin')!r}.")

    metrics = dict(stage7_obj.get("metrics", {}) or {})
    extra = dict(stage7_obj.get("extra", {}) or {})
    config = SoloSpeakConfig.from_yaml(config_path)
    tau = float(extra.get("stage7_tau", metrics.get("dev/tau_stage7", config.fusion.tau_on)))
    export_mode = "stage7_default_tau"

    calibration_payload: dict[str, Any] | None = None
    if calibration_summary is not None:
        calibration_payload = _load_summary(calibration_summary)
        if calibration_payload.get("export_tau") is not None:
            tau = float(calibration_payload["export_tau"])
            export_mode = str(calibration_payload.get("export_mode", "joint_calibrated"))
        elif calibration_payload.get("best_production_candidate") is not None:
            tau = float(calibration_payload["best_production_candidate"]["tau"])
            export_mode = "production_candidate"
        elif calibration_payload.get("best_demo_candidate") is not None:
            tau = float(calibration_payload["best_demo_candidate"]["tau"])
            export_mode = "demo_candidate"
        elif calibration_payload.get("default_tau_candidate") is not None:
            tau = float(calibration_payload["default_tau_candidate"]["tau"])
            export_mode = "stage7_default_tau"

    external_payload: dict[str, Any] = {}
    if external_fa_summary is not None:
        external_payload = _load_summary(external_fa_summary)

    config.fusion.tau_on = tau
    final_metrics = {
        **metrics,
        "stage7/export_tau": float(tau),
        "stage7/export_mode_production": float(export_mode == "production_candidate"),
        "stage7/export_mode_demo": float(export_mode == "demo_candidate"),
    }

    save_checkpoint(
        final_checkpoint,
        stage_origin=7,
        config=config,
        step=int(stage7_obj.get("step") or metrics.get("dev/best_epoch", 0)),
        metrics=final_metrics,
        model_state=stage7_obj["model_state"],
        optimizer_state={},
        extra={
            "stage7_fusion_checkpoint": str(stage7_checkpoint),
            "export_tau": float(tau),
            "export_mode": export_mode,
            "wake_word": extra.get("wake_word", "zero"),
            "n_enroll": int(extra.get("n_enroll", 3)),
            "profiles": extra.get("profiles"),
            "joint_calibration_summary": calibration_payload,
            "external_fa_summary": external_payload,
        },
    )

    deployable.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": stage7_obj["model_state"],
            "tau_on": float(tau),
            "config": _deployable_config(config, tau),
            "stage7_metrics": final_metrics,
            "stage7_extra": {
                "stage7_fusion_checkpoint": str(stage7_checkpoint),
                "export_tau": float(tau),
                "export_mode": export_mode,
                "wake_word": extra.get("wake_word", "zero"),
                "n_enroll": int(extra.get("n_enroll", 3)),
                "profiles": extra.get("profiles"),
                "joint_calibration_summary": calibration_payload,
                "external_fa_summary": external_payload,
            },
        },
        deployable,
    )
    return deployable


def export_corrected_stage7(
    *,
    old_deployable: Path = Path("exports/solospeak_stage7_deployable.pt"),
    corrected_checkpoint: Path = Path("checkpoints/stage7_final_corrected.pt"),
    corrected_deployable: Path = Path("exports/solospeak_stage7_deployable_corrected.pt"),
    calibration_summary: Path | dict[str, Any] = Path(
        "reports/stage7_joint_threshold_calibration_summary.json"
    ),
    config_path: Path = Path("configs/training/stage7_external_fa.yaml"),
    final_summary: Path = Path("reports/stage7_final_summary.json"),
) -> Path:
    """Apply the joint-calibrated Stage-7 tau to a deployable ``.pt`` artifact."""

    import torch

    if not old_deployable.exists():
        raise FileNotFoundError(f"Missing Stage7 deployable: {old_deployable}")
    summary = _load_summary(calibration_summary)
    export_tau = float(summary["export_tau"])
    if abs(export_tau - 0.935) < 1e-6:
        raise ValueError("Refusing to export the known bad external-only tau=0.935.")
    export_mode = str(summary.get("export_mode", "unknown"))

    deploy_obj = torch.load(old_deployable, map_location="cpu", weights_only=False)
    if not isinstance(deploy_obj, dict) or "model_state" not in deploy_obj:
        raise ValueError(f"{old_deployable} must contain a dict with 'model_state'.")
    old_extra = deploy_obj.get("stage7_extra", {})
    if not isinstance(old_extra, dict):
        old_extra = {}

    config = SoloSpeakConfig.from_yaml(config_path)
    config.fusion.tau_on = export_tau
    best = dict(summary.get("best", {}))
    final_metrics = {
        **dict(deploy_obj.get("stage7_metrics", {}) or {}),
        "stage7/export_tau": export_tau,
        "stage7/export_mode_production": float(export_mode == "production_candidate"),
        "stage7/export_mode_demo": float(export_mode == "demo_candidate"),
        "stage7/ta_clean": float(best.get("ta_clean", 0.0)),
        "stage7/q2_rejection": float(best.get("q2_rejection", 0.0)),
        "stage7/q3_rejection": float(best.get("q3_rejection", 0.0)),
        "stage7/q4_rejection": float(best.get("q4_rejection", 0.0)),
        "stage7/quadrant_min": float(best.get("quadrant_min", 0.0)),
        "stage7/overall_external_fa_rate": float(best.get("overall_fa_rate", 0.0)),
        "stage7/overall_external_false_accepts": float(best.get("overall_false_accepts", 0.0)),
    }
    stage7_extra = {
        "source_deployable": str(old_deployable),
        "export_tau": export_tau,
        "export_mode": export_mode,
        "joint_calibration_summary": summary,
        "external_fa_summary": old_extra.get("external_fa_summary", {}),
        "wake_word": old_extra.get("wake_word", "zero"),
        "n_enroll": int(old_extra.get("n_enroll", 3)),
    }

    save_checkpoint(
        corrected_checkpoint,
        stage_origin=7,
        config=config,
        step=0,
        metrics=final_metrics,
        model_state=deploy_obj["model_state"],
        optimizer_state={},
        extra=stage7_extra,
    )

    corrected = {
        "model_state": deploy_obj["model_state"],
        "tau_on": export_tau,
        "config": _deployable_config(config, export_tau),
        "stage7_metrics": final_metrics,
        "stage7_extra": stage7_extra,
    }

    corrected_deployable.parent.mkdir(parents=True, exist_ok=True)
    torch.save(corrected, corrected_deployable)
    final_summary.parent.mkdir(parents=True, exist_ok=True)
    final_summary.write_text(
        json.dumps(
            {
                "final_artifact": str(corrected_deployable),
                "final_checkpoint": str(corrected_checkpoint),
                "final_tau": export_tau,
                "export_mode": export_mode,
                "production_candidate": bool(export_mode == "production_candidate"),
                "internal_metrics": {
                    "ta_clean": float(best.get("ta_clean", 0.0)),
                    "q2_rejection": float(best.get("q2_rejection", 0.0)),
                    "q3_rejection": float(best.get("q3_rejection", 0.0)),
                    "q4_rejection": float(best.get("q4_rejection", 0.0)),
                    "quadrant_min": float(best.get("quadrant_min", 0.0)),
                },
                "external_fa_metrics": {
                    "overall_fa_rate": float(best.get("overall_fa_rate", 0.0)),
                    "overall_false_accepts": int(best.get("overall_false_accepts", 0)),
                },
                "joint_calibration_summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return corrected_deployable
