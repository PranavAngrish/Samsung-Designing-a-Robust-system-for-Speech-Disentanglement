"""Stage-6 final evaluation/export path from the production notebook."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from solospeak.data.datasets import QuadrantDataset, collate_quadrant
from solospeak.data.features import LogMelExtractor
from solospeak.models.solospeak import SoloSpeakModel
from solospeak.training.stages.base import TrainingStage
from solospeak.training.stages.common import (
    default_device,
    is_smoke,
    make_mel,
    save_checkpoint,
    to_device,
)
from solospeak.utils.config import SoloSpeakConfig
from solospeak.utils.types import LossDict, MetricsDict, StageBatch


def _tau_from_stage5(config: SoloSpeakConfig, stage5_obj: dict[str, Any]) -> float:
    metrics = stage5_obj.get("metrics", {}) or {}
    extra = stage5_obj.get("extra", {}) or {}
    if "dev/tau_stage5" in metrics:
        return float(metrics["dev/tau_stage5"])
    if "best_tau" in extra:
        return float(extra["best_tau"])
    if "dev/tau" in metrics:
        return float(metrics["dev/tau"])
    return float(config.fusion.tau_on)


def _stage5_gate(stage5_obj: dict[str, Any]) -> tuple[bool, bool]:
    metrics = stage5_obj.get("metrics", {}) or {}
    extra = stage5_obj.get("extra", {}) or {}
    gate = (
        bool(extra.get("gate_passed", False))
        or float(metrics.get("dev/gate_passed", 0.0)) >= 1.0
    )
    target = bool(extra.get("target_passed", False)) or float(
        metrics.get("dev/target_passed", 0.0)
    ) >= 1.0
    return gate, target


def _repair_profile_paths(manifest: Path, config: SoloSpeakConfig) -> Path:
    """Rewrite missing profile paths when Stage5 profiles were copied locally."""

    if not manifest.exists():
        return manifest
    with open(manifest, newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    if not rows or "profile_path" not in fieldnames:
        return manifest

    fallback_dirs = [
        config.data.root / "stage5" / "profiles" / "final_best",
        Path("data/stage5/profiles/final_best"),
        manifest.parent / "profiles",
    ]
    changed = False
    for row in rows:
        profile_raw = str(row.get("profile_path") or "")
        if not profile_raw:
            continue
        profile_path = Path(profile_raw)
        if profile_path.exists():
            continue
        for profile_dir in fallback_dirs:
            candidate = profile_dir / profile_path.name
            if candidate.exists():
                row["profile_path"] = str(candidate)
                changed = True
                break

    if not changed:
        return manifest
    out = config.data.root / "stage6" / f"repaired_{manifest.name}"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return out


@torch.no_grad()
def _score_quadrant_manifest(
    *,
    model: SoloSpeakModel,
    extractor: LogMelExtractor,
    config: SoloSpeakConfig,
    manifest: Path,
    tau: float,
    output_csv: Path,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    dataset = QuadrantDataset(manifest, config.audio)
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_quadrant,
    )
    rows: list[dict[str, Any]] = []
    probs_all: list[np.ndarray] = []
    sc_all: list[np.ndarray] = []
    ss_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    quadrants_all: list[str] = []

    model.eval()
    extractor.eval()
    for batch in loader:
        batch = to_device(batch, device)
        content_template = F.normalize(batch["content_template"], p=2, dim=-1)
        speaker_template = F.normalize(batch["speaker_template"], p=2, dim=-1)
        labels = batch["quadrant_label"].float()
        z_c, z_s = model(make_mel(batch, extractor))
        z_c = F.normalize(z_c.float(), p=2, dim=-1)
        z_s = F.normalize(z_s.float(), p=2, dim=-1)
        s_c = (z_c * content_template).sum(dim=-1)
        s_s = (z_s * speaker_template).sum(dim=-1)
        probs = model.forward_fusion(s_c, s_s).view(-1)

        probs_all.append(probs.detach().cpu().numpy())
        sc_all.append(s_c.detach().cpu().numpy())
        ss_all.append(s_s.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())
        quadrants_all.extend([str(q) for q in batch["quadrant_class"]])

    probs_np = np.concatenate(probs_all) if probs_all else np.array([], dtype=np.float32)
    sc_np = np.concatenate(sc_all) if sc_all else np.array([], dtype=np.float32)
    ss_np = np.concatenate(ss_all) if ss_all else np.array([], dtype=np.float32)
    labels_np = np.concatenate(labels_all) if labels_all else np.array([], dtype=np.float32)
    accept = probs_np >= tau
    quadrants = np.array(quadrants_all, dtype=object)

    def frac(mask: np.ndarray, values: np.ndarray) -> float:
        denom = int(mask.sum())
        if denom <= 0:
            return 0.0
        return float(values[mask].mean())

    q1 = quadrants == "Q1_accept"
    q2 = quadrants == "Q2_imposter"
    q3 = quadrants == "Q3_wrong_word"
    q4 = quadrants == "Q4_background"
    metrics = {
        "tau": float(tau),
        "rows": int(len(probs_np)),
        "ta_clean": frac(q1, accept.astype(np.float32)),
        "q2_rejection": frac(q2, (~accept).astype(np.float32)),
        "q3_rejection": frac(q3, (~accept).astype(np.float32)),
        "q4_rejection": frac(q4, (~accept).astype(np.float32)),
        "accepted_rate": float(accept.mean()) if len(accept) else 0.0,
        "mean_prob": float(probs_np.mean()) if len(probs_np) else 0.0,
        "mean_s_c": float(sc_np.mean()) if len(sc_np) else 0.0,
        "mean_s_s": float(ss_np.mean()) if len(ss_np) else 0.0,
    }
    vals = [
        metrics["ta_clean"],
        metrics["q2_rejection"],
        metrics["q3_rejection"],
        metrics["q4_rejection"],
    ]
    metrics["quadrant_accuracy_min"] = float(min(vals))
    metrics["quadrant_accuracy_mean"] = float(sum(vals) / len(vals))
    metrics["q1q2_min"] = float(min(metrics["ta_clean"], metrics["q2_rejection"]))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        fields = ["quadrant", "label", "prob", "s_c", "s_s", "accepted", "tau"]
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for quadrant, label, prob, sc, ss, accepted in zip(
            quadrants, labels_np, probs_np, sc_np, ss_np, accept, strict=False
        ):
            writer.writerow(
                {
                    "quadrant": quadrant,
                    "label": float(label),
                    "prob": float(prob),
                    "s_c": float(sc),
                    "s_s": float(ss),
                    "accepted": bool(accepted),
                    "tau": float(tau),
                }
            )
    metrics["score_file"] = str(output_csv)
    return metrics


@torch.no_grad()
def _score_noisy_q1_manifest(
    *,
    model: SoloSpeakModel,
    extractor: LogMelExtractor,
    config: SoloSpeakConfig,
    manifest: Path,
    tau: float,
    snrs: list[int],
) -> dict[str, float]:
    device = next(model.parameters()).device
    dataset = QuadrantDataset(manifest, config.audio)
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_quadrant,
    )
    model.eval()
    extractor.eval()

    metrics: dict[str, float] = {}
    snr_values: list[float] = []
    for snr in snrs:
        torch.manual_seed(6000 + int(snr))
        accepted = 0
        total = 0
        for batch in loader:
            quadrants = [str(q) for q in batch["quadrant_class"]]
            idxs = [idx for idx, q in enumerate(quadrants) if q == "Q1_accept"]
            if not idxs:
                continue
            idx = torch.tensor(idxs, dtype=torch.long)
            wav = batch["wav"][idx].to(device).float()
            content_template = F.normalize(
                batch["content_template"][idx].to(device).float(),
                p=2,
                dim=-1,
            )
            speaker_template = F.normalize(
                batch["speaker_template"][idx].to(device).float(),
                p=2,
                dim=-1,
            )

            signal_rms = torch.sqrt(torch.mean(wav**2, dim=1, keepdim=True)).clamp_min(1e-6)
            noise = torch.randn_like(wav)
            noise_rms = torch.sqrt(torch.mean(noise**2, dim=1, keepdim=True)).clamp_min(1e-6)
            noise_scale = signal_rms / (10.0 ** (float(snr) / 20.0))
            noisy = torch.clamp(wav + noise / noise_rms * noise_scale, -1.0, 1.0)

            z_c, z_s = model(make_mel({"wav": noisy}, extractor))
            z_c = F.normalize(z_c.float(), p=2, dim=-1)
            z_s = F.normalize(z_s.float(), p=2, dim=-1)
            s_c = (z_c * content_template).sum(dim=-1)
            s_s = (z_s * speaker_template).sum(dim=-1)
            probs = model.forward_fusion(s_c, s_s).view(-1)
            accepted += int((probs >= tau).sum().item())
            total += int(probs.numel())
        value = float(accepted / total) if total else 0.0
        metrics[f"test/ta_noisy_snr_{snr}"] = value
        snr_values.append(value)
    metrics["test/ta_noisy_macro"] = float(np.mean(snr_values)) if snr_values else 0.0
    return metrics


def _deployable_payload(
    model: SoloSpeakModel,
    config: SoloSpeakConfig,
    tau: float,
    metrics: MetricsDict,
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "tau_on": float(tau),
        "config": {
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
        },
        "stage6_metrics": metrics,
        "stage6_extra": extra,
    }


def _read_score_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_threshold_diagnostic(
    *,
    test_scores: Path,
    fa_scores: Path,
    output_csv: Path,
) -> None:
    test_rows = _read_score_csv(test_scores)
    fa_rows = _read_score_csv(fa_scores)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "tau",
        "ta_clean",
        "q2_rejection",
        "q3_rejection",
        "q4_rejection",
        "quadrant_min",
        "quadrant_mean",
        "internal_accepted_rate",
        "fa_false_accepts",
        "fa_trials",
        "fa_rate",
    ]

    def frac(mask: np.ndarray, values: np.ndarray) -> float:
        denom = int(mask.sum())
        if denom <= 0:
            return 0.0
        return float(values[mask].mean())

    quadrants = np.array([row.get("quadrant", "") for row in test_rows], dtype=object)
    probs = np.array([float(row.get("prob", 0.0)) for row in test_rows], dtype=np.float32)
    fa_probs = np.array([float(row.get("prob", 0.0)) for row in fa_rows], dtype=np.float32)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for tau in np.linspace(0.05, 0.95, 181):
            accept = probs >= tau
            q1 = quadrants == "Q1_accept"
            q2 = quadrants == "Q2_imposter"
            q3 = quadrants == "Q3_wrong_word"
            q4 = quadrants == "Q4_background"
            ta = frac(q1, accept.astype(np.float32))
            q2r = frac(q2, (~accept).astype(np.float32))
            q3r = frac(q3, (~accept).astype(np.float32))
            q4r = frac(q4, (~accept).astype(np.float32))
            vals = [ta, q2r, q3r, q4r]
            fa_accept = fa_probs >= tau
            writer.writerow(
                {
                    "tau": float(tau),
                    "ta_clean": ta,
                    "q2_rejection": q2r,
                    "q3_rejection": q3r,
                    "q4_rejection": q4r,
                    "quadrant_min": min(vals),
                    "quadrant_mean": sum(vals) / len(vals),
                    "internal_accepted_rate": float(accept.mean()) if len(accept) else 0.0,
                    "fa_false_accepts": int(fa_accept.sum()) if len(fa_accept) else 0,
                    "fa_trials": int(len(fa_accept)),
                    "fa_rate": float(fa_accept.mean()) if len(fa_accept) else 0.0,
                }
            )


def run_stage6_final_eval(config: SoloSpeakConfig) -> Path:
    checkpoint_dir = config.training.checkpoint_dir
    reports_dir = Path("reports")
    exports_dir = Path("exports")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)

    stage5_path = config.training.resume_from or checkpoint_dir / "stage5_fusion.pt"
    recovery_used = False
    if not stage5_path.exists() and config.stage6.recovery_fallback_to_last:
        fallback = checkpoint_dir / "stage5_last.pt"
        if fallback.exists():
            stage5_path = fallback
            recovery_used = True
    if not stage5_path.exists():
        raise FileNotFoundError(f"Missing Stage5 checkpoint for Stage6: {stage5_path}")
    stage5_obj = torch.load(stage5_path, map_location="cpu", weights_only=False)
    if "model_state" not in stage5_obj:
        raise ValueError(f"Stage5 checkpoint missing model_state: {stage5_path}")
    stage5_gate_passed, stage5_target_passed = _stage5_gate(stage5_obj)
    if not stage5_gate_passed:
        raise RuntimeError("Stage5 gate did not pass; refusing Stage6 final export.")

    tau = _tau_from_stage5(config, stage5_obj)
    config.fusion.tau_on = tau
    device = default_device()
    model = SoloSpeakModel(config).to(device).eval()
    missing, unexpected = model.load_state_dict(stage5_obj["model_state"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Bad Stage5 state: missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    extractor = LogMelExtractor(config.audio).to(device).eval()
    param_count = sum(p.numel() for p in model.parameters())

    test_manifest = config.stage6.test_kpi_manifest
    if not test_manifest.exists():
        test_manifest = config.data.root / "stage5" / "stage5_final_test_quadrants.csv"
    if not test_manifest.exists():
        raise FileNotFoundError(f"Missing Stage6 test quadrant manifest: {test_manifest}")
    test_manifest = _repair_profile_paths(test_manifest, config)

    test_metrics = _score_quadrant_manifest(
        model=model,
        extractor=extractor,
        config=config,
        manifest=test_manifest,
        tau=tau,
        output_csv=config.stage6.test_scores_csv,
    )
    noisy_metrics = (
        _score_noisy_q1_manifest(
            model=model,
            extractor=extractor,
            config=config,
            manifest=test_manifest,
            tau=tau,
            snrs=config.stage6.noisy_snrs,
        )
        if config.stage6.evaluate_noisy_q1
        else {"test/ta_noisy_macro": 0.0}
    )
    fa_manifest = config.stage6.test_fa_manifest
    if fa_manifest.exists():
        fa_manifest = _repair_profile_paths(fa_manifest, config)
        fa_metrics = _score_quadrant_manifest(
            model=model,
            extractor=extractor,
            config=config,
            manifest=fa_manifest,
            tau=tau,
            output_csv=config.stage6.fa_scores_csv,
        )
        fa_rows = float(fa_metrics["rows"])
        fa_false_accepts = float(fa_metrics["accepted_rate"]) * fa_rows
        duration_s = float(config.audio.window_samples) / float(config.audio.sample_rate)
        fa_hours = fa_rows * duration_s / 3600.0
        fa_per_hour = fa_false_accepts / max(fa_hours, 1e-9)
    else:
        fa_metrics = {"rows": 0, "accepted_rate": 0.0}
        fa_per_hour = 0.0
    _write_threshold_diagnostic(
        test_scores=config.stage6.test_scores_csv,
        fa_scores=config.stage6.fa_scores_csv,
        output_csv=config.stage6.threshold_diagnostic_csv,
    )

    targets = {
        "ta_clean": 0.92,
        "ta_noisy_macro": 0.80,
        "fa_per_hour": 2.0,
        "q2_rejection": 0.95,
        "q3_rejection": 0.90,
        "q4_rejection": 0.99,
        "quadrant_min": 0.90,
        "param_count": 3_000_000,
        "demo_ta_clean": 0.90,
        "demo_q2_rejection": 0.90,
        "demo_q3_rejection": 0.90,
        "demo_q4_rejection": 0.95,
        "demo_quadrant_min": 0.90,
    }
    strict_stage6_passed = (
        test_metrics["ta_clean"] >= targets["ta_clean"]
        and noisy_metrics["test/ta_noisy_macro"] >= targets["ta_noisy_macro"]
        and fa_per_hour <= targets["fa_per_hour"]
        and test_metrics["q2_rejection"] >= targets["q2_rejection"]
        and test_metrics["q3_rejection"] >= targets["q3_rejection"]
        and test_metrics["q4_rejection"] >= targets["q4_rejection"]
        and test_metrics["quadrant_accuracy_min"] >= targets["quadrant_min"]
        and param_count <= targets["param_count"]
    )
    demo_stage6_passed = (
        test_metrics["ta_clean"] >= targets["demo_ta_clean"]
        and test_metrics["q2_rejection"] >= targets["demo_q2_rejection"]
        and test_metrics["q3_rejection"] >= targets["demo_q3_rejection"]
        and test_metrics["q4_rejection"] >= targets["demo_q4_rejection"]
        and test_metrics["quadrant_accuracy_min"] >= targets["demo_quadrant_min"]
        and param_count <= targets["param_count"]
    )
    metrics: MetricsDict = {
        "stage6/stage5_gate_passed": float(stage5_gate_passed),
        "stage6/stage5_target_passed": float(stage5_target_passed),
        "stage6/strict_stage6_passed": float(strict_stage6_passed),
        "stage6/demo_stage6_passed": float(demo_stage6_passed),
        "stage6/tau_on": float(tau),
        "stage6/param_count": float(param_count),
        "test/ta_clean": float(test_metrics["ta_clean"]),
        **{key: float(value) for key, value in noisy_metrics.items()},
        "test/fa_per_hour": float(fa_per_hour),
        "test/q2_rejection": float(test_metrics["q2_rejection"]),
        "test/q3_rejection": float(test_metrics["q3_rejection"]),
        "test/q4_rejection": float(test_metrics["q4_rejection"]),
        "test/quadrant_accuracy_min": float(test_metrics["quadrant_accuracy_min"]),
        "test/quadrant_accuracy_mean": float(test_metrics["quadrant_accuracy_mean"]),
        "test/accepted_rate": float(test_metrics["accepted_rate"]),
    }
    extra = {
        "source_stage5_checkpoint": str(stage5_path),
        "recovery_fallback_to_last_used": bool(recovery_used),
        "tau_on": float(tau),
        "stage5_gate_passed": bool(stage5_gate_passed),
        "stage5_target_passed": bool(stage5_target_passed),
        "strict_stage6_passed": bool(strict_stage6_passed),
        "demo_stage6_passed": bool(demo_stage6_passed),
        "targets": targets,
        "test_metrics": test_metrics,
        "noisy_metrics": noisy_metrics,
        "fa_metrics": fa_metrics,
    }
    stage6_final = save_checkpoint(
        config.stage6.output_checkpoint,
        stage_origin=6,
        config=config,
        step=int(stage5_obj.get("step", 0)),
        metrics=metrics,
        model_state=model.state_dict(),
        optimizer_state={},
        extra=extra,
    )
    if config.stage6.compatibility_checkpoint != config.stage6.output_checkpoint:
        save_checkpoint(
            config.stage6.compatibility_checkpoint,
            stage_origin=6,
            config=config,
            step=int(stage5_obj.get("step", 0)),
            metrics=metrics,
            model_state=model.state_dict(),
            optimizer_state={},
            extra=extra,
        )
    config.stage6.deployable_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _deployable_payload(model, config, tau, metrics, extra),
        config.stage6.deployable_path,
    )
    config.stage6.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    config.stage6.metrics_json.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    config.stage6.summary_json.parent.mkdir(parents=True, exist_ok=True)
    config.stage6.summary_json.write_text(
        json.dumps(
            {
                "stage6_checkpoint": str(stage6_final),
                "deployable_export": str(config.stage6.deployable_path),
                "strict_stage6_passed": bool(strict_stage6_passed),
                "demo_stage6_passed": bool(demo_stage6_passed),
                "test_metrics": test_metrics,
                "noisy_metrics": noisy_metrics,
                "fa_metrics": fa_metrics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return stage6_final


class Stage6FinalEval(TrainingStage):
    """Stage-6 production path. Non-smoke runs final eval/export, not QAT."""

    stage_id = 6
    stage_name = "stage6_final_eval"
    min_gate_metric = "stage6/demo_stage6_passed"
    min_gate_threshold = 1.0
    target_gate_threshold = 1.0

    def prepare_data(self) -> tuple[DataLoader[Any], ...]:
        return ()

    def build_model(self) -> nn.Module:
        raise NotImplementedError("Stage6FinalEval uses run_stage6_final_eval().")

    def compute_loss(self, batches: StageBatch, step: int) -> LossDict:
        raise NotImplementedError("Stage6FinalEval does not train.")

    def on_epoch_end(self, epoch: int) -> MetricsDict:
        return {}

    def go_no_go_check(self, metrics: MetricsDict) -> tuple[bool, bool]:
        demo = bool(metrics.get("stage6/demo_stage6_passed", 0.0) >= 1.0)
        strict = bool(metrics.get("stage6/strict_stage6_passed", 0.0) >= 1.0)
        if not demo:
            raise RuntimeError("Stage6 final evaluation did not pass the demo gate.")
        return demo, strict

    def run(self) -> Path:
        if is_smoke(self.config):
            from solospeak.training.stages.stage6_qat import Stage6

            return Stage6(self.config).run()
        return run_stage6_final_eval(self.config)
