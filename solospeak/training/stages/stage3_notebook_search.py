"""Notebook-authoritative Stage-3 disentanglement search."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from solospeak.data.datasets import DualHeadDataset, collate_dual_head
from solospeak.data.features import LogMelExtractor
from solospeak.data.samplers import ClassAwareBatchSampler
from solospeak.eval.probes import run_probe_eval
from solospeak.losses.combined import CombinedLoss
from solospeak.training.stages.base import TrainingStage
from solospeak.training.stages.common import (
    checkpoint_path,
    default_device,
    inject_class_counts,
    make_mel,
    save_checkpoint,
    smoke_batch_size,
    to_device,
)
from solospeak.training.training_wrapper import TrainingWrapper
from solospeak.utils.config import SoloSpeakConfig
from solospeak.utils.seeding import seed_everything
from solospeak.utils.types import MetricsDict


@dataclass(frozen=True)
class Stage3Variant:
    name: str
    lr: float
    epochs: int
    adversarial: float
    orthogonality: float
    ramp_steps: int
    extra_speaker_ce: float = 0.0


STAGE3_VARIANTS = [
    Stage3Variant("clean_default_reset_step", 5e-4, 10, 0.10, 0.10, 5000, 0.0),
    Stage3Variant("gentle_adv005_ramp8000", 2e-4, 10, 0.05, 0.10, 8000, 0.0),
    Stage3Variant("gentle_adv010_ortho015_extra025", 2e-4, 10, 0.10, 0.15, 8000, 0.25),
]


def _make_scaler(enabled: bool) -> Any:
    try:
        from torch.amp import GradScaler

        return GradScaler("cuda", enabled=enabled)
    except Exception:
        from torch.cuda.amp import GradScaler

        return GradScaler(enabled=enabled)


def _amp(enabled: bool) -> Any:
    try:
        from torch.amp import autocast

        return autocast("cuda", enabled=enabled)
    except Exception:
        from torch.cuda.amp import autocast

        return autocast(enabled=enabled)


class NotebookStage3(TrainingStage):
    stage_id = 3
    stage_name = "stage3_disentangle_search"

    def __init__(self, config: SoloSpeakConfig, variant: Stage3Variant) -> None:
        super().__init__(config)
        inject_class_counts(self.config)
        self.variant = variant
        self.device = default_device()
        self.extractor = LogMelExtractor(config.audio).to(self.device)
        self.loss_fn = CombinedLoss(stage=3, weights=config.losses)
        self.model: TrainingWrapper | None = None

    def prepare_data(self) -> tuple[DataLoader[Any], DataLoader[Any]]:
        batch_size = smoke_batch_size(self.config)
        content = DualHeadDataset(
            self.config.data.manifests_dir / "train_content.csv",
            self.config.audio,
            self.config.data,
            wav_augmenters=[],
        )
        speaker = DualHeadDataset(
            self.config.data.manifests_dir / "train_speaker.csv",
            self.config.audio,
            self.config.data,
            wav_augmenters=[],
        )
        content_sampler = ClassAwareBatchSampler(
            content.keyword_labels,
            batch_size=batch_size,
            samples_per_class=2,
            seed=self.config.training.seed,
        )
        speaker_sampler = ClassAwareBatchSampler(
            speaker.speaker_labels,
            batch_size=batch_size,
            samples_per_class=2,
            seed=self.config.training.seed + 17,
        )
        kwargs = {
            "num_workers": self.config.training.num_workers,
            "pin_memory": True,
            "persistent_workers": self.config.training.num_workers > 0,
            "collate_fn": collate_dual_head,
        }
        return (
            DataLoader(content, batch_sampler=content_sampler, **kwargs),
            DataLoader(speaker, batch_sampler=speaker_sampler, **kwargs),
        )

    def build_model(self) -> nn.Module:
        self.model = TrainingWrapper(self.config).to(self.device)
        ckpt = torch.load(self.config.training.resume_from, map_location="cpu", weights_only=False)
        if "wrapper_state" in ckpt:
            self.model.load_state_dict(ckpt["wrapper_state"], strict=False)
        elif "model_state" in ckpt:
            self.model.model.load_state_dict(ckpt["model_state"], strict=False)
        else:
            raise ValueError(
                f"Stage2 checkpoint has no usable state: {self.config.training.resume_from}"
            )
        return self.model

    def _train(self) -> tuple[Path, MetricsDict, dict[str, Any]]:
        content_loader, speaker_loader = self.prepare_data()
        model = self.build_model()
        self.step = 0
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )
        use_amp = torch.cuda.is_available()
        scaler = _make_scaler(use_amp)
        for _epoch in range(self.variant.epochs):
            model.train()
            speaker_iter: Iterator[Any] = cycle(speaker_loader)
            for content_batch in content_loader:
                speaker_batch = next(speaker_iter)
                optimizer.zero_grad(set_to_none=True)
                content_batch = to_device(content_batch, self.device)
                speaker_batch = to_device(speaker_batch, self.device)
                with _amp(False):
                    content_mel = make_mel(content_batch, self.extractor).float()
                    speaker_mel = make_mel(speaker_batch, self.extractor).float()
                with _amp(use_amp):
                    content_out = model(content_mel, adv_lambda=1.0)
                    speaker_out = model(speaker_mel, adv_lambda=1.0)
                content_out = {
                    key: value.float()
                    if torch.is_tensor(value) and torch.is_floating_point(value)
                    else value
                    for key, value in content_out.items()
                }
                speaker_out = {
                    key: value.float()
                    if torch.is_tensor(value) and torch.is_floating_point(value)
                    else value
                    for key, value in speaker_out.items()
                }
                components = self.loss_fn(
                    outputs={"content": content_out, "speaker": speaker_out},
                    batches={"content": content_batch, "speaker": speaker_batch},
                    step=self.step,
                )
                loss = components.total
                if self.variant.extra_speaker_ce > 0.0 and "speaker_logits" in content_out:
                    labels = content_batch["speaker_label"].long()
                    extra_ce = F.cross_entropy(content_out["speaker_logits"], labels)
                    loss = loss + self.variant.extra_speaker_ce * extra_ce
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config.training.max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                self.step += 1

        if self.model is None:
            raise RuntimeError("Stage3 model was not built.")
        variant_ckpt = checkpoint_path(self.config, f"stage3_{self.variant.name}.pt")
        save_checkpoint(
            variant_ckpt,
            stage_origin=3,
            config=self.config,
            step=self.step,
            metrics={},
            model_state=self.model.deployable_state_dict(),
            wrapper_state=self.model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            extra={
                "gate_checked": False,
                "variant": asdict(self.variant),
                "pre_probe": True,
            },
        )
        probes = run_probe_eval(
            checkpoint_path=variant_ckpt,
            dev_manifest=self.config.data.manifests_dir / "dev_content.csv",
            probe_epochs=15,
            config_path=Path("configs/defaults.yaml"),
            baseline_checkpoint=self.config.training.resume_from,
        )
        reduction_c = probes.get("reduction_c")
        reduction_s = probes.get("reduction_s")
        rc = 0.0 if reduction_c is None else float(reduction_c)
        rs = 0.0 if reduction_s is None else float(reduction_s)
        metrics: MetricsDict = {
            "dev/probe_reduction_c": rc,
            "dev/probe_reduction_s": rs,
            "dev/probe_reduction": min(rc, rs),
            "dev/acc_c_post": float(probes["acc_c_post"]),
            "dev/acc_s_post": float(probes["acc_s_post"]),
            "dev/stage3_score": min(rc, rs),
        }
        Path("reports").mkdir(parents=True, exist_ok=True)
        (Path("reports") / f"probes_stage3_{self.variant.name}.json").write_text(
            json.dumps(
                {"variant": asdict(self.variant), "metrics": metrics, "probes": probes},
                indent=2,
            )
            + "\n"
        )
        save_checkpoint(
            variant_ckpt,
            stage_origin=3,
            config=self.config,
            step=self.step,
            metrics=metrics,
            model_state=self.model.deployable_state_dict(),
            wrapper_state=self.model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            extra={
                "gate_checked": False,
                "variant": asdict(self.variant),
                "probes": probes,
            },
        )
        return variant_ckpt, metrics, probes


def _resolve_stage2_checkpoint(config: SoloSpeakConfig) -> Path:
    if config.training.resume_from is not None and config.training.resume_from.exists():
        return config.training.resume_from
    for candidate in [
        config.training.checkpoint_dir / "stage2_dualhead.pt",
        config.training.checkpoint_dir / "stage2_last.pt",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No Stage 2 checkpoint found. Expected checkpoints/stage2_dualhead.pt "
        "or checkpoints/stage2_last.pt."
    )


def run_stage3_notebook_search(config: SoloSpeakConfig) -> Path:
    stage2_ckpt = _resolve_stage2_checkpoint(config)
    results: list[dict[str, Any]] = []
    for variant in STAGE3_VARIANTS:
        try:
            variant_config = config.model_copy(deep=True)
            variant_config.training.resume_from = stage2_ckpt
            variant_config.training.lr = variant.lr
            variant_config.training.num_epochs = variant.epochs
            variant_config.losses.adversarial = variant.adversarial
            variant_config.losses.orthogonality = variant.orthogonality
            variant_config.losses.adversarial_ramp_steps = variant.ramp_steps
            seed_everything(variant_config.training.seed)
            ckpt, metrics, _probes = NotebookStage3(variant_config, variant)._train()
            passed = (
                metrics["dev/probe_reduction_c"] >= 0.30
                and metrics["dev/probe_reduction_s"] >= 0.30
            )
            results.append(
                {
                    "name": variant.name,
                    "ckpt": str(ckpt),
                    "score": float(metrics["dev/stage3_score"]),
                    "passed": bool(passed),
                    "metrics": metrics,
                    "variant": asdict(variant),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "name": variant.name,
                    "ckpt": "",
                    "score": -1.0,
                    "passed": False,
                    "error": repr(exc),
                    "variant": asdict(variant),
                }
            )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/stage3_search_results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    valid = [r for r in results if r["ckpt"]]
    if not valid:
        raise RuntimeError("No Stage3 variants produced checkpoints.")
    best = max(valid, key=lambda r: r["score"])
    best_ckpt = Path(best["ckpt"])
    last_path = config.training.checkpoint_dir / "stage3_last.pt"
    final_path = config.training.checkpoint_dir / "stage3_disentangle.pt"
    shutil.copyfile(best_ckpt, last_path)
    if best["passed"]:
        shutil.copyfile(best_ckpt, final_path)
        return final_path
    return last_path
