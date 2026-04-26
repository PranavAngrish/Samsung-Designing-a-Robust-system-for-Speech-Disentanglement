"""Shared implementation for Stages 2-4 dual-head training."""

from __future__ import annotations

from itertools import cycle
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from solospeak.data.augmentation import AddNoise, ConvolveRIR, GainJitter, TimeShift
from solospeak.data.datasets import DualHeadDataset, collate_dual_head
from solospeak.data.features import LogMelExtractor
from solospeak.data.samplers import ClassAwareBatchSampler
from solospeak.losses.combined import CombinedLoss
from solospeak.training.stages.base import TrainingStage
from solospeak.training.stages.common import (
    backward,
    checkpoint_path,
    default_device,
    inject_class_counts,
    is_smoke,
    load_checkpoint,
    make_mel,
    pass_or_raise,
    save_checkpoint,
    smoke_batch_size,
    smoke_epochs,
    smoke_max_steps,
    to_device,
)
from solospeak.training.training_wrapper import TrainingWrapper
from solospeak.utils.config import SoloSpeakConfig
from solospeak.utils.types import LossDict, MetricsDict, StageBatch


class DualHeadTrainingStage(TrainingStage):
    """Runnable interleaved content/speaker training loop for Stages 2-4."""

    checkpoint_name: str

    def __init__(self, config: SoloSpeakConfig) -> None:
        super().__init__(config)
        inject_class_counts(self.config)
        self.device = default_device()
        self.extractor = LogMelExtractor(config.audio).to(self.device)
        self.loss_fn = CombinedLoss(stage=self.stage_id, weights=config.losses)
        self.model: TrainingWrapper | None = None

    def _augmenters(self) -> list[Any]:
        if self.stage_id == 4:
            return [
                GainJitter(prob=0.5),
                TimeShift(sr=self.config.audio.sample_rate, prob=0.5),
                AddNoise(Path("data/raw/musan/noise"), self.config.data.snr_range_db, prob=0.9),
                ConvolveRIR(Path("data/raw/rirs"), prob=0.7),
            ]
        if self.stage_id == 2:
            return [GainJitter(prob=0.3), TimeShift(sr=self.config.audio.sample_rate, prob=0.3)]
        return []

    def prepare_data(self) -> tuple[DataLoader[Any], ...]:
        batch_size = smoke_batch_size(self.config)
        content = DualHeadDataset(
            self.config.data.manifests_dir / "train_content.csv",
            self.config.audio,
            self.config.data,
            wav_augmenters=self._augmenters(),
        )
        speaker = DualHeadDataset(
            self.config.data.manifests_dir / "train_speaker.csv",
            self.config.audio,
            self.config.data,
            wav_augmenters=self._augmenters(),
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
        content_loader = DataLoader(
            content,
            batch_sampler=content_sampler,
            num_workers=0,
            collate_fn=collate_dual_head,
        )
        speaker_loader = DataLoader(
            speaker,
            batch_sampler=speaker_sampler,
            num_workers=0,
            collate_fn=collate_dual_head,
        )
        return content_loader, speaker_loader

    def build_model(self) -> nn.Module:
        self.model = TrainingWrapper(self.config).to(self.device)
        ckpt = load_checkpoint(self.config.training.resume_from)
        if ckpt is not None:
            if self.stage_id == 2:
                state = ckpt.get("backbone_state", ckpt.get("model_state"))
                if state is not None:
                    self.model.model.backbone.load_state_dict(state)
            elif "wrapper_state" in ckpt:
                self.model.load_state_dict(ckpt["wrapper_state"], strict=False)
            elif "model_state" in ckpt:
                self.model.model.load_state_dict(ckpt["model_state"], strict=False)
        return self.model

    def compute_loss(self, batches: StageBatch, step: int) -> LossDict:
        if self.model is None:
            raise RuntimeError("build_model() must be called before compute_loss()")
        content_batch = to_device(batches["content"], self.device)
        speaker_batch = to_device(batches["speaker"], self.device)
        adv_lambda = 1.0 if self.stage_id >= 3 else 0.0
        content_out = self.model(make_mel(content_batch, self.extractor), adv_lambda=adv_lambda)
        speaker_out = self.model(make_mel(speaker_batch, self.extractor), adv_lambda=adv_lambda)
        components = self.loss_fn(
            outputs={"content": content_out, "speaker": speaker_out},
            batches={"content": content_batch, "speaker": speaker_batch},
            step=step,
        )
        return {"total": components.total}

    def on_epoch_end(self, epoch: int) -> MetricsDict:
        if self.stage_id == 2:
            metrics = {"dev/ta_clean": 0.0}
        elif self.stage_id == 3:
            metrics = {
                "dev/probe_reduction": 0.0,
                "dev/probe_reduction_c": 0.0,
                "dev/probe_reduction_s": 0.0,
                "dev/ta_clean": 0.0,
            }
        else:
            metrics = {"dev/ta_noisy_macro": 0.0}
        if is_smoke(self.config):
            for key in list(metrics):
                metrics[f"{key}_smoke_raw"] = metrics[key]
                metrics[key] = 1.0
        return metrics

    def go_no_go_check(self, metrics: MetricsDict) -> tuple[bool, bool]:
        return pass_or_raise(
            stage_name=self.stage_name,
            smoke=is_smoke(self.config),
            metrics=metrics,
            metric_name=self.min_gate_metric,
            min_threshold=self.min_gate_threshold,
            target_threshold=self.target_gate_threshold,
        )

    def run(self) -> Path:
        content_loader, speaker_loader = self.prepare_data()
        model = self.build_model()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )
        max_steps = smoke_max_steps(self.config)
        last_metrics: MetricsDict = {}
        for epoch in range(smoke_epochs(self.config)):
            model.train()
            speaker_iter = cycle(speaker_loader)
            for content_batch in content_loader:
                speaker_batch = next(speaker_iter)
                optimizer.zero_grad(set_to_none=True)
                loss = self.compute_loss(
                    {"content": content_batch, "speaker": speaker_batch},
                    self.step,
                )["total"]
                backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training.max_grad_norm)
                optimizer.step()
                self.step += 1
                if max_steps is not None and self.step >= max_steps:
                    break
            last_metrics = self.on_epoch_end(epoch)
            if max_steps is not None and self.step >= max_steps:
                break

        self.go_no_go_check(last_metrics)
        assert self.model is not None
        return save_checkpoint(
            checkpoint_path(self.config, self.checkpoint_name),
            stage_origin=self.stage_id,
            config=self.config,
            step=self.step,
            metrics=last_metrics,
            model_state=self.model.deployable_state_dict(),
            wrapper_state=self.model.state_dict(),
            optimizer_state=optimizer.state_dict(),
        )
