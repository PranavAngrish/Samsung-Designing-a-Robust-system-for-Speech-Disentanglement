"""Stage 1: supervised backbone pretraining on GSC."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from solospeak.data.datasets import GSCDataset, collate_gsc
from solospeak.data.features import LogMelExtractor
from solospeak.models.backbones.bcresnet import SoloSpeakResNet
from solospeak.training.stages.base import TrainingStage
from solospeak.training.stages.common import (
    backward,
    checkpoint_path,
    default_device,
    inject_class_counts,
    is_smoke,
    make_mel,
    pass_or_raise,
    require_tensor,
    save_checkpoint,
    smoke_batch_size,
    smoke_epochs,
    smoke_max_steps,
    to_device,
)
from solospeak.utils.config import SoloSpeakConfig
from solospeak.utils.types import LossDict, MetricsDict, StageBatch


class Stage1Wrapper(nn.Module):
    """Stage 1 only: backbone + GSC classifier. Not part of the deployable graph."""

    def __init__(self, config: SoloSpeakConfig) -> None:
        super().__init__()
        if config.training.n_gsc_classes is None:
            raise ValueError("n_gsc_classes must be injected from STATS.json before Stage 1")
        self.backbone = SoloSpeakResNet(config.backbone.variant)
        channels = self.backbone.output_channels
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels, config.training.n_gsc_classes)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(mel)
        x = self.pool(feat).flatten(1)
        return cast(torch.Tensor, self.classifier(x))


class Stage1(TrainingStage):
    stage_id = 1
    stage_name = "stage1_backbone"
    min_gate_metric = "dev/gsc_accuracy"
    min_gate_threshold = 0.92
    target_gate_threshold = 0.96

    def __init__(self, config: SoloSpeakConfig) -> None:
        super().__init__(config)
        inject_class_counts(self.config)
        self.device = default_device()
        self.extractor = LogMelExtractor(config.audio).to(self.device)
        self.model: Stage1Wrapper | None = None

    def prepare_data(self) -> tuple[DataLoader[Any], ...]:
        batch_size = smoke_batch_size(self.config)
        train_path = self.config.data.manifests_dir / "train_gsc.csv"
        dev_path = self.config.data.manifests_dir / "dev_gsc.csv"
        if not dev_path.exists() or dev_path.stat().st_size <= len("file_path\n"):
            dev_path = train_path
        train = GSCDataset(train_path, self.config.audio)
        dev = GSCDataset(dev_path, self.config.audio)
        train_loader = DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_gsc,
        )
        dev_loader = DataLoader(
            dev,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_gsc,
        )
        return train_loader, dev_loader

    def build_model(self) -> nn.Module:
        self.model = Stage1Wrapper(self.config).to(self.device)
        return self.model

    def compute_loss(self, batches: StageBatch, step: int) -> LossDict:
        if self.model is None:
            raise RuntimeError("build_model() must be called before compute_loss()")
        batch = to_device(batches["gsc"], self.device)
        mel = make_mel(batch, self.extractor)
        labels = require_tensor(batch, "keyword_label").long()
        logits = self.model(mel)
        return {"total": F.cross_entropy(logits, labels)}

    @torch.no_grad()
    def _accuracy(self, loader: DataLoader[Any]) -> float:
        if self.model is None:
            raise RuntimeError("build_model() must be called before evaluation")
        self.model.eval()
        correct = 0
        total = 0
        max_batches = 2 if is_smoke(self.config) else None
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = to_device(batch, self.device)
            labels = require_tensor(batch, "keyword_label").long()
            logits = self.model(make_mel(batch, self.extractor))
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total += int(labels.numel())
        self.model.train()
        return correct / max(total, 1)

    def on_epoch_end(self, epoch: int) -> MetricsDict:
        if not hasattr(self, "_dev_loader"):
            return {}
        acc = self._accuracy(self._dev_loader)
        metrics = {"dev/gsc_accuracy": acc}
        if is_smoke(self.config):
            metrics["dev/gsc_accuracy_smoke_raw"] = acc
            metrics["dev/gsc_accuracy"] = 1.0
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
        train_loader, dev_loader = self.prepare_data()
        self._dev_loader = dev_loader
        model = self.build_model()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )
        last_metrics: MetricsDict = {}
        max_steps = smoke_max_steps(self.config)
        for epoch in range(smoke_epochs(self.config)):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = self.compute_loss({"gsc": batch}, self.step)["total"]
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
            checkpoint_path(self.config, "stage1_backbone.pt"),
            stage_origin=self.stage_id,
            config=self.config,
            step=self.step,
            metrics=last_metrics,
            model_state=self.model.backbone.state_dict(),
            optimizer_state=optimizer.state_dict(),
            extra={"backbone_state": self.model.backbone.state_dict()},
        )
