"""Stage 1 backbone pretraining stub."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn
from torch.utils.data import DataLoader

from solospeak.training.stages.base import TrainingStage
from solospeak.utils.types import LossDict, MetricsDict, StageBatch


class Stage1(TrainingStage):
    stage_id = 1
    stage_name = "stage1_backbone"
    min_gate_metric = "dev/gsc_accuracy"
    min_gate_threshold = 0.92
    target_gate_threshold = 0.96

    def prepare_data(self) -> tuple[DataLoader[Any], ...]:
        raise NotImplementedError("Implement in Phase 3")

    def build_model(self) -> nn.Module:
        raise NotImplementedError("Implement in Phase 3")

    def compute_loss(self, batches: StageBatch, step: int) -> LossDict:
        raise NotImplementedError("Implement in Phase 3")

    def on_epoch_end(self, epoch: int) -> MetricsDict:
        raise NotImplementedError("Implement in Phase 3")

    def go_no_go_check(self, metrics: MetricsDict) -> tuple[bool, bool]:
        raise NotImplementedError("Implement in Phase 3")

    def run(self) -> Path:
        raise NotImplementedError("Implement in Phase 3")
