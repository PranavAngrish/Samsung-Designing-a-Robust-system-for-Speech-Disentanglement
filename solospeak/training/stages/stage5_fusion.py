"""Stage 5 fusion-MLP training stub."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn
from torch.utils.data import DataLoader

from solospeak.training.stages.base import TrainingStage
from solospeak.utils.types import LossDict, MetricsDict, StageBatch


class Stage5(TrainingStage):
    stage_id = 5
    stage_name = "stage5_fusion"
    min_gate_metric = "dev/quadrant_accuracy_min"
    min_gate_threshold = 0.85
    target_gate_threshold = 0.95

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
