"""Stage 6 quantization-aware fine-tuning stub."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn
from torch.utils.data import DataLoader

from solospeak.training.stages.base import TrainingStage
from solospeak.utils.types import LossDict, MetricsDict, StageBatch


class Stage6(TrainingStage):
    stage_id = 6
    stage_name = "stage6_qat"
    min_gate_metric = "dev/int8_vs_fp32_degradation_pp"
    min_gate_threshold = 1.0
    target_gate_threshold = 0.3

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


def copy_float_weights_from_prepared(prepared: nn.Module, target: nn.Module) -> None:
    """Copy float weights from a QAT-prepared model into a clean target model."""
    raise NotImplementedError("Implement in Phase 3")
