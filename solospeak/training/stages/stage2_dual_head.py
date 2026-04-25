"""Stage 2 training stage — implement in Phase 3."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from solospeak.training.stages.base import TrainingStage

if TYPE_CHECKING:
    from solospeak.utils.config import SoloSpeakConfig


class Stage2(TrainingStage):
    stage_id = 2
    stage_name = "stage2_dual_head"

    def prepare_data(self) -> tuple[DataLoader, DataLoader]:
        raise NotImplementedError("Implement in Phase 3")

    def build_model(self) -> torch.nn.Module:
        raise NotImplementedError("Implement in Phase 3")

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Implement in Phase 3")

    def on_epoch_end(self, epoch: int) -> dict[str, float]:
        raise NotImplementedError("Implement in Phase 3")

    def go_no_go_check(self) -> bool:
        raise NotImplementedError("Implement in Phase 3")
