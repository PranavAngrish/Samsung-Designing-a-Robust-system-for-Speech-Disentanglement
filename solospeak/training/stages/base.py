"""Abstract base class for all training stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from solospeak.utils.config import SoloSpeakConfig


class TrainingStage(ABC):
    stage_id: int
    stage_name: str
    go_no_go_metric: str
    go_no_go_threshold: float

    def __init__(self, config: "SoloSpeakConfig") -> None:
        self.config = config
        self.step = 0

    @abstractmethod
    def prepare_data(self) -> tuple[DataLoader, DataLoader]: ...

    @abstractmethod
    def build_model(self) -> nn.Module: ...

    @abstractmethod
    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...

    @abstractmethod
    def on_epoch_end(self, epoch: int) -> dict[str, float]: ...

    @abstractmethod
    def go_no_go_check(self) -> bool: ...

    def run(self) -> Path:
        """Main training loop. Returns path to saved checkpoint."""
        raise NotImplementedError("Implement in Phase 3")
