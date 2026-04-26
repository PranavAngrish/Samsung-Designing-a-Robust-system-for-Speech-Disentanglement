"""Abstract base for training stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from torch import nn
from torch.utils.data import DataLoader

from solospeak.utils.config import SoloSpeakConfig
from solospeak.utils.types import LossDict, MetricsDict, StageBatch


class TrainingStage(ABC):
    stage_id: ClassVar[int]
    stage_name: ClassVar[str]
    min_gate_metric: ClassVar[str]
    min_gate_threshold: ClassVar[float]
    target_gate_threshold: ClassVar[float]

    def __init__(self, config: SoloSpeakConfig) -> None:
        self.config = config
        self.step = 0

    @abstractmethod
    def prepare_data(self) -> tuple[DataLoader[Any], ...]: ...

    @abstractmethod
    def build_model(self) -> nn.Module: ...

    @abstractmethod
    def compute_loss(self, batches: StageBatch, step: int) -> LossDict: ...

    @abstractmethod
    def on_epoch_end(self, epoch: int) -> MetricsDict: ...

    @abstractmethod
    def go_no_go_check(self, metrics: MetricsDict) -> tuple[bool, bool]:
        """Return ``(passed_min, passed_target)``."""
        ...

    @abstractmethod
    def run(self) -> Path:
        """Full training loop. Returns path to saved checkpoint."""
        ...
