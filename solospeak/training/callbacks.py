"""Training callbacks: W&B logging, checkpoint saving, early stopping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class WandbCallback:
    """Weights & Biases logging callback.

    Logs every N steps: train/loss_*, train/lr, train/grad_norm, gpu/*
    Logs every epoch: dev/ta_clean, dev/ta_noisy_avg, dev/fa_per_hour,
                      dev/probe_speaker_on_content, dev/probe_word_on_speaker
    """

    def __init__(self, project: str = "solospeak", run_name: str = "run") -> None:
        raise NotImplementedError("Implement in Phase 3")

    def log_step(self, step: int, metrics: dict[str, float]) -> None:
        raise NotImplementedError("Implement in Phase 3")

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        raise NotImplementedError("Implement in Phase 3")

    def finish(self) -> None:
        raise NotImplementedError("Implement in Phase 3")


class CheckpointCallback:
    """Save model checkpoints every N epochs and on GO/NO-GO pass."""

    def __init__(self, checkpoint_dir: Path, stage_id: int, keep_last_n: int = 3) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.stage_id = stage_id
        self.keep_last_n = keep_last_n

    def save(
        self,
        model: nn.Module,
        step: int,
        metrics: dict[str, Any],
        config_dict: dict[str, Any],
        filename: str | None = None,
    ) -> Path:
        raise NotImplementedError("Implement in Phase 3")


class EarlyStoppingCallback:
    """Stop training if monitored metric stops improving."""

    def __init__(self, metric: str, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self._best: float = float("-inf")
        self._no_improve = 0

    def should_stop(self, metrics: dict[str, float]) -> bool:
        value = metrics.get(self.metric, float("-inf"))
        if value > self._best + self.min_delta:
            self._best = value
            self._no_improve = 0
            return False
        self._no_improve += 1
        return self._no_improve >= self.patience
