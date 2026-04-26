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
        self.project = project
        self.run_name = run_name
        self._run: Any | None = None
        try:
            import wandb

            self._run = wandb.init(project=project, name=run_name, mode="disabled")
        except Exception:
            self._run = None

    def log_step(self, step: int, metrics: dict[str, float]) -> None:
        if self._run is not None:
            self._run.log(metrics, step=step)

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        if self._run is not None:
            self._run.log({f"epoch/{key}": value for key, value in metrics.items()}, step=epoch)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()


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
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        name = filename or f"stage{self.stage_id}_step{step}.pt"
        path = self.checkpoint_dir / name
        torch.save(
            {
                "model_state": model.state_dict(),
                "step": step,
                "metrics": metrics,
                "config_snapshot": config_dict,
                "stage_origin": self.stage_id,
            },
            path,
        )
        self._prune_old_checkpoints()
        return path

    def _prune_old_checkpoints(self) -> None:
        if self.keep_last_n <= 0:
            return
        pattern = f"stage{self.stage_id}_step*.pt"
        checkpoints = sorted(
            self.checkpoint_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in checkpoints[self.keep_last_n :]:
            old.unlink(missing_ok=True)


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
