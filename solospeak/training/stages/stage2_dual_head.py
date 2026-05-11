"""Stage 2: dual-head joint training."""

from __future__ import annotations

from itertools import cycle
from pathlib import Path

import torch

from solospeak.training.stages.common import (
    backward,
    checkpoint_path,
    is_smoke,
    save_checkpoint,
    smoke_epochs,
    smoke_max_steps,
)
from solospeak.training.stages.dual_head_base import DualHeadTrainingStage
from solospeak.utils.types import MetricsDict


class Stage2(DualHeadTrainingStage):
    stage_id = 2
    stage_name = "stage2_dual_head"
    min_gate_metric = "dev/ta_clean"
    min_gate_threshold = 0.90
    target_gate_threshold = 0.96
    checkpoint_name = "stage2_dualhead.pt"

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
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config.training.max_grad_norm,
                )
                optimizer.step()
                self.step += 1
                if max_steps is not None and self.step >= max_steps:
                    break
            last_metrics = self.on_epoch_end(epoch)
            if max_steps is not None and self.step >= max_steps:
                break

        if self.model is None:
            raise RuntimeError("build_model() must be called before Stage2 save")

        last_path = save_checkpoint(
            checkpoint_path(self.config, "stage2_last.pt"),
            stage_origin=self.stage_id,
            config=self.config,
            step=self.step,
            metrics=last_metrics,
            model_state=self.model.deployable_state_dict(),
            wrapper_state=self.model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            extra={"gate_checked": False},
        )

        try:
            self.go_no_go_check(last_metrics)
        except RuntimeError:
            if is_smoke(self.config):
                raise
            return last_path

        return save_checkpoint(
            checkpoint_path(self.config, self.checkpoint_name),
            stage_origin=self.stage_id,
            config=self.config,
            step=self.step,
            metrics=last_metrics,
            model_state=self.model.deployable_state_dict(),
            wrapper_state=self.model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            extra={"gate_checked": True},
        )
