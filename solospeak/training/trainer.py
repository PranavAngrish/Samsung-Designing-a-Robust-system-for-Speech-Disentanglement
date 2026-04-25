"""Orchestrator: runs training stages sequentially, handles checkpoint handoff.

Usage:
    trainer = Trainer(config)
    trainer.run_stage(1)
    trainer.run_all_stages()
    trainer.resume_from_stage(3)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from solospeak.utils.config import SoloSpeakConfig

# Checkpoint naming convention
STAGE_CHECKPOINTS = {
    1: "stage1_backbone.pt",
    2: "stage2_dualhead.pt",
    3: "stage3_disentangle.pt",
    4: "stage4_robust.pt",
    5: "stage5_fusion.pt",
    6: "stage6_qat.pt",
}


class Trainer:
    def __init__(self, config: "SoloSpeakConfig") -> None:
        self.config = config
        self.checkpoint_dir = config.training.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def run_stage(self, stage_id: int) -> Path:
        """Run a single stage. Returns the saved checkpoint path."""
        raise NotImplementedError("Implement in Phase 3")

    def run_all_stages(self) -> None:
        """Run stages 1–6 sequentially. Abort if any GO/NO-GO gate fails."""
        raise NotImplementedError("Implement in Phase 3")

    def resume_from_stage(self, stage_id: int) -> None:
        """Load checkpoint from stage_id - 1, then run stage_id onwards."""
        raise NotImplementedError("Implement in Phase 3")

    def _checkpoint_path(self, stage_id: int) -> Path:
        return self.checkpoint_dir / STAGE_CHECKPOINTS[stage_id]

    def _checkpoint_exists(self, stage_id: int) -> bool:
        return self._checkpoint_path(stage_id).exists()
