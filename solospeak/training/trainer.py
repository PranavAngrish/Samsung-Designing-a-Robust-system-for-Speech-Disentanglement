"""Phase-3 stage orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from solospeak.training.stages.base import TrainingStage
    from solospeak.utils.config import SoloSpeakConfig

from solospeak.training.stages.common import inject_class_counts

STAGE_CHECKPOINTS = {
    1: "stage1_backbone.pt",
    2: "stage2_dualhead.pt",
    3: "stage3_disentangle.pt",
    4: "stage4_robust.pt",
    5: "stage5_fusion.pt",
    6: "stage6_final.pt",
    7: "stage7_fusion.pt",
}


def _stage_class(stage_id: int) -> type["TrainingStage"]:
    if stage_id == 1:
        from solospeak.training.stages.stage1_backbone import Stage1

        return Stage1
    if stage_id == 2:
        from solospeak.training.stages.stage2_dual_head import Stage2

        return Stage2
    if stage_id == 3:
        from solospeak.training.stages.stage3_disentangle import Stage3

        return Stage3
    if stage_id == 4:
        from solospeak.training.stages.stage4_robustness import Stage4

        return Stage4
    if stage_id == 5:
        from solospeak.training.stages.stage5_fusion import Stage5

        return Stage5
    if stage_id == 6:
        from solospeak.training.stages.stage6_final_eval import Stage6FinalEval

        return Stage6FinalEval
    if stage_id == 7:
        from solospeak.training.stages.stage7_fa_tuning import Stage7

        return Stage7
    raise ValueError(f"Unknown stage_id {stage_id!r}")


class Trainer:
    """Run one stage with MIN/TARGET gate semantics."""

    def __init__(self, config: "SoloSpeakConfig") -> None:
        self.config = config
        inject_class_counts(self.config)
        self.checkpoint_dir = config.training.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def run_stage(self, stage_id: int) -> Path:
        """Run a single stage and return its checkpoint path."""
        self.config.training.stage = stage_id  # type: ignore[assignment]
        stage_cls = _stage_class(stage_id)
        stage = stage_cls(self.config)
        return stage.run()

    def run_all_stages(self) -> list[Path]:
        """Run stages 1-7 sequentially. Abort only on MIN-gate failure."""
        checkpoints: list[Path] = []
        for stage_id in range(1, 8):
            if stage_id > 1:
                self.config.training.resume_from = self._checkpoint_path(stage_id - 1)
            checkpoints.append(self.run_stage(stage_id))
        return checkpoints

    def resume_from_stage(self, stage_id: int) -> list[Path]:
        """Run ``stage_id`` through Stage 7, loading the previous checkpoint first."""
        if stage_id < 1 or stage_id > 7:
            raise ValueError("stage_id must be in [1, 7]")
        checkpoints: list[Path] = []
        for current in range(stage_id, 8):
            if current > 1:
                self.config.training.resume_from = self._checkpoint_path(current - 1)
            checkpoints.append(self.run_stage(current))
        return checkpoints

    def _checkpoint_path(self, stage_id: int) -> Path:
        return self.checkpoint_dir / STAGE_CHECKPOINTS[stage_id]

    def _checkpoint_exists(self, stage_id: int) -> bool:
        return self._checkpoint_path(stage_id).exists()
