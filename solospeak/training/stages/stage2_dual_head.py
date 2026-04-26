"""Stage 2: dual-head joint training."""

from __future__ import annotations

from solospeak.training.stages.dual_head_base import DualHeadTrainingStage


class Stage2(DualHeadTrainingStage):
    stage_id = 2
    stage_name = "stage2_dual_head"
    min_gate_metric = "dev/ta_clean"
    min_gate_threshold = 0.90
    target_gate_threshold = 0.96
    checkpoint_name = "stage2_dualhead.pt"

