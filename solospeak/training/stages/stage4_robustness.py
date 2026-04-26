"""Stage 4: robustness training with heavy waveform augmentation."""

from __future__ import annotations

from solospeak.training.stages.dual_head_base import DualHeadTrainingStage


class Stage4(DualHeadTrainingStage):
    stage_id = 4
    stage_name = "stage4_robustness"
    min_gate_metric = "dev/ta_noisy_macro"
    min_gate_threshold = 0.80
    target_gate_threshold = 0.88
    checkpoint_name = "stage4_robust.pt"

