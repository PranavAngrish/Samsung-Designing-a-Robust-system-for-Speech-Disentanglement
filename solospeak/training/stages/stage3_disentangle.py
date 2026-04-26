"""Stage 3: disentanglement turn-on."""

from __future__ import annotations

from solospeak.training.stages.common import is_smoke, pass_or_raise
from solospeak.training.stages.dual_head_base import DualHeadTrainingStage
from solospeak.utils.types import MetricsDict


class Stage3(DualHeadTrainingStage):
    stage_id = 3
    stage_name = "stage3_disentangle"
    min_gate_metric = "dev/probe_reduction"
    min_gate_threshold = 0.30
    target_gate_threshold = 0.60
    checkpoint_name = "stage3_disentangle.pt"

    def go_no_go_check(self, metrics: MetricsDict) -> tuple[bool, bool]:
        if is_smoke(self.config):
            metrics["gate/smoke_only"] = 1.0
            return True, True
        reduction_c = metrics.get("dev/probe_reduction_c")
        reduction_s = metrics.get("dev/probe_reduction_s")
        if reduction_c is None or reduction_s is None:
            return pass_or_raise(
                stage_name=self.stage_name,
                smoke=False,
                metrics=metrics,
                metric_name=self.min_gate_metric,
                min_threshold=self.min_gate_threshold,
                target_threshold=self.target_gate_threshold,
            )
        passed_min = reduction_c >= self.min_gate_threshold and reduction_s >= self.min_gate_threshold
        passed_target = (
            reduction_c >= self.target_gate_threshold and reduction_s >= self.target_gate_threshold
        )
        if not passed_min:
            raise RuntimeError(
                f"{self.stage_name}: MIN gate failed for probe reductions: "
                f"content={reduction_c:.4f}, speaker={reduction_s:.4f}"
            )
        return passed_min, passed_target

