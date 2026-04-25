"""FGSM adversarial audio evaluation on mel-spectrogram input.

Not a blocker for Phase 1. Listed to show Samsung judges it's on the roadmap.
Pass criterion: quantify the epsilon budget required for 50% FAR.
"""

from __future__ import annotations

from pathlib import Path


def run_fgsm_eval(
    checkpoint_path: Path,
    test_manifest: Path,
    epsilon_values: list[float] | None = None,
) -> dict[float, float]:
    """Run FGSM attack at each epsilon. Returns {epsilon: far_at_50pct_fpr}."""
    raise NotImplementedError("Implement in Phase 6")
