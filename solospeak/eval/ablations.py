"""7-configuration ablation runner.

Ablation configs (all start from Stage 2 shared checkpoint):
    full           — reference (all components active)
    no_ortho       — orthogonality loss off
    no_adv         — adversarial probes off
    no_disent      — both no_ortho + no_adv
    no_tts_enroll  — no TTS augmentation at enrollment
    no_gated_fusion — hand-coded min(s_c, s_s) combiner
    no_curriculum  — uniform SNR sampling instead of curriculum
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    "full": {},
    "no_ortho": {"losses": {"orthogonality": 0.0}},
    "no_adv": {"losses": {"adversarial": 0.0}},
    "no_disent": {"losses": {"orthogonality": 0.0, "adversarial": 0.0}},
    "no_tts_enroll": {"enrollment": {"tts_n_variants": 0}},
    "no_gated_fusion": {"fusion": {"mode": "min"}},
    "no_curriculum": {"data": {"curriculum": False}},
}


def run_all_ablations(
    base_checkpoint: Path,
    output_dir: Path,
    seeds: list[int] | None = None,
) -> dict[str, dict[str, float]]:
    """Run all ablation configs × seeds. Returns results keyed by ablation name."""
    raise NotImplementedError("Implement in Phase 4")


def format_ablation_table(results: dict[str, dict[str, float]]) -> str:
    """Render results as a Markdown table for the final report."""
    raise NotImplementedError("Implement in Phase 4")
