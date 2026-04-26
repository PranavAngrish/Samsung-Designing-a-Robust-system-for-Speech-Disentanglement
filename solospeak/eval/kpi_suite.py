"""Full Samsung KPI evaluation suite.

One call, one JSON output, everything a judge needs.
All numbers in the final report flow from this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING


from solospeak.utils.types import KPIResult

if TYPE_CHECKING:
    from solospeak.utils.config import SoloSpeakConfig


def run_kpi_suite(
    checkpoint_path: Path,
    config: "SoloSpeakConfig",
    test_manifest: Path,
    fa_manifest: Path,
    seed: int = 42,
) -> KPIResult:
    """Run all Samsung KPIs and return a KPIResult.

    📋 CONTRACT
        Evaluates on held-out test set only — never dev set.
        Returns ta_clean, ta_noisy (per SNR), fa_per_hour,
                q2_rejection_rate, q3_rejection_rate,
                param_count, xrt_fp32, xrt_int8.
    """
    raise NotImplementedError("Implement in Phase 4")


def measure_ta_clean(model: object, test_manifest: Path, n_pairs: int = 500) -> float:
    """True Acceptance on clean audio. Target: >= 99%."""
    raise NotImplementedError("Implement in Phase 4")


def measure_ta_noisy(
    model: object,
    test_manifest: Path,
    snr_points: list[int],
    noise_types: list[str],
) -> dict[int, float]:
    """True Acceptance at each SNR bucket. Target macro-avg: >= 90%."""
    raise NotImplementedError("Implement in Phase 4")


def measure_fa_rate(model: object, fa_manifest: Path, eval_hours: float = 10.0) -> float:
    """False Acceptance per hour on continuous background audio. Target: < 1/hr."""
    raise NotImplementedError("Implement in Phase 4")


def measure_q2_rejection(model: object, test_manifest: Path) -> float:
    """Rejection rate on imposter utterances (correct word, wrong speaker). Target: > 95%."""
    raise NotImplementedError("Implement in Phase 4")


def measure_q3_rejection(model: object, test_manifest: Path) -> float:
    """Rejection rate on phonetically similar words from correct speaker. Target: > 90%."""
    raise NotImplementedError("Implement in Phase 4")
