"""Per-user decision threshold calibration.

The default fusion threshold (tau=0.5) is tuned per-user at enrollment time
using one rejection sample. This reduces false accepts for users whose voice
sits at an unusual location in the embedding space.
"""

from __future__ import annotations

from solospeak.utils.types import FloatArray


def calibrate_threshold(
    rejection_score: float,
    default_tau: float = 0.5,
    high_confidence: float = 0.9,
) -> float:
    """Set tau as midpoint between the rejection sample score and high confidence.

    Ensures the threshold is above the rejection score but below certainty.
    """
    return 0.5 * (rejection_score + high_confidence)


def estimate_rejection_score(
    rejection_audio: FloatArray,
    content_template: FloatArray,
    speaker_template: FloatArray,
    encoder: object,
    fusion: object,
    sr: int = 16000,
) -> float:
    """Run the rejection audio through the full pipeline and return the fusion score."""
    raise NotImplementedError("Implement in Phase 5")
