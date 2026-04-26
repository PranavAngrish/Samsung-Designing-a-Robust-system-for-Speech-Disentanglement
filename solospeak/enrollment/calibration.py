"""Per-user decision threshold calibration.

The default fusion threshold (tau=0.5) is tuned per-user at enrollment time
using one rejection sample. This reduces false accepts for users whose voice
sits at an unusual location in the embedding space.
"""

from __future__ import annotations

import numpy as np

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
    del sr
    if hasattr(encoder, "embed"):
        result = getattr(encoder, "embed")(rejection_audio)
    elif hasattr(encoder, "encode"):
        result = getattr(encoder, "encode")(rejection_audio)
    elif callable(encoder):
        result = encoder(rejection_audio)
    else:
        raise TypeError("encoder must be callable or expose embed()/encode().")

    if isinstance(result, tuple) and len(result) == 2:
        content = np.asarray(result[0], dtype=np.float32).reshape(-1)
        speaker = np.asarray(result[1], dtype=np.float32).reshape(-1)
    elif isinstance(result, dict):
        content = np.asarray(result["content_template"], dtype=np.float32).reshape(-1)
        speaker = np.asarray(result["speaker_template"], dtype=np.float32).reshape(-1)
    else:
        content = np.asarray(result, dtype=np.float32).reshape(-1)
        speaker = content

    content = content / max(float(np.linalg.norm(content)), 1e-8)
    speaker = speaker / max(float(np.linalg.norm(speaker)), 1e-8)
    c_template = content_template / max(float(np.linalg.norm(content_template)), 1e-8)
    s_template = speaker_template / max(float(np.linalg.norm(speaker_template)), 1e-8)
    content_score = float(np.dot(content, c_template))
    speaker_score = float(np.dot(speaker, s_template))

    if callable(fusion):
        return float(fusion(content_score, speaker_score))
    if hasattr(fusion, "forward"):
        return float(getattr(fusion, "forward")(content_score, speaker_score))
    return min(content_score, speaker_score)
