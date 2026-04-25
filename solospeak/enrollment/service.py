"""Production enrollment flow.

Enrollment steps:
    1. Validate recordings (length, SNR, clipping)
    2. Optional TTS augmentation via Parler-TTS
    3. Embed each clip through shared encoder
    4. Compute mean content template C_w and speaker template S_u (L2-normalized)
    5. Calibrate per-user threshold on one rejection sample
    6. Persist profile (~1.1 KB)
"""

from __future__ import annotations

import numpy as np

from solospeak.utils.types import UserProfile


def enroll(
    user_id: str,
    keyword_text: str,
    recordings: list[np.ndarray],
    encoder: object,
    tts_n_variants: int = 10,
    sr: int = 16000,
) -> UserProfile:
    """Run the full enrollment pipeline and return a UserProfile.

    📋 CONTRACT
        recordings: list of raw float32 mono waveforms at sr Hz
        Returns UserProfile with content_template, speaker_template, tau
    """
    raise NotImplementedError("Implement in Phase 5")


def compute_mean_template(embeddings: list[np.ndarray]) -> np.ndarray:
    """Average embeddings and L2-normalize the result."""
    stack = np.stack(embeddings, axis=0)         # (N, 128)
    mean = stack.mean(axis=0)                     # (128,)
    norm = np.linalg.norm(mean)
    return np.asarray(mean / max(norm, 1e-8), dtype=np.float32)
