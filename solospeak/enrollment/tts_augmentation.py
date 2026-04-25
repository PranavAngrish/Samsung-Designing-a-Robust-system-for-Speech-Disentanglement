"""Parler-TTS enrollment augmentation.

Synthesises acoustic variants of the keyword text across speaking rates,
prosodic patterns, and microphone conditions. Only used for the content
template — TTS audio is NOT used for speaker template (different voices).
"""

from __future__ import annotations

import numpy as np

TTS_PROMPTS = [
    "a young female speaker says '{keyword}' clearly",
    "an elderly male speaker says '{keyword}' slowly",
    "'{keyword}' spoken with a British accent",
    "'{keyword}' spoken with an Indian accent",
    "a child says '{keyword}'",
    "'{keyword}' mumbled quickly",
    "'{keyword}' spoken loudly and clearly",
    "a middle-aged female speaker says '{keyword}'",
    "'{keyword}' with background noise",
    "a male speaker with a deep voice says '{keyword}'",
]


def synthesize_variants(keyword_text: str, n: int = 10, sr: int = 16000) -> list[np.ndarray]:
    """Generate n TTS variants of keyword_text using Parler-TTS.

    Returns list of float32 mono waveforms at sr Hz.
    Requires the 'tts' optional dependency: pip install solospeak[tts]
    """
    raise NotImplementedError("Implement in Phase 5 — requires parler-tts")
