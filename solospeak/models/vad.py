"""Silero-VAD v4 wrapper.

VAD parameters are NOT counted toward the <3M budget — it's a universal preprocessor.
Role: gate the expensive encoder. If no speech in last 500 ms, skip inference entirely.
"""

from __future__ import annotations

import numpy as np
import torch


class SileroVAD:
    """Thin wrapper around Silero-VAD v4 (MIT license, ~1 MB).

    📋 CONTRACT
        is_speech(chunk: np.ndarray, sr: int) → bool
        chunk should be 30 ms of audio (480 samples at 16 kHz)
    """

    def __init__(self, threshold: float = 0.5, sr: int = 16000) -> None:
        raise NotImplementedError("Implement in Phase 1")

    def is_speech(self, chunk: np.ndarray, sr: int = 16000) -> bool:
        raise NotImplementedError("Implement in Phase 1")

    def reset_states(self) -> None:
        raise NotImplementedError("Implement in Phase 1")
