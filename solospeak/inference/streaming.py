"""Production streaming inference loop.

Processes audio in 100 ms strides over a 1.5 s ring buffer.
VAD gates the expensive encoder — skips when no speech detected.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from solospeak.inference.hysteresis import HysteresisDetector
from solospeak.utils.types import UserProfile, WakeEvent


class RingBuffer:
    """Fixed-size circular buffer for audio samples."""

    def __init__(self, capacity: int) -> None:
        self._buf: deque[float] = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, chunk: np.ndarray) -> None:
        self._buf.extend(chunk.tolist())

    def last(self, n: int) -> np.ndarray:
        buf = list(self._buf)
        if len(buf) < n:
            pad = np.zeros(n - len(buf), dtype=np.float32)
            return np.concatenate([pad, np.array(buf, dtype=np.float32)])
        return np.array(buf[-n:], dtype=np.float32)


class StreamingDetector:
    """Single-user streaming detector.

    📋 CONTRACT
        step(chunk: np.ndarray) → WakeEvent | None
        chunk: 100 ms of audio (1600 samples at 16 kHz)
    """

    def __init__(
        self,
        onnx_session: object,
        profile: UserProfile,
        sr: int = 16000,
    ) -> None:
        raise NotImplementedError("Implement in Phase 5")

    def step(self, chunk: np.ndarray) -> WakeEvent | None:
        raise NotImplementedError("Implement in Phase 5")
