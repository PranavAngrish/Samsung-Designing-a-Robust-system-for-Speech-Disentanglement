"""Streaming inference utilities for live wake-word detection."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from solospeak.data.features_deploy import LogMelExtractorDeploy
from solospeak.inference.hysteresis import HysteresisDetector
from solospeak.utils.config import AudioConfig
from solospeak.utils.types import FloatArray, UserProfile, WakeEvent


class RingBuffer:
    """Fixed-size circular buffer for audio samples."""

    def __init__(self, capacity: int) -> None:
        self._buf: deque[float] = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, chunk: FloatArray) -> None:
        self._buf.extend(chunk.astype(np.float32, copy=False).tolist())

    def last(self, n: int) -> FloatArray:
        buf = list(self._buf)
        if len(buf) < n:
            pad = np.zeros(n - len(buf), dtype=np.float32)
            return np.concatenate([pad, np.array(buf, dtype=np.float32)])
        return np.array(buf[-n:], dtype=np.float32)


def _template(value: FloatArray) -> NDArray[np.float32]:
    arr = np.asarray(value, dtype=np.float32).reshape(1, -1)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    return cast(NDArray[np.float32], (arr / np.maximum(norm, np.float32(1e-8))).astype(np.float32))


def _session_from_path(path: Path) -> Any:
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


class StreamingDetector:
    """Single-user streaming detector."""

    def __init__(
        self,
        onnx_session: object,
        profile: UserProfile,
        sr: int = 16000,
    ) -> None:
        self.session: Any = onnx_session
        self.profile = profile
        self.audio_config = AudioConfig(sample_rate=sr)
        self.extractor = LogMelExtractorDeploy(self.audio_config)
        self.ring = RingBuffer(self.audio_config.window_samples)
        tau_off = max(0.05, profile.tau - 0.20)
        self.hysteresis = HysteresisDetector(
            tau_on=profile.tau,
            tau_off=tau_off,
            refractory_s=0.25,
            stride_s=0.16,
        )
        self.timestamp_ms = 0

    def _infer(self, wav: FloatArray) -> tuple[float, float, float]:
        mel = self.extractor(wav)[None, :, :, :].astype(np.float32)
        outputs = self.session.run(
            None,
            {
                "mel": mel,
                "content_template": _template(self.profile.content_template),
                "speaker_template": _template(self.profile.speaker_template),
            },
        )
        content_score = float(np.asarray(outputs[2]).reshape(-1)[0])
        speaker_score = float(np.asarray(outputs[3]).reshape(-1)[0])
        fusion_score = float(np.asarray(outputs[4]).reshape(-1)[0])
        return content_score, speaker_score, fusion_score

    def step(self, chunk: FloatArray) -> WakeEvent | None:
        frame = np.asarray(chunk, dtype=np.float32).reshape(-1)
        self.ring.push(frame)
        self.timestamp_ms += int(round(1000.0 * len(frame) / self.audio_config.sample_rate))
        if float(np.sqrt(np.mean(frame**2))) < 1e-4:
            return None
        content_score, speaker_score, fusion_score = self._infer(
            self.ring.last(self.audio_config.window_samples)
        )
        if not self.hysteresis.step(fusion_score):
            return None
        return WakeEvent(
            user_id=self.profile.user_id,
            keyword_text=self.profile.keyword_text,
            timestamp_ms=self.timestamp_ms,
            fusion_score=fusion_score,
            content_score=content_score,
            speaker_score=speaker_score,
        )


class StreamingInference:
    """Path-based streaming inference API with enroll/remove helpers."""

    def __init__(self, onnx_path: Path) -> None:
        self.session = _session_from_path(onnx_path)
        self.detectors: dict[str, StreamingDetector] = {}

    def enroll_user(self, profile: UserProfile) -> None:
        self.detectors[profile.user_id] = StreamingDetector(self.session, profile)

    def remove_user(self, user_id: str) -> None:
        self.detectors.pop(user_id, None)

    def step(self, audio_frame: FloatArray) -> list[WakeEvent]:
        events = [
            event
            for detector in self.detectors.values()
            if (event := detector.step(audio_frame)) is not None
        ]
        if not events:
            return []
        return [max(events, key=lambda event: event.fusion_score)]
