"""Multi-user wake detection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from solospeak.data.features_deploy import LogMelExtractorDeploy
from solospeak.inference.hysteresis import HysteresisDetector
from solospeak.inference.streaming import RingBuffer, _session_from_path, _template
from solospeak.utils.config import AudioConfig
from solospeak.utils.types import FloatArray, UserProfile, WakeEvent


class MultiUserDetector:
    """Evaluate one mel window against multiple enrolled profiles."""

    def __init__(
        self,
        onnx_session: object,
        user_profiles: dict[str, UserProfile],
    ) -> None:
        self.session: Any = onnx_session
        self.user_profiles = dict(user_profiles)
        self.hysteresis = {
            user_id: HysteresisDetector(
                tau_on=profile.tau,
                tau_off=max(0.05, profile.tau - 0.20),
                refractory_s=0.25,
                stride_s=0.16,
            )
            for user_id, profile in self.user_profiles.items()
        }
        self.timestamp_ms = 0

    def _run_profile(self, mel: FloatArray, profile: UserProfile) -> tuple[float, float, float]:
        mel_input = mel.astype(np.float32, copy=False)
        if mel_input.ndim == 3:
            mel_input = mel_input[None, :, :, :]
        outputs = self.session.run(
            None,
            {
                "mel": mel_input,
                "content_template": _template(profile.content_template),
                "speaker_template": _template(profile.speaker_template),
            },
        )
        return (
            float(np.asarray(outputs[2]).reshape(-1)[0]),
            float(np.asarray(outputs[3]).reshape(-1)[0]),
            float(np.asarray(outputs[4]).reshape(-1)[0]),
        )

    def step(self, mel: FloatArray) -> WakeEvent | None:
        candidates: list[WakeEvent] = []
        for user_id, profile in self.user_profiles.items():
            content_score, speaker_score, fusion_score = self._run_profile(mel, profile)
            if self.hysteresis[user_id].step(fusion_score):
                candidates.append(
                    WakeEvent(
                        user_id=profile.user_id,
                        keyword_text=profile.keyword_text,
                        timestamp_ms=self.timestamp_ms,
                        fusion_score=fusion_score,
                        content_score=content_score,
                        speaker_score=speaker_score,
                    )
                )
        self.timestamp_ms += 160
        if not candidates:
            return None
        return max(candidates, key=lambda event: event.fusion_score)

    def add_user(self, profile: UserProfile) -> None:
        self.user_profiles[profile.user_id] = profile
        self.hysteresis[profile.user_id] = HysteresisDetector(
            tau_on=profile.tau,
            tau_off=max(0.05, profile.tau - 0.20),
            refractory_s=0.25,
            stride_s=0.16,
        )

    def remove_user(self, user_id: str) -> None:
        self.user_profiles.pop(user_id, None)
        self.hysteresis.pop(user_id, None)


class MultiUserInference:
    """Streaming multi-user detector with one public ``step`` method."""

    def __init__(self, onnx_path: Path) -> None:
        self.session = _session_from_path(onnx_path)
        self.audio_config = AudioConfig()
        self.extractor = LogMelExtractorDeploy(self.audio_config)
        self.ring = RingBuffer(self.audio_config.window_samples)
        self.detector = MultiUserDetector(self.session, {})

    def add_user(self, profile: UserProfile) -> None:
        self.detector.add_user(profile)

    def remove_user(self, user_id: str) -> None:
        self.detector.remove_user(user_id)

    def active_users(self) -> list[str]:
        return list(self.detector.user_profiles)

    def step(self, audio_frame: FloatArray) -> list[WakeEvent]:
        frame = np.asarray(audio_frame, dtype=np.float32).reshape(-1)
        self.ring.push(frame)
        if float(np.sqrt(np.mean(frame**2))) < 1e-4:
            return []
        mel = self.extractor(self.ring.last(self.audio_config.window_samples))
        event = self.detector.step(mel)
        return [] if event is None else [event]
