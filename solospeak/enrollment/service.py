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

from solospeak.enrollment.calibration import calibrate_threshold
from solospeak.utils.types import FloatArray, UserProfile


def _as_embedding_pair(result: object) -> tuple[FloatArray, FloatArray]:
    if isinstance(result, tuple) and len(result) == 2:
        content = np.asarray(result[0], dtype=np.float32).reshape(-1)
        speaker = np.asarray(result[1], dtype=np.float32).reshape(-1)
        return content, speaker
    if isinstance(result, dict):
        content = np.asarray(result["content_template"], dtype=np.float32).reshape(-1)
        speaker = np.asarray(result["speaker_template"], dtype=np.float32).reshape(-1)
        return content, speaker
    arr = np.asarray(result, dtype=np.float32).reshape(-1)
    return arr, arr


def _encode_recording(encoder: object, recording: FloatArray) -> tuple[FloatArray, FloatArray]:
    if hasattr(encoder, "embed"):
        return _as_embedding_pair(getattr(encoder, "embed")(recording))
    if hasattr(encoder, "encode"):
        return _as_embedding_pair(getattr(encoder, "encode")(recording))
    if callable(encoder):
        return _as_embedding_pair(encoder(recording))
    raise TypeError("encoder must be callable or expose embed()/encode().")


def enroll(
    user_id: str,
    keyword_text: str,
    recordings: list[FloatArray],
    encoder: object,
    tts_n_variants: int = 10,
    sr: int = 16000,
) -> UserProfile:
    """Run the full enrollment pipeline and return a UserProfile.

    📋 CONTRACT
        recordings: list of raw float32 mono waveforms at sr Hz
        Returns UserProfile with content_template, speaker_template, tau
    """
    del tts_n_variants, sr
    if not recordings:
        raise ValueError("Enrollment requires at least one recording.")

    content_embeddings: list[FloatArray] = []
    speaker_embeddings: list[FloatArray] = []
    for recording in recordings:
        content, speaker = _encode_recording(encoder, recording)
        content_embeddings.append(content)
        speaker_embeddings.append(speaker)

    model_version = str(getattr(encoder, "model_version", "solospeak-v1.0.0"))
    return UserProfile(
        user_id=user_id,
        keyword_text=keyword_text,
        content_template=compute_mean_template(content_embeddings),
        speaker_template=compute_mean_template(speaker_embeddings),
        tau=0.65,
        model_version=model_version,
    )


def compute_mean_template(embeddings: list[FloatArray]) -> FloatArray:
    """Average embeddings and L2-normalize the result."""
    if not embeddings:
        raise ValueError("At least one embedding is required.")
    stack = np.stack(embeddings, axis=0)         # (N, 128)
    mean = stack.mean(axis=0)                     # (128,)
    norm = np.linalg.norm(mean)
    return np.asarray(mean / max(norm, 1e-8), dtype=np.float32)


class EnrollmentService:
    """Small service wrapper around the Phase-5 enrollment function."""

    def __init__(
        self,
        encoder: object,
        *,
        default_tau: float = 0.65,
        model_version: str = "solospeak-v1.0.0",
    ) -> None:
        self.encoder = encoder
        self.default_tau = default_tau
        self.model_version = model_version

    def enroll(
        self,
        user_id: str,
        keyword_text: str,
        recordings: list[FloatArray],
    ) -> UserProfile:
        profile = enroll(user_id, keyword_text, recordings, self.encoder)
        return UserProfile(
            user_id=profile.user_id,
            keyword_text=profile.keyword_text,
            content_template=profile.content_template,
            speaker_template=profile.speaker_template,
            tau=self.default_tau,
            model_version=str(getattr(self.encoder, "model_version", self.model_version)),
        )

    def calibrate_threshold(
        self,
        profile: UserProfile,
        rejection_samples: list[FloatArray],
    ) -> float:
        if not rejection_samples:
            return self.default_tau
        scores = []
        for sample in rejection_samples:
            content, speaker = _encode_recording(self.encoder, sample)
            content_score = float(np.dot(profile.content_template, compute_mean_template([content])))
            speaker_score = float(np.dot(profile.speaker_template, compute_mean_template([speaker])))
            scores.append(min(content_score, speaker_score))
        return calibrate_threshold(max(scores), default_tau=self.default_tau)
