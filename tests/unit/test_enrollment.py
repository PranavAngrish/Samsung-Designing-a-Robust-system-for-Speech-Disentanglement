"""Unit tests for enrollment utilities."""

from __future__ import annotations

import numpy as np
import pytest

from solospeak.enrollment.service import compute_mean_template
from solospeak.enrollment.calibration import calibrate_threshold
from solospeak.enrollment.templates import save_profile, load_profile
from solospeak.utils.types import UserProfile


def test_compute_mean_template_is_l2_normalized() -> None:
    embeddings = [np.random.randn(128).astype(np.float32) for _ in range(5)]
    template = compute_mean_template(embeddings)
    norm = np.linalg.norm(template)
    assert abs(norm - 1.0) < 1e-5


def test_compute_mean_template_shape() -> None:
    embeddings = [np.random.randn(128).astype(np.float32) for _ in range(3)]
    template = compute_mean_template(embeddings)
    assert template.shape == (128,)


def test_calibrate_threshold_between_rejection_and_confidence() -> None:
    tau = calibrate_threshold(rejection_score=0.3, high_confidence=0.9)
    assert 0.3 < tau < 0.9


def test_save_load_profile_roundtrip(tmp_path: "pathlib.Path") -> None:
    c = np.random.randn(128).astype(np.float32)
    s = np.random.randn(128).astype(np.float32)
    c /= np.linalg.norm(c)
    s /= np.linalg.norm(s)

    profile = UserProfile(
        user_id="user_001",
        keyword_text="hey prism",
        content_template=c,
        speaker_template=s,
        tau=0.7,
        model_version="solospeak-v1.0.0",
    )
    path = tmp_path / "profile.json"
    save_profile(profile, path)
    loaded = load_profile(path)

    assert loaded.user_id == profile.user_id
    assert loaded.keyword_text == profile.keyword_text
    assert loaded.tau == profile.tau
    np.testing.assert_allclose(loaded.content_template, profile.content_template, atol=1e-6)
    np.testing.assert_allclose(loaded.speaker_template, profile.speaker_template, atol=1e-6)
