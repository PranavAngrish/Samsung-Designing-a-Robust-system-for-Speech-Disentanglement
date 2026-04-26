"""Tests for Phase-6 security and observability utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from solospeak.observability.dp_noise import add_dp_noise
from solospeak.observability.metrics import (
    SoloSpeakLocalMetrics,
    record_enrollment_attempt,
    record_wake_event,
    update_latency,
)
from solospeak.security.replay_eval import run_replay_eval
from solospeak.security.threat_model import documented_threats


def test_local_metrics_to_dict_contract() -> None:
    metrics = SoloSpeakLocalMetrics()
    record_wake_event(metrics, "user_a")
    record_enrollment_attempt(metrics, succeeded=False)
    update_latency(metrics, [10.0, 20.0, 30.0])

    data = metrics.to_dict()
    assert data["wake_events_total"] == 1
    assert data["wake_events_per_user"] == {"user_a": 1}
    assert data["enrollment_attempts"] == 1
    assert data["enrollment_failures"] == 1
    assert data["inference_latency_ms_p50"] == 20.0


def test_dp_noise_laplace_variance() -> None:
    rng = np.random.default_rng(123)
    sensitivity = 2.0
    epsilon = 0.5
    samples = np.asarray(
        [
            add_dp_noise(0.0, sensitivity, epsilon, rng=rng)
            for _ in range(10_000)
        ],
        dtype=np.float64,
    )
    expected_variance = 2.0 * (sensitivity / epsilon) ** 2
    assert abs(float(samples.var()) - expected_variance) / expected_variance < 0.15


def test_dp_noise_rejects_non_positive_epsilon() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        add_dp_noise(1.0, 1.0, 0.0)


def test_documented_threats_include_phase6_scope() -> None:
    threats = set(documented_threats())
    assert "replay_attacks" in threats
    assert "profile_theft" in threats
    assert "multi_user_confusion" in threats


def test_replay_eval_smoke_numeric_output(tmp_path: Path) -> None:
    result = run_replay_eval(Path("missing.pt"), tmp_path / "replay", tmp_path / "profiles")
    assert result["replay_success_rate"] == 0.0
    assert result["num_trials"] == 0.0
    assert result["smoke_only"] == 1.0

