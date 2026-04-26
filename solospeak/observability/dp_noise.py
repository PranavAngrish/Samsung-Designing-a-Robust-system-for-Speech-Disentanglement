"""Laplace mechanism for future opt-in telemetry uploads.

SoloSpeak v1.0 does not upload telemetry. This module documents and tests the
planned privacy mechanism for aggregate scalar metrics.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator


def add_dp_noise(
    value: float,
    sensitivity: float,
    epsilon: float,
    *,
    rng: Generator | None = None,
) -> float:
    """Add Laplace(0, sensitivity / epsilon) noise to a scalar metric."""

    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if sensitivity < 0:
        raise ValueError("sensitivity must be >= 0")
    generator = rng or np.random.default_rng()
    scale = sensitivity / epsilon
    return float(value + generator.laplace(0.0, scale))


def add_laplace_noise(value: float, sensitivity: float, epsilon: float) -> float:
    """Backward-compatible alias for ``add_dp_noise``."""

    return add_dp_noise(value, sensitivity, epsilon)


def privatize_metrics(
    metrics: dict[str, float],
    sensitivities: dict[str, float],
    epsilon: float = 1.0,
    *,
    rng: Generator | None = None,
) -> dict[str, float]:
    """Apply Laplace noise to each metric with its per-metric sensitivity."""

    generator = rng or np.random.default_rng()
    return {
        key: add_dp_noise(value, sensitivities.get(key, 1.0), epsilon, rng=generator)
        for key, value in metrics.items()
    }

