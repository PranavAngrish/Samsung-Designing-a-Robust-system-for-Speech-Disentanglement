"""Differential privacy noise for fleet statistics upload.

Local DP budget: epsilon = 1.0 per device per week.
Mechanism: Laplace noise added to each metric before upload.
Aggregation: Samsung cloud averages over >= 1000 devices minimum.
"""

from __future__ import annotations

import numpy as np


def add_laplace_noise(value: float, sensitivity: float, epsilon: float) -> float:
    """Add Laplace noise with given sensitivity and privacy budget epsilon."""
    scale = sensitivity / epsilon
    return float(value + np.random.laplace(0.0, scale))


def privatize_metrics(
    metrics: dict[str, float],
    sensitivities: dict[str, float],
    epsilon: float = 1.0,
) -> dict[str, float]:
    """Apply Laplace noise to each metric with its per-metric sensitivity."""
    return {
        k: add_laplace_noise(v, sensitivities.get(k, 1.0), epsilon)
        for k, v in metrics.items()
    }
