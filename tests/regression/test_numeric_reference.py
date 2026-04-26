"""Numeric regression tests — committed reference outputs.

These tests lock in the numeric behaviour of pure-Python components
that don't depend on trained weights.
"""

from __future__ import annotations

import torch

from solospeak.data.splits import assign_split
from solospeak.inference.hysteresis import HysteresisDetector
from solospeak.losses.orthogonality import orthogonality_loss
from solospeak.enrollment.calibration import calibrate_threshold


def test_split_assignment_reference() -> None:
    """Fixed seed → fixed split assignments. Catches accidental hash changes."""
    expected = {
        "spk_0000": "train",
        "spk_0001": "train",
        "spk_0002": "dev",
        "spk_0003": "train",
    }
    for sid, expected_split in expected.items():
        assert assign_split(sid, seed=42) == expected_split, f"Split changed for {sid}"


def test_calibrate_threshold_reference() -> None:
    tau = calibrate_threshold(rejection_score=0.2, high_confidence=0.9)
    assert abs(tau - 0.55) < 1e-6


def test_orthogonality_loss_zero_reference() -> None:
    """Perfectly disjoint subspaces → loss exactly 0."""
    B, D = 8, 64
    z_c = torch.zeros(B, D)
    z_s = torch.zeros(B, D)
    z_c[:, :32] = 1.0
    z_s[:, 32:] = 1.0
    loss = orthogonality_loss(z_c, z_s)
    assert loss.item() < 1e-9


def test_hysteresis_sequence_reference() -> None:
    """Fixed input sequence → fixed output sequence."""
    det = HysteresisDetector(tau_on=0.75, tau_off=0.45, refractory_s=1.5, stride_s=0.1)
    scores = [0.1, 0.3, 0.8, 0.9, 0.8, 0.2, 0.1, 0.0]
    results = [det.step(s) for s in scores]
    # Only frame 2 (score=0.8) should fire; rest in refractory or below threshold
    assert results[2] is True
    assert sum(results) == 1
