"""Unit tests for augmentation operators."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from solospeak.data.augmentation import (
    AddNoise,
    ConvolveRIR,
    CurriculumAugmenter,
    GainJitter,
    PitchShift,
    SpecAugment,
    TimeShift,
)


# ---------------------------------------------------------------------------
# CurriculumAugmenter (unchanged from Phase 0)
# ---------------------------------------------------------------------------

def test_curriculum_starts_mild() -> None:
    aug = CurriculumAugmenter(total_steps=10_000, snr_final=(-5, 30))
    ranges = aug.current_ranges(0)
    assert ranges["snr_range_db"][0] == 30   # starts at clean (SNR min = 30)


def test_curriculum_reaches_full_at_halfway() -> None:
    aug = CurriculumAugmenter(total_steps=10_000, snr_final=(-5, 30))
    ranges = aug.current_ranges(5_000)
    assert ranges["snr_range_db"][0] == -5   # full range at 50% of steps


def test_curriculum_clamps_beyond_halfway() -> None:
    aug = CurriculumAugmenter(total_steps=10_000, snr_final=(-5, 30))
    ranges = aug.current_ranges(99_999)
    assert ranges["snr_range_db"][0] == -5


def test_curriculum_distance_scales() -> None:
    aug = CurriculumAugmenter(total_steps=10_000, distance_final=(0.5, 5.0))
    r0 = aug.current_ranges(0)
    r_mid = aug.current_ranges(5_000)
    assert r0["distance_range_m"][1] == pytest.approx(0.5, abs=0.1)
    assert r_mid["distance_range_m"][1] == pytest.approx(5.0, abs=0.1)


# ---------------------------------------------------------------------------
# GainJitter
# ---------------------------------------------------------------------------

def test_gain_jitter_increases_amplitude() -> None:
    aug = GainJitter(gain_range_db=(3.0, 3.0), prob=1.0)  # fixed +3 dB
    wav = torch.ones(1000) * 0.1
    out = aug(wav)
    # +3 dB = ×1.41 — should be clearly louder
    assert out.abs().mean().item() > wav.abs().mean().item() * 1.3


def test_gain_jitter_preserves_shape() -> None:
    aug = GainJitter(prob=1.0)
    wav = torch.randn(16000)
    assert aug(wav).shape == wav.shape


def test_gain_jitter_skipped_when_prob_zero() -> None:
    aug = GainJitter(gain_range_db=(20.0, 20.0), prob=0.0)  # never apply
    wav = torch.ones(100) * 0.1
    out = aug(wav)
    assert torch.allclose(out, wav)


# ---------------------------------------------------------------------------
# TimeShift
# ---------------------------------------------------------------------------

def test_time_shift_preserves_shape() -> None:
    aug = TimeShift(prob=1.0)
    wav = torch.randn(16000)
    assert aug(wav).shape == wav.shape


def test_time_shift_is_circular() -> None:
    """torch.roll wraps samples — sum must be preserved."""
    aug = TimeShift(shift_range_ms=(50, 50), prob=1.0)  # fixed +50 ms
    wav = torch.arange(16000, dtype=torch.float32)
    out = aug(wav)
    assert torch.allclose(out.sum(), wav.sum())


# ---------------------------------------------------------------------------
# SpecAugment
# ---------------------------------------------------------------------------

def test_spec_augment_preserves_shape() -> None:
    aug = SpecAugment(prob=1.0)
    mel = torch.randn(1, 80, 100)
    assert aug(mel).shape == mel.shape


def test_spec_augment_preserves_shape_batched() -> None:
    aug = SpecAugment(prob=1.0)
    mel = torch.randn(4, 1, 80, 100)
    assert aug(mel).shape == mel.shape


def test_spec_augment_zeroes_values() -> None:
    """With large masks, some values must be zeroed."""
    random.seed(42)
    aug = SpecAugment(n_freq_masks=2, freq_mask_width=30,
                      n_time_masks=2, time_mask_width=30, prob=1.0)
    mel = torch.ones(1, 80, 100)
    out = aug(mel)
    assert (out == 0.0).any(), "SpecAugment should zero at least one value"


def test_spec_augment_skipped_when_prob_zero() -> None:
    aug = SpecAugment(prob=0.0)
    mel = torch.ones(1, 80, 100)
    out = aug(mel)
    assert torch.allclose(out, mel)


# ---------------------------------------------------------------------------
# AddNoise — graceful with missing noise dir
# ---------------------------------------------------------------------------

def test_add_noise_passthrough_without_files() -> None:
    aug = AddNoise(noise_dir=Path("/nonexistent_dir_xyz"), prob=1.0)
    wav = torch.randn(16000)
    out = aug(wav)
    assert out.shape == wav.shape
    assert torch.allclose(out, wav)  # unchanged — no noise files found


# ---------------------------------------------------------------------------
# ConvolveRIR — graceful with missing RIR dir
# ---------------------------------------------------------------------------

def test_convolve_rir_passthrough_without_files() -> None:
    aug = ConvolveRIR(rir_dir=Path("/nonexistent_dir_xyz"), prob=1.0)
    wav = torch.randn(16000)
    out = aug(wav)
    assert out.shape == wav.shape
    assert torch.allclose(out, wav)


# ---------------------------------------------------------------------------
# PitchShift — shape contract only (no librosa I/O needed)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_pitch_shift_preserves_shape() -> None:
    aug = PitchShift(prob=1.0)
    wav = torch.randn(16000)
    out = aug(wav)
    assert out.shape == wav.shape
