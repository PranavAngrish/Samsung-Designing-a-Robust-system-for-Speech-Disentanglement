"""Unit tests for Phase 0 configuration contracts."""

from __future__ import annotations

from pathlib import Path
from typing import get_args

from solospeak.utils.config import (
    AudioConfig,
    SoloSpeakConfig,
    TrainingConfig,
    ValidationGates,
    load_eval_config_bundle,
)
from solospeak.utils.types import BackboneVariant


def test_audio_window_contract() -> None:
    config = AudioConfig()
    assert config.window_duration_s == 1.6
    assert config.window_samples == 25600
    assert config.window_frames == 160


def test_training_config_accepts_injected_class_counts() -> None:
    config = TrainingConfig(stage=2, n_aux_word_classes=128, n_aux_speaker_classes=256)
    assert config.n_aux_word_classes == 128
    assert config.n_aux_speaker_classes == 256


def test_yaml_inheritance_loads_defaults() -> None:
    config = SoloSpeakConfig.from_yaml(Path("configs/training/stage2_dual_head.yaml"))
    assert config.audio.window_duration_s == 1.6
    assert config.training.stage == 2
    assert config.training.lr == 0.001


def test_eval_bundle_uses_hard_gates() -> None:
    eval_config, gates = load_eval_config_bundle(Path("configs/eval/full_kpi_suite.yaml"))
    assert eval_config.fa_audio_hours == 10.0
    assert isinstance(gates, ValidationGates)
    assert gates.min_ta_clean == 0.92
    assert gates.max_xrt == 0.20


def test_backbone_variant_excludes_legacy_backbone() -> None:
    legacy_name = "match" + "boxnet"
    assert legacy_name not in get_args(BackboneVariant)
