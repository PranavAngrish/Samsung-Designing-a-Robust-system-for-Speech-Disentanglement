"""Shared pytest fixtures used across unit and integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from solospeak.utils.config import (
    AudioConfig,
    BackboneConfig,
    SoloSpeakConfig,
    TrainingConfig,
)

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
AUDIO_DIR = FIXTURES_DIR / "audio"


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def audio_config() -> AudioConfig:
    return AudioConfig()


@pytest.fixture()
def default_config() -> SoloSpeakConfig:
    return SoloSpeakConfig(
        run_name="test-run",
        backbone=BackboneConfig(variant="bcresnet1"),  # smallest for fast tests
        training=TrainingConfig(stage=1, batch_size=4, num_epochs=1, seed=42),
    )


# ---------------------------------------------------------------------------
# Audio fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def random_waveform() -> torch.Tensor:
    """1-second random waveform at 16 kHz, shape (1, 16000)."""
    torch.manual_seed(0)
    return torch.randn(1, 16000) * 0.1


@pytest.fixture()
def random_mel(audio_config: AudioConfig) -> torch.Tensor:
    """Random log-mel spectrogram, shape (1, 1, 80, 100)."""
    torch.manual_seed(0)
    return torch.randn(1, 1, audio_config.n_mels, 100)


@pytest.fixture()
def random_batch_mel(audio_config: AudioConfig) -> torch.Tensor:
    """Batch of 4 log-mel spectrograms, shape (4, 1, 80, 100)."""
    torch.manual_seed(0)
    return torch.randn(4, 1, audio_config.n_mels, 100)


# ---------------------------------------------------------------------------
# Embedding fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def random_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    """Pair of L2-normalized (4, 128) embedding tensors."""
    torch.manual_seed(0)
    z_c = torch.randn(4, 128)
    z_s = torch.randn(4, 128)
    import torch.nn.functional as F
    return F.normalize(z_c, p=2, dim=-1), F.normalize(z_s, p=2, dim=-1)


@pytest.fixture()
def random_labels() -> torch.Tensor:
    """4 random class labels in [0, 9]."""
    return torch.tensor([0, 1, 0, 2])
