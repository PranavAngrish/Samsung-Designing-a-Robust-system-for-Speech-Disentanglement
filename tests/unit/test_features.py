"""Unit tests for log-mel feature extractors."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from solospeak.data.features import LogMelExtractor
from solospeak.data.features_deploy import LogMelExtractorDeploy
from solospeak.utils.config import AudioConfig


@pytest.fixture()
def ref_waveform() -> np.ndarray:
    """Deterministic 1.6-second waveform: mix of 440 Hz + 1 kHz tones."""
    sr = 16000
    t = np.linspace(0, 1.6, int(1.6 * sr), dtype=np.float32)
    return 0.4 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 1000 * t)


def test_train_extractor_output_shape(audio_config: AudioConfig, ref_waveform: np.ndarray) -> None:
    extractor = LogMelExtractor(audio_config)
    wav_t = torch.from_numpy(ref_waveform).unsqueeze(0)  # (1, T)
    with torch.no_grad():
        out = extractor(wav_t)
    assert out.ndim == 4                        # (B, 1, n_mels, T')
    assert out.shape[1] == 1
    assert out.shape[2] == audio_config.n_mels
    assert out.shape[-1] == audio_config.window_frames


def test_deploy_extractor_output_shape(audio_config: AudioConfig, ref_waveform: np.ndarray) -> None:
    extractor = LogMelExtractorDeploy(audio_config)
    out = extractor(ref_waveform)
    assert out.ndim == 3                        # (1, n_mels, T')
    assert out.shape[0] == 1
    assert out.shape[1] == audio_config.n_mels
    assert out.shape[-1] == audio_config.window_frames


def test_train_deploy_same_time_frames(audio_config: AudioConfig, ref_waveform: np.ndarray) -> None:
    """Both extractors must produce the same number of time frames."""
    train_ext = LogMelExtractor(audio_config)
    deploy_ext = LogMelExtractorDeploy(audio_config)

    wav_t = torch.from_numpy(ref_waveform).unsqueeze(0)
    with torch.no_grad():
        train_out = train_ext(wav_t)
    deploy_out = deploy_ext(ref_waveform)

    assert train_out.shape[-1] == deploy_out.shape[-1], (
        f"Frame count mismatch: train={train_out.shape[-1]}, deploy={deploy_out.shape[-1]}"
    )


def test_train_deploy_parity(audio_config: AudioConfig, ref_waveform: np.ndarray) -> None:
    """Training and deploy extractors must agree within 3e-3 absolute tolerance.

    Both use identical algorithms (same window, padding, mel scale, clamp=1e-6).
    Tolerance is 3e-3 rather than 1e-5 because PyTorch's FFT and numpy's pocketfft
    accumulate float32 rounding differently — max empirical divergence is ~2.4e-3.
    """
    train_ext = LogMelExtractor(audio_config)
    deploy_ext = LogMelExtractorDeploy(audio_config)

    wav_t = torch.from_numpy(ref_waveform).unsqueeze(0)
    with torch.no_grad():
        train_out = train_ext(wav_t).squeeze().numpy()  # (n_mels, T')
    deploy_out = deploy_ext(ref_waveform).squeeze()     # (n_mels, T')

    np.testing.assert_allclose(
        train_out, deploy_out, atol=3e-3, rtol=0,
        err_msg="Training and deploy extractors diverge > 3e-3",
    )


def test_deploy_extractor_finite_on_silence(audio_config: AudioConfig) -> None:
    """Deploy extractor must not produce NaN/Inf on a silent waveform."""
    extractor = LogMelExtractorDeploy(audio_config)
    silent = np.zeros(16000, dtype=np.float32)
    out = extractor(silent)
    assert np.isfinite(out).all(), "Deploy extractor produced non-finite values on silence"


def test_train_extractor_finite_on_silence(audio_config: AudioConfig) -> None:
    """Train extractor must not produce NaN/Inf on a silent waveform."""
    extractor = LogMelExtractor(audio_config)
    silent = torch.zeros(1, 16000)
    with torch.no_grad():
        out = extractor(silent)
    assert out.isfinite().all()


def test_train_extractor_batch(audio_config: AudioConfig, ref_waveform: np.ndarray) -> None:
    """Batched input produces correctly shaped output."""
    extractor = LogMelExtractor(audio_config)
    batch = torch.from_numpy(ref_waveform).unsqueeze(0).expand(4, -1)  # (4, T)
    with torch.no_grad():
        out = extractor(batch)
    assert out.shape[0] == 4
    assert out.shape[1] == 1
    assert out.shape[2] == audio_config.n_mels


def test_deploy_mel_filterbank_shape(audio_config: AudioConfig) -> None:
    extractor = LogMelExtractorDeploy(audio_config)
    n_freqs = audio_config.n_fft // 2 + 1
    assert extractor._fb.shape == (n_freqs, audio_config.n_mels)
    assert extractor._fb.dtype == np.float32
    assert (extractor._fb.sum(axis=0) > 0).all()


def test_train_filterbank_buffer_shape(audio_config: AudioConfig) -> None:
    extractor = LogMelExtractor(audio_config)
    n_freqs = audio_config.n_fft // 2 + 1
    assert extractor.fb.shape == (n_freqs, audio_config.n_mels)
    assert (extractor.fb.sum(dim=0) > 0).all()
