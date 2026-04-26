"""Log-mel extraction, deployment variant.

Pure NumPy implementation of the same algorithm as ``LogMelExtractor``. This
module intentionally does not import PyTorch.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from solospeak.utils.config import AudioConfig
from solospeak.utils.types import FloatArray


def _build_mel_filterbank_numpy(config: AudioConfig) -> FloatArray:
    """HTK mel filterbank, shape ``(n_fft // 2 + 1, n_mels)``."""
    n_freqs = config.n_fft // 2 + 1
    freq_bins = np.linspace(0.0, config.sample_rate / 2.0, n_freqs, dtype=np.float32)
    m_min = np.float32(2595.0) * np.log10(np.float32(1.0 + config.fmin / 700.0))
    m_max = np.float32(2595.0) * np.log10(np.float32(1.0 + config.fmax / 700.0))
    mel_pts = np.linspace(float(m_min), float(m_max), config.n_mels + 2, dtype=np.float32)
    hz_pts = np.float32(700.0) * (
        np.float32(10.0) ** (mel_pts / np.float32(2595.0)) - np.float32(1.0)
    )
    f = freq_bins[:, None]
    lo = hz_pts[:-2][None, :]
    center = hz_pts[1:-1][None, :]
    hi = hz_pts[2:][None, :]
    lower = (f - lo) / np.maximum(center - lo, np.float32(1e-10))
    upper = (hi - f) / np.maximum(hi - center, np.float32(1e-10))
    return cast(FloatArray, np.maximum(np.minimum(lower, upper), np.float32(0.0)).astype(np.float32))


class LogMelExtractorDeploy:
    """Deployment log-mel extractor, pure NumPy."""

    def __init__(self, config: AudioConfig) -> None:
        if config.win_length != config.n_fft:
            raise ValueError("Deployment extractor requires win_length == n_fft for parity.")
        self.config = config
        self.fb = _build_mel_filterbank_numpy(config)
        self._fb = self.fb

    def _fix_time_dim(self, log_mel: FloatArray) -> FloatArray:
        target = self.config.window_frames
        frames = log_mel.shape[-1]
        if frames == target:
            return log_mel
        if frames > target:
            start = (frames - target) // 2
            return log_mel[..., start : start + target]
        return np.pad(log_mel, ((0, 0), (0, target - frames)), mode="constant").astype(np.float32)

    def __call__(self, waveform: FloatArray) -> FloatArray:
        """Convert ``(T,)`` float32 waveform to ``(1, n_mels, window_frames)``."""
        c = self.config
        x = waveform.astype(np.float32, copy=False)
        if x.ndim != 1:
            raise ValueError(f"Expected mono waveform shape (T,), got {x.shape}.")

        x = np.pad(x, pad_width=c.n_fft // 2, mode="reflect")
        n = np.arange(c.win_length, dtype=np.float32)
        window = np.float32(0.5) * (
            np.float32(1.0) - np.cos(np.float32(2.0 * np.pi / c.win_length) * n)
        )

        n_frames = 1 + (len(x) - c.n_fft) // c.hop_length
        starts = np.arange(n_frames, dtype=np.int64)[:, None] * c.hop_length
        idx = starts + np.arange(c.n_fft, dtype=np.int64)[None, :]
        frames = x[idx].astype(np.float32)
        frames *= window[None, :]

        spectrum = np.fft.rfft(frames, n=c.n_fft, axis=-1)
        power = (spectrum.real**2 + spectrum.imag**2).astype(np.float32)
        mel = (power @ self.fb).astype(np.float32)
        log_mel = np.log(np.maximum(mel, np.float32(1e-6))).astype(np.float32).T
        return self._fix_time_dim(log_mel)[None, :, :]
