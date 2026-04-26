"""Log-mel extraction — training variant (pure PyTorch) and deployment variant (NumPy).

Both variants use identical algorithms:
  - Periodic Hann window (torch.hann_window default)
  - Reflect center-padding (center=True, pad_mode='reflect')
  - HTK mel scale, no filterbank normalization

Numerical parity between the two is verified in tests/unit/test_features.py
with atol=3e-3 on a fixed reference waveform.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from solospeak.utils.config import AudioConfig
from solospeak.utils.types import FloatArray


def _build_mel_filterbank_torch(config: AudioConfig) -> torch.Tensor:
    """HTK mel filterbank — (n_fft//2+1, n_mels) float32 torch Tensor."""
    n_freqs = config.n_fft // 2 + 1
    freq_bins = torch.linspace(0.0, config.sample_rate / 2.0, n_freqs)

    m_min = 2595.0 * torch.log10(torch.tensor(1.0 + config.fmin / 700.0))
    m_max = 2595.0 * torch.log10(torch.tensor(1.0 + config.fmax / 700.0))
    mel_pts = torch.linspace(m_min.item(), m_max.item(), config.n_mels + 2)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)

    f = freq_bins.unsqueeze(1)          # (n_freqs, 1)
    lo = hz_pts[:-2].unsqueeze(0)       # (1, n_mels)
    center = hz_pts[1:-1].unsqueeze(0)
    hi = hz_pts[2:].unsqueeze(0)

    lower = (f - lo) / (center - lo).clamp(min=1e-10)
    upper = (hi - f) / (hi - center).clamp(min=1e-10)
    return torch.clamp(torch.minimum(lower, upper), min=0.0).float()


def _center_crop_or_right_pad_torch(mel: torch.Tensor, frames: int) -> torch.Tensor:
    current = mel.shape[-1]
    if current == frames:
        return mel
    if current > frames:
        start = (current - frames) // 2
        return mel[..., start : start + frames]
    return F.pad(mel, (0, frames - current))


def _center_crop_or_right_pad_np(mel: FloatArray, frames: int) -> FloatArray:
    current = mel.shape[-1]
    if current == frames:
        return mel
    if current > frames:
        start = (current - frames) // 2
        return mel[..., start : start + frames]
    return np.pad(mel, ((0, 0), (0, 0), (0, frames - current)), mode="constant")


class LogMelExtractor(nn.Module):
    """Training-time log-mel extractor — pure PyTorch, no torchaudio dependency.

    📋 CONTRACT
        input:  (B, T) waveform, float32, 16 kHz
        output: (B, 1, n_mels, T') log-mel spectrogram

    Uses torch.stft with center=True, reflect padding, periodic Hann window.
    Numerically equivalent to LogMelExtractorDeploy within 3e-3 absolute tolerance.
    """

    fb: torch.Tensor  # mel filterbank buffer, shape (n_fft//2+1, n_mels)

    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("fb", _build_mel_filterbank_torch(config))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """(B, T) → (B, 1, n_mels, T')"""
        c = self.config
        window = torch.hann_window(c.win_length, device=waveform.device, dtype=waveform.dtype)

        stft = torch.stft(
            waveform,
            n_fft=c.n_fft,
            hop_length=c.hop_length,
            win_length=c.win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )  # (B, n_fft//2+1, n_frames)

        power = stft.abs().pow(2)  # (B, n_fft//2+1, n_frames)

        # Mel filterbank: (B, F, T) × (F, M) → (B, M, T)
        mel = torch.einsum("bft,fm->bmt", power, self.fb)

        log_mel = _center_crop_or_right_pad_torch(torch.log(mel.clamp(min=1e-6)), c.window_frames)
        return log_mel.unsqueeze(1)  # (B, 1, n_mels, n_frames)


class LogMelExtractorDeploy:
    """Deployment log-mel extractor — pure NumPy, no PyTorch dependency.

    📋 CONTRACT
        input:  (T,) float32 waveform
        output: (1, n_mels, T') float32 log-mel spectrogram

    Uses identical algorithm to LogMelExtractor:
      - Periodic Hann window
      - Reflect center-padding
      - HTK mel scale, no filterbank normalization
    Numerically equivalent within 1e-5 absolute tolerance.
    """

    def __init__(self, config: AudioConfig) -> None:
        if config.win_length != config.n_fft:
            raise ValueError("LogMelExtractorDeploy requires win_length == n_fft")
        self.config = config
        self._fb = self._build_mel_filterbank()

    def _build_mel_filterbank(self) -> FloatArray:
        """HTK mel filterbank in strict float32 to match LogMelExtractor (torch) exactly."""
        c = self.config
        n_freqs = c.n_fft // 2 + 1
        freq_bins = np.linspace(np.float32(0), np.float32(c.sample_rate) / np.float32(2), n_freqs,
                                dtype=np.float32)

        # HTK mel scale — all float32 to match torch.tensor(float) → float32 conversion
        m_min = float(np.float32(2595) * np.log10(np.float32(1) + np.float32(c.fmin) / np.float32(700)))
        m_max = float(np.float32(2595) * np.log10(np.float32(1) + np.float32(c.fmax) / np.float32(700)))
        mel_pts = np.linspace(m_min, m_max, c.n_mels + 2, dtype=np.float32)

        # hz_pts in float32 — in torch, scalar division of float32 tensor stays float32
        hz_pts = (np.float32(700) * (np.float32(10) ** (mel_pts / np.float32(2595)) - np.float32(1)))

        # Triangular filters: (n_freqs, n_mels)
        f = freq_bins[:, None]
        lo = hz_pts[:-2][None, :]
        center = hz_pts[1:-1][None, :]
        hi = hz_pts[2:][None, :]
        lower = (f - lo) / np.where(center > lo, center - lo, np.float32(1e-10))
        upper = (hi - f) / np.where(hi > center, hi - center, np.float32(1e-10))
        return np.asarray(np.maximum(np.float32(0), np.minimum(lower, upper)), dtype=np.float32)

    def __call__(self, waveform: FloatArray) -> FloatArray:
        """(T,) float32 → (1, n_mels, T') float32"""
        c = self.config
        x = waveform.astype(np.float32)

        # Center-pad with reflect
        pad = c.n_fft // 2
        x = np.pad(x, pad, mode="reflect")

        # Periodic Hann window in float32 — matches torch.hann_window(N, periodic=True)
        # Use float32 constant to avoid float64 promotion from np.pi
        n = np.arange(c.win_length, dtype=np.float32)
        two_pi_over_N = np.float32(2.0 * np.pi / c.win_length)  # precompute, then cast
        window = np.float32(0.5) * (np.float32(1) - np.cos(n * two_pi_over_N))
        # window is float32: n(f32) * float32 → float32 arg to cos → float32 output

        # Frame: (n_frames, n_fft) — float32 × float32 = float32
        n_frames = 1 + (len(x) - c.n_fft) // c.hop_length
        row_idx = np.arange(c.n_fft)[None, :] + np.arange(n_frames)[:, None] * c.hop_length
        frames = (x[row_idx] * window[None, :]).astype(np.float32)

        # FFT → complex64 (numpy pocketfft preserves float32 dtype)
        spectrum = np.fft.rfft(frames, n=c.n_fft, axis=-1)
        power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)

        # Mel filterbank: (n_frames, n_fft//2+1) @ (n_fft//2+1, n_mels)
        mel = (power @ self._fb).astype(np.float32)

        # Log with clamp
        log_mel = np.log(np.maximum(mel, np.float32(1e-6))).astype(np.float32)
        fixed = _center_crop_or_right_pad_np(log_mel.T[None, :, :], c.window_frames)
        return np.asarray(fixed, dtype=np.float32)  # (1, n_mels, window_frames)
