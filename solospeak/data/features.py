"""Log-mel extraction — training variant (pure PyTorch) and deployment variant (NumPy).

Both variants use identical algorithms:
  - Periodic Hann window (torch.hann_window default)
  - Reflect center-padding (center=True, pad_mode='reflect')
  - HTK mel scale, no filterbank normalization

Numerical parity between the two is verified in tests/unit/test_features.py
with atol=3e-3 on a fixed reference waveform.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from solospeak.data.features_deploy import LogMelExtractorDeploy
from solospeak.utils.config import AudioConfig

__all__ = ["LogMelExtractor", "LogMelExtractorDeploy"]


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
