"""On-the-fly audio augmentation pipeline.

Each waveform operator: __call__(wav: Tensor) → Tensor, applied probabilistically.
SpecAugment operates on the log-mel spectrogram instead.
CurriculumAugmenter controls SNR/distance schedules during training.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import cast

import numpy as np
import torch


class AddNoise:
    """Mix MUSAN noise at a random SNR."""

    def __init__(
        self,
        noise_dir: Path,
        snr_range_db: tuple[int, int] = (-5, 30),
        prob: float = 0.7,
    ) -> None:
        self.noise_dir = Path(noise_dir)
        self.snr_range_db = snr_range_db
        self.prob = prob
        self._files: list[Path] = []
        if self.noise_dir.exists():
            self._files = list(self.noise_dir.rglob("*.wav"))

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if random.random() > self.prob or not self._files:
            return wav

        import soundfile as sf

        snr_db = random.uniform(self.snr_range_db[0], self.snr_range_db[1])
        noise_np, _ = sf.read(str(random.choice(self._files)), dtype="float32", always_2d=False)
        if noise_np.ndim > 1:
            noise_np = noise_np.mean(axis=1)
        noise = torch.from_numpy(noise_np)

        if len(noise) < len(wav):
            reps = len(wav) // len(noise) + 1
            noise = noise.repeat(reps)
        noise = noise[: len(wav)]

        wav_rms = wav.pow(2).mean().sqrt().clamp(min=1e-8)
        noise_rms = noise.pow(2).mean().sqrt().clamp(min=1e-8)
        snr_lin = 10.0 ** (snr_db / 20.0)
        noise = noise * (wav_rms / (noise_rms * snr_lin))
        return (wav + noise).clamp(-1.0, 1.0)


class ConvolveRIR:
    """Convolve with a random room impulse response."""

    def __init__(self, rir_dir: Path, prob: float = 0.5) -> None:
        self.rir_dir = Path(rir_dir)
        self.prob = prob
        self._files: list[Path] = []
        if self.rir_dir.exists():
            self._files = list(self.rir_dir.rglob("*.wav"))

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if random.random() > self.prob or not self._files:
            return wav

        import soundfile as sf
        from scipy.signal import fftconvolve

        rir_np, _ = sf.read(str(random.choice(self._files)), dtype="float32", always_2d=False)
        if rir_np.ndim > 1:
            rir_np = rir_np.mean(axis=1)

        wav_np = wav.numpy()
        convolved = fftconvolve(wav_np, rir_np)[: len(wav_np)]

        # Normalize to original RMS to avoid clipping
        orig_rms = float(np.sqrt(np.mean(wav_np**2))) + 1e-8
        conv_rms = float(np.sqrt(np.mean(convolved**2))) + 1e-8
        convolved = convolved * (orig_rms / conv_rms)
        return torch.from_numpy(convolved.astype(np.float32))


class GainJitter:
    """Apply random linear gain ±6 dB."""

    def __init__(
        self,
        gain_range_db: tuple[float, float] = (-6.0, 6.0),
        prob: float = 0.3,
    ) -> None:
        self.gain_range_db = gain_range_db
        self.prob = prob

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if random.random() > self.prob:
            return wav
        gain_db = random.uniform(self.gain_range_db[0], self.gain_range_db[1])
        gain = 10.0 ** (gain_db / 20.0)
        return cast(torch.Tensor, (wav * gain).clamp(-1.0, 1.0))


class TimeShift:
    """Circular shift within ±100 ms."""

    def __init__(
        self,
        shift_range_ms: tuple[int, int] = (-100, 100),
        sr: int = 16000,
        prob: float = 0.3,
    ) -> None:
        self.shift_range_ms = shift_range_ms
        self.sr = sr
        self.prob = prob

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if random.random() > self.prob:
            return wav
        shift_ms = random.uniform(self.shift_range_ms[0], self.shift_range_ms[1])
        shift_samples = int(shift_ms * self.sr / 1000.0)
        return torch.roll(wav, shift_samples)


class PitchShift:
    """Small pitch shift (±1 semitone) to simulate speaker variation."""

    def __init__(
        self,
        semitone_range: tuple[float, float] = (-1.0, 1.0),
        sr: int = 16000,
        prob: float = 0.1,
    ) -> None:
        self.semitone_range = semitone_range
        self.sr = sr
        self.prob = prob

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if random.random() > self.prob:
            return wav
        import librosa

        semitones = random.uniform(self.semitone_range[0], self.semitone_range[1])
        shifted = librosa.effects.pitch_shift(
            wav.numpy().astype(np.float32), sr=self.sr, n_steps=semitones
        )
        return torch.from_numpy(shifted[: len(wav)])


class SpecAugment:
    """Time and frequency masking on the log-mel spectrogram."""

    def __init__(
        self,
        n_freq_masks: int = 2,
        freq_mask_width: int = 15,
        n_time_masks: int = 2,
        time_mask_width: int = 20,
        prob: float = 0.5,
    ) -> None:
        self.n_freq_masks = n_freq_masks
        self.freq_mask_width = freq_mask_width
        self.n_time_masks = n_time_masks
        self.time_mask_width = time_mask_width
        self.prob = prob

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """(..., F, T) → (..., F, T)"""
        if random.random() > self.prob:
            return mel
        result = mel.clone()
        n_freqs = mel.shape[-2]
        n_frames = mel.shape[-1]
        for _ in range(self.n_freq_masks):
            f = random.randint(0, min(self.freq_mask_width, n_freqs - 1))
            if f > 0:
                f0 = random.randint(0, n_freqs - f)
                result[..., f0 : f0 + f, :] = 0.0
        for _ in range(self.n_time_masks):
            t = random.randint(0, min(self.time_mask_width, n_frames - 1))
            if t > 0:
                t0 = random.randint(0, n_frames - t)
                result[..., t0 : t0 + t] = 0.0
        return result


class CurriculumAugmenter:
    """Progressively widens noise/distance range during training.

    Starts with mild augmentation, reaches full range at 50% of total steps.
    """

    def __init__(
        self,
        total_steps: int,
        snr_final: tuple[int, int] = (-5, 30),
        distance_final: tuple[float, float] = (0.5, 5.0),
    ) -> None:
        self.total_steps = total_steps
        self.snr_final = snr_final
        self.distance_final = distance_final

    def current_ranges(self, step: int) -> dict[str, tuple[int, int] | tuple[float, float]]:
        """Return the current SNR and distance ranges for a given training step."""
        progress = min(1.0, step / (0.5 * self.total_steps))
        snr_min = int(30 - progress * (30 - self.snr_final[0]))
        dist_max = 0.5 + progress * (self.distance_final[1] - 0.5)
        return {
            "snr_range_db": (snr_min, self.snr_final[1]),
            "distance_range_m": (0.5, dist_max),
        }
