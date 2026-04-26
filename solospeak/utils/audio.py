"""PCM/WAV/FLAC audio helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from solospeak.utils.types import FloatArray


def load_audio(path: Path, target_sr: int = 16000) -> FloatArray:
    """Load audio file, resample if needed, return float32 mono array in [-1, 1]."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return np.asarray(audio, dtype=np.float32)


def save_audio(path: Path, audio: FloatArray, sr: int = 16000) -> None:
    """Save float32 mono array as WAV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


def trim_silence(
    audio: FloatArray,
    sr: int = 16000,
    top_db: float = 30.0,
) -> FloatArray:
    """VAD-trim leading/trailing silence using energy threshold."""
    import librosa
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return np.asarray(trimmed, dtype=np.float32)


def rms_normalize(audio: FloatArray, target_rms: float = 0.1) -> FloatArray:
    """Normalize audio to target RMS level."""
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-8:
        return audio
    return np.asarray(audio * (target_rms / rms), dtype=audio.dtype)
