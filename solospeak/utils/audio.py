"""PCM/WAV/FLAC/MP3 audio helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from solospeak.utils.types import FloatArray


def load_audio(path: Path | str, target_sr: int = 16000) -> FloatArray:
    """Load audio file, resample if needed, return float32 mono array in [-1, 1]."""
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        import librosa

        audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return np.asarray(np.clip(audio, -1.0, 1.0), dtype=np.float32)


def load_wav(path: Path | str, target_sr: int = 16000) -> FloatArray:
    """Compatibility alias for the Phase-0 audio contract."""
    return load_audio(path, target_sr)


# def load_audio_segment(
#     path: Path | str,
#     start_s: float,
#     end_s: float,
#     target_sr: int = 16000,
# ) -> FloatArray:
#     """Load a mono waveform segment after resampling to ``target_sr``."""
#     wav = load_audio(path, target_sr)
#     start = max(0, int(round(start_s * target_sr)))
#     end = int(round(end_s * target_sr)) if end_s > 0 else len(wav)
#     end = min(len(wav), max(start, end))
#     return np.asarray(wav[start:end], dtype=np.float32)

def load_audio_segment(path, start_s, duration_s, target_sr=16000):
    wav = load_audio(path, target_sr)
    start = max(0, int(round(start_s * target_sr)))
    end = min(len(wav), start + int(round(duration_s * target_sr)))
    return wav[start:end].astype(np.float32)


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


def pad_or_crop_to_window(wav: FloatArray, window_samples: int = 25600) -> FloatArray:
    """Pad right or center-crop a waveform to exactly ``window_samples`` samples."""
    if len(wav) == window_samples:
        return np.asarray(wav, dtype=np.float32)
    if len(wav) < window_samples:
        pad = window_samples - len(wav)
        if len(wav) <= 1:
            return np.pad(wav, (0, pad), mode="constant").astype(np.float32)
        return np.pad(wav, (0, pad), mode="reflect").astype(np.float32)
    start = (len(wav) - window_samples) // 2
    return np.asarray(wav[start : start + window_samples], dtype=np.float32)


def make_lmdb_key(file_path: str, start_s: float, end_s: float) -> str:
    """Deterministic LMDB key for a cropped audio clip."""
    raw = f"{file_path}:{start_s:.6f}:{end_s:.6f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]
