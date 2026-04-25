"""PyTorch Dataset classes for each training stage.

Stage 1: GSCDataset          — 35-class KWS classification (Google Speech Commands v2)
Stage 2+: DualHeadDataset    — (wav, keyword_label, speaker_label) triplets
Stage 4: RobustnessDataset   — DualHeadDataset with heavy augmentation
Stage 5: QuadrantDataset     — (wav, quadrant_label) for fusion training

All datasets:
  - Read from a CSV manifest (file_path, speaker_id, keyword_text, quadrant_class, ...)
  - Load audio lazily in __getitem__ via solospeak.utils.audio.load_audio
  - Return {'wav': Tensor(T,), ...} dicts — use collate_variable_length for batching
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset

from solospeak.utils.config import AudioConfig, DataConfig

# pandas imported lazily inside methods — avoids NumPy ABI issues at import time


class GSCDataset(Dataset[dict[str, torch.Tensor]]):
    """Google Speech Commands v2 — 35-class classification for Stage 1."""

    def __init__(self, manifest_path: Path, config: AudioConfig) -> None:
        import pandas as pd

        self.config = config
        self.df = pd.read_csv(manifest_path)
        words = sorted(self.df["keyword_text"].unique())
        self.word_to_idx: dict[str, int] = {w: i for i, w in enumerate(words)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        from solospeak.utils.audio import load_audio

        row = self.df.iloc[idx]
        wav = load_audio(Path(str(row["file_path"])), self.config.sample_rate)
        label = self.word_to_idx[str(row["keyword_text"])]
        return {
            "wav": torch.from_numpy(wav),
            "label": torch.tensor(label, dtype=torch.long),
        }


class DualHeadDataset(Dataset[dict[str, torch.Tensor]]):
    """LibriPhrase + VoxCeleb — (wav, keyword_label, speaker_label) for Stages 2–4."""

    def __init__(
        self,
        manifest_path: Path,
        config: AudioConfig,
        data_config: DataConfig,
        wav_augmenters: Optional[list[Callable[[torch.Tensor], torch.Tensor]]] = None,
    ) -> None:
        import pandas as pd

        self.config = config
        self.data_config = data_config
        # Waveform-domain augmenters only. SpecAugment operates on the mel
        # spectrogram and must be applied in the training loop after feature
        # extraction — do NOT pass it here.
        self.wav_augmenters = wav_augmenters or []
        self.df = pd.read_csv(manifest_path)

        keywords = sorted(self.df["keyword_text"].unique())
        speakers = sorted(self.df["speaker_id"].astype(str).unique())
        self.word_to_idx: dict[str, int] = {w: i for i, w in enumerate(keywords)}
        self.spk_to_idx: dict[str, int] = {s: i for i, s in enumerate(speakers)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        from solospeak.utils.audio import load_audio

        row = self.df.iloc[idx]
        wav = load_audio(Path(str(row["file_path"])), self.config.sample_rate)
        wav_t = torch.from_numpy(wav)

        for aug in self.wav_augmenters:
            wav_t = aug(wav_t)

        keyword_text = str(row["keyword_text"])
        speaker_id = str(row["speaker_id"])
        if keyword_text not in self.word_to_idx:
            raise KeyError(
                f"keyword '{keyword_text}' at row {idx} not in training vocab. "
                "Ensure dev/test manifests only contain labels seen during vocab construction."
            )
        if speaker_id not in self.spk_to_idx:
            raise KeyError(
                f"speaker '{speaker_id}' at row {idx} not in training speaker set. "
                "Build separate vocab dicts per split or use an open-vocabulary label scheme."
            )
        return {
            "wav": wav_t,
            "keyword_label": torch.tensor(self.word_to_idx[keyword_text], dtype=torch.long),
            "speaker_label": torch.tensor(self.spk_to_idx[speaker_id], dtype=torch.long),
        }


class QuadrantDataset(Dataset[dict[str, torch.Tensor]]):
    """4-quadrant dataset for Stage 5 fusion training.

    quadrant_class column: Q1=wake, Q2=wrong_speaker, Q3=wrong_word, Q4=background.
    """

    _QUADRANT_MAP = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}

    def __init__(self, manifest_path: Path, config: AudioConfig) -> None:
        import pandas as pd

        self.config = config
        self.df = pd.read_csv(manifest_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        from solospeak.utils.audio import load_audio

        row = self.df.iloc[idx]
        wav = load_audio(Path(str(row["file_path"])), self.config.sample_rate)
        quadrant = self._QUADRANT_MAP.get(str(row.get("quadrant_class", "Q4")), 3)
        return {
            "wav": torch.from_numpy(wav),
            "quadrant_label": torch.tensor(quadrant, dtype=torch.long),
        }


def collate_variable_length(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-length audio to longest in batch.

    Pad value: 0.0 (silence).
    Returns dict with 'wav' (B, T_max), 'lengths' (B,), and all other keys stacked.
    """
    wavs = [item["wav"] for item in batch]
    lengths = torch.tensor([w.shape[-1] for w in wavs], dtype=torch.long)
    max_len = int(lengths.max().item())

    padded = torch.zeros(len(batch), max_len, dtype=wavs[0].dtype)
    for i, w in enumerate(wavs):
        padded[i, : w.shape[-1]] = w

    result: dict[str, torch.Tensor] = {"wav": padded, "lengths": lengths}
    for key in batch[0]:
        if key != "wav":
            result[key] = torch.stack([item[key] for item in batch])
    return result
