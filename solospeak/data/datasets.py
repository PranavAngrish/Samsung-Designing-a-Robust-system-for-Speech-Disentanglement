"""PyTorch Dataset classes and collators for SoloSpeak training stages."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.data import Dataset

from solospeak.utils.audio import load_audio_segment, pad_or_crop_to_window
from solospeak.utils.config import AudioConfig, DataConfig
from solospeak.utils.types import (
    BatchDict,
    BatchValue,
    DatasetItem,
    DatasetValue,
    FloatArray,
    WORD_IGNORE_INDEX,
)

MANIFEST_COLUMNS = [
    "file_path",
    "start_s",
    "end_s",
    "duration_s",
    "speaker_id",
    "keyword_text",
    "split",
    "source_dataset",
    "quadrant_class",
    "profile_id",
    "enrolled_user_id",
    "enrolled_keyword_text",
    "profile_path",
    "trial_source",
    "q3_gate_eligible",
    "synthesis_backend",
    "speaker_verification_score",
]

_LMDB_PREFIX = "lmdb://"


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [{col: (row.get(col) or "") for col in MANIFEST_COLUMNS} for row in reader]
    return rows


def _load_vocab(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return {str(value): idx for idx, value in enumerate(data)}
    if isinstance(data, dict):
        return {str(key): int(value) for key, value in data.items()}
    raise ValueError(f"Unsupported vocab format in {path}")


def _as_float(value: str, default: float = 0.0) -> float:
    if value == "":
        return default
    return float(value)


def _parse_lmdb_uri(uri: str) -> tuple[Path, str]:
    body = uri.removeprefix(_LMDB_PREFIX)
    lmdb_path, key = body.rsplit("/", 1)
    return Path(lmdb_path), key


def _load_lmdb_audio(uri: str) -> FloatArray:
    import lmdb

    lmdb_path, key = _parse_lmdb_uri(uri)
    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=2048,
    )
    try:
        with env.begin(buffers=True) as txn:
            raw = txn.get(key.encode())
            if raw is None:
                raise KeyError(f"LMDB key not found: {uri}")
            return np.asarray(np.load(io.BytesIO(bytes(raw)), allow_pickle=False), dtype=np.float32)
    finally:
        env.close()


def _load_row_audio(row: dict[str, str], config: AudioConfig) -> torch.Tensor:
    file_path = row["file_path"]
    if file_path.startswith(_LMDB_PREFIX):
        wav = _load_lmdb_audio(file_path)
    else:
        duration = _as_float(row["duration_s"], 0.0)
        start_s = _as_float(row["start_s"], 0.0)
        end_s = _as_float(row["end_s"], duration)
        wav = load_audio_segment(file_path, start_s, end_s, config.sample_rate)
    fixed = pad_or_crop_to_window(wav, config.window_samples)
    fixed = np.clip(fixed, -1.0, 1.0).astype(np.float32)
    return torch.from_numpy(fixed)


def parse_bool_csv(value: DatasetValue) -> bool:
    """Parse canonical manifest booleans. Never replace this with ``bool(value)``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"Expected manifest boolean 'true'/'false', got {value!r}")


class GSCDataset(Dataset[DatasetItem]):
    """Google Speech Commands v2, 35-class classification for Stage 1."""

    def __init__(self, manifest_path: Path, config: AudioConfig) -> None:
        self.manifest_path = Path(manifest_path)
        self.config = config
        self.rows = _read_manifest(self.manifest_path)
        self.gsc_vocab = _load_vocab(self.manifest_path.parent / "gsc_vocab.json")
        self.labels = [self.gsc_vocab[row["keyword_text"]] for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> DatasetItem:
        row = self.rows[idx]
        label = self.gsc_vocab[row["keyword_text"]]
        return {
            "wav": _load_row_audio(row, self.config),
            "keyword_label": label,
            "label": torch.tensor(label, dtype=torch.long),
            "keyword_text": row["keyword_text"],
        }


class DualHeadDataset(Dataset[DatasetItem]):
    """LibriPhrase/VoxCeleb/Common Voice rows for Stages 2-4."""

    def __init__(
        self,
        manifest_path: Path,
        config: AudioConfig,
        data_config: DataConfig,
        wav_augmenters: list[Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.config = config
        self.data_config = data_config
        self.wav_augmenters = wav_augmenters or []
        self.rows = _read_manifest(self.manifest_path)
        self.keyword_vocab = _load_vocab(data_config.manifests_dir / "keyword_vocab.json")
        self.speaker_vocab = _load_vocab(data_config.manifests_dir / "speaker_vocab.json")
        self.keyword_labels = [self._keyword_label(row["keyword_text"]) for row in self.rows]
        self.speaker_labels = [self._speaker_label(row["speaker_id"]) for row in self.rows]
        self.labels = self.keyword_labels

    def __len__(self) -> int:
        return len(self.rows)

    def _keyword_label(self, keyword_text: str) -> int:
        if keyword_text == "" or keyword_text not in self.keyword_vocab:
            return WORD_IGNORE_INDEX
        return self.keyword_vocab[keyword_text]

    def _speaker_label(self, speaker_id: str) -> int:
        try:
            return self.speaker_vocab[speaker_id]
        except KeyError as exc:
            raise KeyError(f"speaker_id {speaker_id!r} is absent from speaker_vocab.json") from exc

    def __getitem__(self, idx: int) -> DatasetItem:
        row = self.rows[idx]
        wav = _load_row_audio(row, self.config)
        for aug in self.wav_augmenters:
            wav = aug(wav)
        return {
            "wav": wav.clamp(-1.0, 1.0).float(),
            "keyword_label": self.keyword_labels[idx],
            "speaker_label": self.speaker_labels[idx],
            "speaker_id_raw": row["speaker_id"],
            "keyword_text": row["keyword_text"],
        }


class QuadrantDataset(Dataset[DatasetItem]):
    """Stage-5/Q4 quadrant rows with enrollment profile templates."""

    def __init__(self, manifest_path: Path, config: AudioConfig) -> None:
        self.manifest_path = Path(manifest_path)
        self.config = config
        self.rows = _read_manifest(self.manifest_path)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_profile(self, profile_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        if profile_path == "":
            raise FileNotFoundError("QuadrantDataset row has blank profile_path")
        path = Path(profile_path)
        if not path.exists():
            raise FileNotFoundError(f"Profile file not found: {path}")
        with np.load(path, allow_pickle=False) as profile:
            content = torch.from_numpy(profile["content_template"].astype(np.float32))
            speaker = torch.from_numpy(profile["speaker_template"].astype(np.float32))
        return content, speaker

    def __getitem__(self, idx: int) -> DatasetItem:
        row = self.rows[idx]
        content_template, speaker_template = self._load_profile(row["profile_path"])
        quadrant_class = row["quadrant_class"]
        score = row["speaker_verification_score"]
        return {
            "wav": _load_row_audio(row, self.config),
            "quadrant_label": 1 if quadrant_class == "Q1_accept" else 0,
            "quadrant_class": quadrant_class,
            "user_id": row["enrolled_user_id"],
            "keyword_text": row["keyword_text"],
            "enrolled_keyword_text": row["enrolled_keyword_text"],
            "profile_id": row["profile_id"],
            "trial_source": row["trial_source"],
            "q3_gate_eligible": parse_bool_csv(row["q3_gate_eligible"]),
            "synthesis_backend": row["synthesis_backend"],
            "speaker_verification_score": None if score == "" else float(score),
            "content_template": content_template,
            "speaker_template": speaker_template,
        }


def _stack_tensor_field(batch: list[DatasetItem], key: str) -> torch.Tensor:
    return torch.stack([cast(torch.Tensor, b[key]) for b in batch])


def collate_gsc(batch: list[DatasetItem]) -> BatchDict:
    """Collate ``GSCDataset`` rows."""
    return {
        "wav": _stack_tensor_field(batch, "wav"),
        "keyword_label": torch.tensor([int(cast(int, b["keyword_label"])) for b in batch]),
    }


def collate_dual_head(batch: list[DatasetItem]) -> BatchDict:
    """Collate ``DualHeadDataset`` rows."""
    return {
        "wav": _stack_tensor_field(batch, "wav"),
        "keyword_label": torch.tensor([int(cast(int, b["keyword_label"])) for b in batch]),
        "speaker_label": torch.tensor([int(cast(int, b["speaker_label"])) for b in batch]),
    }


def collate_quadrant(batch: list[DatasetItem]) -> BatchDict:
    """Collate ``QuadrantDataset`` rows."""
    score_values = [
        float("nan")
        if b["speaker_verification_score"] is None
        else float(cast(float, b["speaker_verification_score"]))
        for b in batch
    ]
    return {
        "wav": _stack_tensor_field(batch, "wav"),
        "quadrant_label": torch.tensor([int(cast(int, b["quadrant_label"])) for b in batch]),
        "quadrant_class": [str(b["quadrant_class"]) for b in batch],
        "user_id": [str(b["user_id"]) for b in batch],
        "keyword_text": [str(b["keyword_text"]) for b in batch],
        "enrolled_keyword_text": [str(b["enrolled_keyword_text"]) for b in batch],
        "profile_id": [str(b["profile_id"]) for b in batch],
        "trial_source": [str(b["trial_source"]) for b in batch],
        "q3_gate_eligible": torch.tensor(
            [parse_bool_csv(b["q3_gate_eligible"]) for b in batch], dtype=torch.bool
        ),
        "synthesis_backend": [str(b["synthesis_backend"]) for b in batch],
        "speaker_verification_score": torch.tensor(score_values, dtype=torch.float32),
        "content_template": _stack_tensor_field(batch, "content_template"),
        "speaker_template": _stack_tensor_field(batch, "speaker_template"),
    }


collate_fixed_length = collate_dual_head


def collate_variable_length(batch: list[DatasetItem]) -> dict[str, BatchValue]:
    """Legacy collator retained for tests and old exploratory notebooks."""
    wavs = [cast(torch.Tensor, item["wav"]) for item in batch]
    lengths = torch.tensor([w.shape[-1] for w in wavs], dtype=torch.long)
    max_len = int(lengths.max().item())

    padded = torch.zeros(len(batch), max_len, dtype=wavs[0].dtype)
    for i, wav in enumerate(wavs):
        padded[i, : wav.shape[-1]] = wav

    result: dict[str, BatchValue] = {"wav": padded, "lengths": lengths}
    for key, value in batch[0].items():
        if key == "wav":
            continue
        if isinstance(value, torch.Tensor):
            result[key] = torch.stack([cast(torch.Tensor, item[key]) for item in batch])
        elif isinstance(value, int):
            result[key] = torch.tensor([int(cast(int, item[key])) for item in batch])
        elif isinstance(value, float):
            result[key] = torch.tensor([float(cast(float, item[key])) for item in batch])
        elif isinstance(value, str):
            result[key] = [str(item[key]) for item in batch]
        elif isinstance(value, bool):
            result[key] = torch.tensor([bool(item[key]) for item in batch])
    return result
