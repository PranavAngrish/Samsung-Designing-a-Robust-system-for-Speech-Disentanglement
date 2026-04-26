"""Unit tests for Phase 1 dataset contracts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from scripts.prepare_manifests import GSC_CLASSES, MANIFEST_COLUMNS, write_smoke_manifests
from scripts.pack_audio_lmdb import pack_manifests
from solospeak.data.datasets import (
    DualHeadDataset,
    GSCDataset,
    QuadrantDataset,
    collate_dual_head,
    collate_gsc,
    parse_bool_csv,
)
from solospeak.data.samplers import ClassAwareBatchSampler
from solospeak.utils.config import AudioConfig, DataConfig
from solospeak.utils.types import WORD_IGNORE_INDEX


def _write_wav(path: Path, seconds: float = 1.6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sr = 16000
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    sf.write(str(path), (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _row(path: Path, speaker: str, keyword: str = "") -> dict[str, str]:
    return {
        "file_path": str(path),
        "start_s": "0.000000",
        "end_s": "1.600000",
        "duration_s": "1.600000",
        "speaker_id": speaker,
        "keyword_text": keyword,
        "split": "train",
        "source_dataset": "smoke",
        "quadrant_class": "",
        "profile_id": "",
        "enrolled_user_id": "",
        "enrolled_keyword_text": "",
        "profile_path": "",
        "trial_source": "real",
        "q3_gate_eligible": "true",
        "synthesis_backend": "",
        "speaker_verification_score": "",
    }


def test_parse_bool_csv_rejects_python_truthiness() -> None:
    assert parse_bool_csv("true") is True
    assert parse_bool_csv("false") is False
    with pytest.raises(ValueError):
        parse_bool_csv("yes")


def test_smoke_manifests_preserve_gsc_class_shape(tmp_path: Path) -> None:
    write_smoke_manifests(tmp_path)
    rows = list(csv.DictReader(open(tmp_path / "train_gsc.csv", newline="")))
    stats = json.loads((tmp_path / "STATS.json").read_text())
    vocab = json.loads((tmp_path / "gsc_vocab.json").read_text())
    assert len(rows) == 350
    assert set(row["keyword_text"] for row in rows) == set(GSC_CLASSES)
    assert stats["n_gsc_classes"] == 35
    assert list(vocab) == GSC_CLASSES


def test_gsc_dataset_uses_committed_vocab_and_fixed_window(tmp_path: Path) -> None:
    wav = tmp_path / "yes.wav"
    _write_wav(wav)
    manifest = tmp_path / "train_gsc.csv"
    _write_manifest(manifest, [_row(wav, "gsc_spk", "yes")])
    (tmp_path / "gsc_vocab.json").write_text(json.dumps({"yes": 0}))

    dataset = GSCDataset(manifest, AudioConfig())
    item = dataset[0]
    batch = collate_gsc([item])
    assert item["wav"].shape == (25600,)
    assert batch["keyword_label"].dtype == torch.long
    assert batch["keyword_label"].tolist() == [0]


def test_dual_head_oov_keyword_uses_ignore_index(tmp_path: Path) -> None:
    wav = tmp_path / "speaker.wav"
    _write_wav(wav)
    manifest = tmp_path / "train_speaker.csv"
    _write_manifest(manifest, [_row(wav, "speaker_a", "")])
    (tmp_path / "keyword_vocab.json").write_text(json.dumps({"wake phrase": 0}))
    (tmp_path / "speaker_vocab.json").write_text(json.dumps({"speaker_a": 0}))

    dataset = DualHeadDataset(
        manifest,
        AudioConfig(),
        DataConfig(manifests_dir=tmp_path),
    )
    item = dataset[0]
    batch = collate_dual_head([item])
    assert item["keyword_label"] == WORD_IGNORE_INDEX
    assert batch["speaker_label"].tolist() == [0]


def test_quadrant_dataset_requires_profile_path(tmp_path: Path) -> None:
    wav = tmp_path / "trial.wav"
    _write_wav(wav)
    row = _row(wav, "speaker_a", "wake phrase")
    row.update(
        {
            "quadrant_class": "Q1_accept",
            "profile_id": "profile_a",
            "enrolled_user_id": "user_a",
            "enrolled_keyword_text": "wake phrase",
        }
    )
    manifest = tmp_path / "test_kpi.csv"
    _write_manifest(manifest, [row])
    dataset = QuadrantDataset(manifest, AudioConfig())
    with pytest.raises(FileNotFoundError):
        _ = dataset[0]


def test_class_aware_sampler_has_multiple_samples_per_class() -> None:
    labels = [0, 0, 0, 1, 1, 2]
    sampler = ClassAwareBatchSampler(labels, batch_size=6, num_classes_per_batch=3,
                                     num_samples_per_class=2, seed=7)
    batch = next(iter(sampler))
    seen: dict[int, int] = {}
    for idx in batch:
        seen[labels[idx]] = seen.get(labels[idx], 0) + 1
    assert len(batch) == 6
    assert min(seen.values()) >= 2


def test_pack_lmdb_rewrites_manifest_and_dataset_reads_it(tmp_path: Path) -> None:
    wav = tmp_path / "clip.wav"
    _write_wav(wav)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    manifest = manifests / "train_content.csv"
    _write_manifest(manifest, [_row(wav, "speaker_a", "wake phrase")])
    (manifests / "keyword_vocab.json").write_text(json.dumps({"wake phrase": 0}))
    (manifests / "speaker_vocab.json").write_text(json.dumps({"speaker_a": 0}))

    pack_manifests(
        manifests_dir=manifests,
        lmdb_path=tmp_path / "processed" / "audio.lmdb",
        index_path=tmp_path / "processed" / "audio_lmdb_index.json",
        map_size_gb=0.01,
    )

    packed = list(csv.DictReader(open(manifest, newline="")))
    assert packed[0]["file_path"].startswith("lmdb://")
    dataset = DualHeadDataset(
        manifest,
        AudioConfig(),
        DataConfig(manifests_dir=manifests),
    )
    assert dataset[0]["wav"].shape == (25600,)
