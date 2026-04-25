"""Integration tests for the data pipeline.

These tests require manifest files to exist but do NOT require the actual
audio files — they test the split logic and manifest schema.
The collate and dataset tests use in-memory synthetic data.
"""

from __future__ import annotations

import pytest
import torch

from solospeak.data.splits import assign_split
from solospeak.data.datasets import collate_variable_length


# ---------------------------------------------------------------------------
# Split logic (no data needed)
# ---------------------------------------------------------------------------

def test_assign_split_is_deterministic() -> None:
    for speaker_id in ["spk001", "spk002", "spk003"]:
        s1 = assign_split(speaker_id, seed=42)
        s2 = assign_split(speaker_id, seed=42)
        assert s1 == s2


def test_assign_split_returns_valid_split() -> None:
    for i in range(100):
        s = assign_split(f"speaker_{i}", seed=42)
        assert s in {"train", "dev", "test"}


def test_no_speaker_overlap() -> None:
    """No speaker_id should appear in more than one split — guaranteed by design."""
    speaker_ids = [f"spk_{i:04d}" for i in range(1000)]
    splits: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    for sid in speaker_ids:
        s = assign_split(sid, seed=42)
        splits[s].add(sid)

    all_pairs = [
        (splits["train"], splits["dev"]),
        (splits["train"], splits["test"]),
        (splits["dev"],   splits["test"]),
    ]
    for a, b in all_pairs:
        overlap = a & b
        assert len(overlap) == 0, f"Overlap found: {overlap}"


def test_split_proportions_approx() -> None:
    """Proportions should be within 2pp of targets."""
    speaker_ids = [f"spk_{i:06d}" for i in range(10_000)]
    counts: dict[str, int] = {"train": 0, "dev": 0, "test": 0}
    for sid in speaker_ids:
        counts[assign_split(sid)] += 1

    n = len(speaker_ids)
    assert abs(counts["train"] / n - 0.85) < 0.02
    assert abs(counts["dev"]   / n - 0.10) < 0.02
    assert abs(counts["test"]  / n - 0.05) < 0.02


# ---------------------------------------------------------------------------
# Collation (no data needed)
# ---------------------------------------------------------------------------

def test_collate_pads_to_max_length() -> None:
    batch = [
        {"wav": torch.randn(1000), "label": torch.tensor(0)},
        {"wav": torch.randn(1500), "label": torch.tensor(1)},
        {"wav": torch.randn(800),  "label": torch.tensor(2)},
    ]
    out = collate_variable_length(batch)
    assert out["wav"].shape == (3, 1500)
    assert out["lengths"].tolist() == [1000, 1500, 800]
    assert out["label"].shape == (3,)


def test_collate_padding_is_zero() -> None:
    batch = [
        {"wav": torch.ones(500)},
        {"wav": torch.ones(1000)},
    ]
    out = collate_variable_length(batch)
    # Second half of shorter clip must be zero-padded
    assert out["wav"][0, 500:].abs().max().item() == 0.0


def test_collate_uniform_length_no_padding() -> None:
    batch = [{"wav": torch.randn(1000)} for _ in range(4)]
    out = collate_variable_length(batch)
    assert out["wav"].shape == (4, 1000)
    assert (out["lengths"] == 1000).all()


def test_collate_preserves_label_values() -> None:
    batch = [
        {"wav": torch.randn(800), "label": torch.tensor(i)}
        for i in range(5)
    ]
    out = collate_variable_length(batch)
    assert out["label"].tolist() == list(range(5))


# ---------------------------------------------------------------------------
# Manifests (require make prepare-manifests)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_manifests_exist() -> None:
    """Skip if manifests not yet generated — run after make prepare-manifests."""
    from pathlib import Path
    manifests_dir = Path("data/manifests")
    expected = [
        "train_content.csv", "train_speaker.csv", "train_gsc.csv",
        "dev_content.csv", "dev_speaker.csv",
        "test_kpi.csv", "test_fa.csv",
    ]
    for name in expected:
        assert (manifests_dir / name).exists(), f"Missing manifest: {name}"


@pytest.mark.slow
def test_dataloader_throughput(tmp_path: "pytest.TempdirFixture") -> None:
    """DataLoader must yield ≥200 samples/sec on CPU."""
    import time
    import csv
    import numpy as np
    import soundfile as sf
    from torch.utils.data import DataLoader
    from solospeak.data.datasets import DualHeadDataset, collate_variable_length
    from solospeak.utils.config import AudioConfig, DataConfig

    # Build a small synthetic manifest + audio files
    n_items = 50
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    manifest = tmp_path / "manifest.csv"

    with open(manifest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "duration_s", "speaker_id", "keyword_text",
                          "split", "source_dataset", "quadrant_class"])
        for i in range(n_items):
            wav_path = audio_dir / f"clip_{i}.wav"
            wav = np.random.randn(16000).astype(np.float32) * 0.1
            sf.write(str(wav_path), wav, 16000)
            writer.writerow([str(wav_path), "1.0", f"spk_{i % 5}", "hello",
                              "train", "synthetic", ""])

    dataset = DualHeadDataset(manifest, AudioConfig(), DataConfig())
    loader = DataLoader(dataset, batch_size=16, num_workers=0,
                        collate_fn=collate_variable_length)

    start = time.perf_counter()
    total = 0
    for batch in loader:
        total += batch["wav"].shape[0]
    elapsed = time.perf_counter() - start

    throughput = total / elapsed
    assert throughput >= 200, f"DataLoader throughput {throughput:.0f} samples/sec < 200"
