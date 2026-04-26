"""Generate CSV manifests from raw downloaded datasets.

Usage:
    python -m scripts.prepare_manifests
    python -m scripts.prepare_manifests --data-root data/raw --output-dir data/manifests

Output: data/manifests/*.csv + data/manifests/STATS.md
Each CSV row: file_path,duration_s,speaker_id,keyword_text,split,source_dataset,quadrant_class
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from solospeak.data.splits import assign_split


MANIFEST_COLUMNS = [
    "file_path",
    "duration_s",
    "speaker_id",
    "keyword_text",
    "split",
    "source_dataset",
    "quadrant_class",
]

# GSC v2 classes — 35 word labels
GSC_CLASSES = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow",
    "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine",
    "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three",
    "tree", "two", "up", "visual", "wow", "yes", "zero",
]


def audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def _write_manifests(
    rows: dict[str, list[dict[str, str]]],
    output_dir: Path,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, data in rows.items():
        dest = output_dir / name
        with open(dest, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(data)
        counts[name] = len(data)
    return counts


def _write_stats(counts: dict[str, int], output_dir: Path) -> None:
    lines = ["# Manifest Statistics\n"]
    for name, n in sorted(counts.items()):
        lines.append(f"- **{name}**: {n:,} rows\n")
    (output_dir / "STATS.md").write_text("".join(lines))


def _smoke_wave(path: Path, frequency: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sr = 16000
    t = np.linspace(0.0, 0.4, int(sr * 0.4), endpoint=False, dtype=np.float32)
    wav = (0.1 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    sf.write(str(path), wav, sr)


def write_smoke_manifests(output_dir: Path) -> None:
    """Create tiny deterministic manifests for Phase-0 CI smoke checks."""
    audio_dir = Path("data/processed/smoke/audio")
    clips = {
        "spk_a_hello": audio_dir / "spk_a_hello.wav",
        "spk_b_hello": audio_dir / "spk_b_hello.wav",
        "spk_c_world": audio_dir / "spk_c_world.wav",
        "gsc_yes": audio_dir / "gsc_yes.wav",
    }
    for i, path in enumerate(clips.values(), start=1):
        _smoke_wave(path, 220.0 + 55.0 * i)

    def row(path: Path, speaker: str, keyword: str, split: str, dataset: str) -> dict[str, str]:
        return {
            "file_path": str(path),
            "duration_s": "0.400",
            "speaker_id": speaker,
            "keyword_text": keyword,
            "split": split,
            "source_dataset": dataset,
            "quadrant_class": "",
        }

    rows = {
        "train_content.csv": [row(clips["spk_a_hello"], "spk_a", "hello", "train", "smoke")],
        "dev_content.csv": [row(clips["spk_b_hello"], "spk_b", "hello", "dev", "smoke")],
        "train_speaker.csv": [row(clips["spk_a_hello"], "spk_a", "", "train", "smoke")],
        "dev_speaker.csv": [row(clips["spk_b_hello"], "spk_b", "", "dev", "smoke")],
        "train_gsc.csv": [row(clips["gsc_yes"], "gsc_spk_a", "yes", "train", "smoke_gsc")],
        "dev_gsc.csv": [row(clips["gsc_yes"], "gsc_spk_b", "yes", "dev", "smoke_gsc")],
        "test_kpi.csv": [row(clips["spk_c_world"], "spk_c", "world", "test", "smoke")],
        "test_fa.csv": [row(clips["spk_c_world"], "spk_c", "", "test", "smoke")],
    }
    counts = _write_manifests(rows, output_dir)
    _write_stats(counts, output_dir)

    (output_dir / "keyword_vocab.json").write_text(json.dumps(["hello", "world"], indent=2))
    (output_dir / "speaker_vocab.json").write_text(json.dumps(["spk_a", "spk_b", "spk_c"], indent=2))
    (output_dir / "gsc_vocab.json").write_text(json.dumps(GSC_CLASSES, indent=2))
    stats_json = {
        "n_gsc_classes": len(GSC_CLASSES),
        "n_aux_word_classes": 2,
        "n_aux_speaker_classes": 3,
        "smoke": True,
    }
    (output_dir / "STATS.json").write_text(json.dumps(stats_json, indent=2))


def process_gsc(data_root: Path, seed: int) -> dict[str, list[dict[str, str]]]:
    """Process Google Speech Commands v2 into train/dev/test manifest rows."""
    gsc_root = data_root / "gsc_v2"
    rows: dict[str, list[dict[str, str]]] = {
        "train_gsc.csv": [],
        "dev_gsc.csv": [],
    }
    if not gsc_root.exists():
        print(f"  GSC v2 not found at {gsc_root}, skipping.")
        return rows

    for class_dir in sorted(gsc_root.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("_"):
            continue
        keyword = class_dir.name
        if keyword not in GSC_CLASSES:
            continue
        for wav_file in sorted(class_dir.glob("*.wav")):
            # GSC speaker IDs embedded in filename: <hash>_nohash_<n>.wav
            parts = wav_file.stem.split("_nohash_")
            speaker_id = f"gsc_{parts[0]}" if len(parts) >= 1 else f"gsc_{wav_file.stem}"
            split = assign_split(speaker_id, seed=seed)
            try:
                duration = audio_duration(wav_file)
            except Exception:
                continue
            row = {
                "file_path": str(wav_file),
                "duration_s": f"{duration:.3f}",
                "speaker_id": speaker_id,
                "keyword_text": keyword,
                "split": split,
                "source_dataset": "gsc_v2",
                "quadrant_class": "",
            }
            manifest_key = "train_gsc.csv" if split == "train" else "dev_gsc.csv"
            rows[manifest_key].append(row)

    return rows


def process_libri_phrase(data_root: Path, seed: int) -> dict[str, list[dict[str, str]]]:
    """Process LibriPhrase into content train/dev manifest rows."""
    lp_root = data_root / "libriphrase"
    rows: dict[str, list[dict[str, str]]] = {
        "train_content.csv": [],
        "dev_content.csv": [],
    }
    if not lp_root.exists():
        print(f"  LibriPhrase not found at {lp_root}, skipping.")
        return rows

    # Expected structure: libriphrase/<speaker_id>/<keyword>/<file>.wav
    for spk_dir in sorted(lp_root.iterdir()):
        if not spk_dir.is_dir():
            continue
        speaker_id = spk_dir.name
        split = assign_split(speaker_id, seed=seed)
        for kw_dir in sorted(spk_dir.iterdir()):
            if not kw_dir.is_dir():
                continue
            keyword = kw_dir.name
            for wav_file in sorted(kw_dir.glob("*.wav")):
                try:
                    duration = audio_duration(wav_file)
                except Exception:
                    continue
                row = {
                    "file_path": str(wav_file),
                    "duration_s": f"{duration:.3f}",
                    "speaker_id": speaker_id,
                    "keyword_text": keyword,
                    "split": split,
                    "source_dataset": "libriphrase",
                    "quadrant_class": "",
                }
                manifest_key = "train_content.csv" if split == "train" else "dev_content.csv"
                rows[manifest_key].append(row)

    return rows


def process_voxceleb(data_root: Path, seed: int) -> dict[str, list[dict[str, str]]]:
    """Process VoxCeleb2 into speaker train/dev manifest rows."""
    vc_root = data_root / "voxceleb2" / "dev" / "aac"
    rows: dict[str, list[dict[str, str]]] = {
        "train_speaker.csv": [],
        "dev_speaker.csv": [],
    }
    if not vc_root.exists():
        print(f"  VoxCeleb2 not found at {vc_root}, skipping.")
        return rows

    # Expected structure: voxceleb2/dev/aac/<speaker_id>/<video_id>/<segment>.m4a
    for spk_dir in sorted(vc_root.iterdir()):
        if not spk_dir.is_dir():
            continue
        speaker_id = spk_dir.name
        split = assign_split(speaker_id, seed=seed)
        for video_dir in sorted(spk_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            for audio_file in sorted(video_dir.glob("*.wav")) + sorted(video_dir.glob("*.m4a")):  # type: ignore[operator]
                try:
                    duration = audio_duration(audio_file)
                except Exception:
                    continue
                row = {
                    "file_path": str(audio_file),
                    "duration_s": f"{duration:.3f}",
                    "speaker_id": speaker_id,
                    "keyword_text": "",
                    "split": split,
                    "source_dataset": "voxceleb2",
                    "quadrant_class": "",
                }
                manifest_key = "train_speaker.csv" if split == "train" else "dev_speaker.csv"
                rows[manifest_key].append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare training manifests")
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--output-dir", default="data/manifests", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Write tiny deterministic manifests without requiring downloaded datasets.",
    )
    args = parser.parse_args()

    if args.smoke:
        write_smoke_manifests(args.output_dir)
        print(f"Smoke manifests written to {args.output_dir}/")
        return

    print(f"Processing datasets from {args.data_root}...")
    all_rows: dict[str, list[dict[str, str]]] = {}

    for process_fn in [process_gsc, process_libri_phrase, process_voxceleb]:
        rows = process_fn(args.data_root, args.seed)
        for k, v in rows.items():
            all_rows.setdefault(k, []).extend(v)

    # Ensure all required manifests exist (even if empty)
    required = [
        "train_content.csv", "train_speaker.csv", "train_gsc.csv",
        "dev_content.csv", "dev_speaker.csv",
        "test_kpi.csv", "test_fa.csv",
    ]
    for name in required:
        all_rows.setdefault(name, [])

    counts = _write_manifests(all_rows, args.output_dir)
    _write_stats(counts, args.output_dir)

    print("\nManifest summary:")
    for name, n in sorted(counts.items()):
        print(f"  {name}: {n:,} rows")
    print(f"\nOutput written to {args.output_dir}/")


if __name__ == "__main__":
    main()
