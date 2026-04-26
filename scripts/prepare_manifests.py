"""Generate canonical SoloSpeak CSV manifests from raw or smoke data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import soundfile as sf

from solospeak.data.splits import assign_split

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

REQUIRED_MANIFESTS = [
    "train_content.csv",
    "train_speaker.csv",
    "train_gsc.csv",
    "dev_content.csv",
    "dev_speaker.csv",
    "test_kpi.csv",
    "test_fa.csv",
]

GSC_CLASSES = [
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero",
]


def audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames / info.samplerate)


def _row(
    file_path: Path | str,
    duration_s: float,
    speaker_id: str,
    keyword_text: str,
    split: str,
    source_dataset: str,
    *,
    start_s: float = 0.0,
    end_s: float | None = None,
    quadrant_class: str = "",
    profile_id: str = "",
    enrolled_user_id: str = "",
    enrolled_keyword_text: str = "",
    profile_path: str = "",
    trial_source: str = "real",
    q3_gate_eligible: bool = True,
    synthesis_backend: str = "",
    speaker_verification_score: str = "",
) -> dict[str, str]:
    end = duration_s if end_s is None else end_s
    return {
        "file_path": str(file_path),
        "start_s": f"{start_s:.6f}",
        "end_s": f"{end:.6f}",
        "duration_s": f"{duration_s:.6f}",
        "speaker_id": speaker_id,
        "keyword_text": keyword_text,
        "split": split,
        "source_dataset": source_dataset,
        "quadrant_class": quadrant_class,
        "profile_id": profile_id,
        "enrolled_user_id": enrolled_user_id,
        "enrolled_keyword_text": enrolled_keyword_text,
        "profile_path": profile_path,
        "trial_source": trial_source,
        "q3_gate_eligible": "true" if q3_gate_eligible else "false",
        "synthesis_backend": synthesis_backend,
        "speaker_verification_score": speaker_verification_score,
    }


def _write_manifests(
    rows: dict[str, list[dict[str, str]]],
    output_dir: Path,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name in REQUIRED_MANIFESTS:
        data = rows.get(name, [])
        with open(output_dir / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
        counts[name] = len(data)
    return counts


def _vocab_from(values: set[str]) -> dict[str, int]:
    return {value: idx for idx, value in enumerate(sorted(v for v in values if v))}


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _write_stats(
    rows: dict[str, list[dict[str, str]]],
    output_dir: Path,
    *,
    smoke: bool,
) -> None:
    train_content = rows.get("train_content.csv", [])
    manifest_stats: dict[str, dict[str, int]] = {
        name: {
            "row_count": len(data),
            "unique_speakers": len({row["speaker_id"] for row in data if row["speaker_id"]}),
            "unique_keywords": len({row["keyword_text"] for row in data if row["keyword_text"]}),
        }
        for name, data in rows.items()
    }
    stats: dict[str, Any] = {
        "smoke": smoke,
        "manifests": manifest_stats,
        "train_content": {
            "row_count": len(train_content),
            "unique_phrase_speaker_pairs": len(
                {(row["keyword_text"], row["speaker_id"]) for row in train_content}
            ),
        },
        "n_gsc_classes": len(GSC_CLASSES),
        "n_aux_word_classes": len(json.loads((output_dir / "keyword_vocab.json").read_text())),
        "n_aux_speaker_classes": len(json.loads((output_dir / "speaker_vocab.json").read_text())),
    }
    _write_json(output_dir / "STATS.json", stats)

    lines = ["# Manifest Statistics\n\n"]
    for name in REQUIRED_MANIFESTS:
        entry = manifest_stats[name]
        lines.append(
            f"- **{name}**: {entry['row_count']:,} rows, "
            f"{entry['unique_speakers']:,} speakers, {entry['unique_keywords']:,} keywords\n"
        )
    lines.extend(
        [
            f"- **n_gsc_classes**: {cast(int, stats['n_gsc_classes'])}\n",
            f"- **n_aux_word_classes**: {cast(int, stats['n_aux_word_classes'])}\n",
            f"- **n_aux_speaker_classes**: {cast(int, stats['n_aux_speaker_classes'])}\n",
            f"- **smoke**: {str(smoke).lower()}\n",
        ]
    )
    (output_dir / "STATS.md").write_text("".join(lines))


def _write_wave(path: Path, seconds: float, frequency: float, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sr = 16000
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    tone = 0.08 * np.sin(2 * np.pi * frequency * t)
    noise = 0.01 * rng.standard_normal(len(t), dtype=np.float32)
    wav = np.clip(tone + noise, -1.0, 1.0).astype(np.float32)
    sf.write(str(path), wav, sr)


def write_smoke_manifests(output_dir: Path) -> None:
    """Create tiny deterministic manifests for Phase-1 CI/dev smoke checks."""
    rows: dict[str, list[dict[str, str]]] = {name: [] for name in REQUIRED_MANIFESTS}
    raw_root = Path("data/raw/smoke")

    smoke_keywords = [f"wake phrase {i}" for i in range(8)]
    train_speakers = [f"spk_train_{i:02d}" for i in range(8)]
    dev_speakers = [f"spk_dev_{i:02d}" for i in range(4)]
    speaker_train_ids = [f"speaker_train_{i:02d}" for i in range(12)]
    speaker_dev_ids = [f"speaker_dev_{i:02d}" for i in range(4)]

    for keyword_idx, keyword in enumerate(GSC_CLASSES):
        for clip_idx in range(10):
            path = raw_root / "gsc" / keyword / f"{keyword}_{clip_idx:02d}.wav"
            _write_wave(path, 1.6, 180.0 + keyword_idx * 7.0, seed=keyword_idx * 100 + clip_idx)
            rows["train_gsc.csv"].append(
                _row(path, 1.6, f"gsc_{keyword}_{clip_idx:02d}", keyword, "train", "smoke")
            )

    for i in range(80):
        speaker = train_speakers[i % len(train_speakers)]
        keyword = smoke_keywords[i % len(smoke_keywords)]
        path = raw_root / "content" / speaker / f"train_content_{i:03d}.wav"
        _write_wave(path, 1.6, 260.0 + (i % 8) * 11.0, seed=1000 + i)
        rows["train_content.csv"].append(_row(path, 1.6, speaker, keyword, "train", "smoke"))

    for i in range(24):
        speaker = dev_speakers[i % len(dev_speakers)]
        keyword = smoke_keywords[i % 4]
        path = raw_root / "content" / speaker / f"dev_content_{i:03d}.wav"
        _write_wave(path, 1.6, 310.0 + (i % 4) * 13.0, seed=2000 + i)
        rows["dev_content.csv"].append(_row(path, 1.6, speaker, keyword, "dev", "smoke"))

    for i in range(80):
        speaker = speaker_train_ids[i % len(speaker_train_ids)]
        path = raw_root / "speaker" / speaker / f"train_speaker_{i:03d}.wav"
        _write_wave(path, 1.6, 400.0 + (i % 12) * 9.0, seed=3000 + i)
        rows["train_speaker.csv"].append(_row(path, 1.6, speaker, "", "train", "smoke"))

    for i in range(24):
        speaker = speaker_dev_ids[i % len(speaker_dev_ids)]
        path = raw_root / "speaker" / speaker / f"dev_speaker_{i:03d}.wav"
        _write_wave(path, 1.6, 500.0 + (i % 4) * 17.0, seed=4000 + i)
        rows["dev_speaker.csv"].append(_row(path, 1.6, speaker, "", "dev", "smoke"))

    quadrants = ["Q1_accept", "Q2_imposter", "Q3_wrong_word", "Q4_background"]
    for i in range(40):
        quadrant = quadrants[i // 10]
        profile_idx = i % 4
        speaker = f"profile_{profile_idx:02d}" if quadrant != "Q2_imposter" else "imposter"
        keyword = smoke_keywords[profile_idx]
        path = raw_root / "kpi" / f"{quadrant}_{i:03d}.wav"
        _write_wave(path, 1.6, 620.0 + i * 3.0, seed=5000 + i)
        rows["test_kpi.csv"].append(
            _row(
                path,
                1.6,
                speaker,
                keyword if quadrant != "Q4_background" else "",
                "test",
                "smoke",
                quadrant_class=quadrant,
                profile_id=f"profile_{profile_idx:02d}",
                enrolled_user_id=f"user_{profile_idx:02d}",
                enrolled_keyword_text=keyword,
                profile_path="",
                trial_source="background" if quadrant == "Q4_background" else "real",
                q3_gate_eligible=True,
            )
        )

    for i in range(6):
        path = raw_root / "fa" / f"background_{i:02d}.wav"
        _write_wave(path, 10.0, 90.0 + i * 15.0, seed=6000 + i)
        rows["test_fa.csv"].append(
            _row(path, 10.0, f"background_{i:02d}", "", "test", "smoke", trial_source="background")
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    keyword_vocab = {keyword: idx for idx, keyword in enumerate(smoke_keywords)}
    speaker_vocab = _vocab_from(
        {
            row["speaker_id"]
            for name in ["train_speaker.csv", "dev_speaker.csv", "train_content.csv", "dev_content.csv"]
            for row in rows[name]
            if row["speaker_id"]
        }
    )
    source_vocab = _vocab_from({row["source_dataset"] for name in rows for row in rows[name]})
    _write_json(output_dir / "keyword_vocab.json", keyword_vocab)
    _write_json(output_dir / "gsc_vocab.json", {word: idx for idx, word in enumerate(GSC_CLASSES)})
    _write_json(output_dir / "speaker_vocab.json", speaker_vocab)
    _write_json(output_dir / "source_vocab.json", source_vocab)
    _write_manifests(rows, output_dir)
    _write_stats(rows, output_dir, smoke=True)


def process_gsc(data_root: Path, seed: int) -> dict[str, list[dict[str, str]]]:
    gsc_root = data_root / "gsc_v2"
    rows: dict[str, list[dict[str, str]]] = {"train_gsc.csv": [], "dev_gsc.csv": []}
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
            speaker_hash = wav_file.stem.split("_nohash_")[0]
            speaker_id = f"gsc_{speaker_hash}"
            split = assign_split(speaker_id, seed=seed)
            manifest_key = "train_gsc.csv" if split == "train" else "dev_gsc.csv"
            duration = audio_duration(wav_file)
            rows[manifest_key].append(
                _row(wav_file, duration, speaker_id, keyword, split, "gsc_v2")
            )
    return rows


def process_libriphrase(data_root: Path, seed: int) -> dict[str, list[dict[str, str]]]:
    lp_root = data_root / "libriphrase"
    rows: dict[str, list[dict[str, str]]] = {"train_content.csv": [], "dev_content.csv": []}
    if not lp_root.exists():
        print(f"  LibriPhrase not found at {lp_root}, skipping.")
        return rows

    for spk_dir in sorted(lp_root.iterdir()):
        if not spk_dir.is_dir():
            continue
        speaker_id = spk_dir.name
        split = assign_split(speaker_id, seed=seed)
        manifest_key = "train_content.csv" if split == "train" else "dev_content.csv"
        for kw_dir in sorted(spk_dir.iterdir()):
            if not kw_dir.is_dir():
                continue
            keyword = kw_dir.name.replace("_", " ")
            for wav_file in sorted(kw_dir.glob("*.wav")):
                duration = audio_duration(wav_file)
                rows[manifest_key].append(
                    _row(wav_file, duration, speaker_id, keyword, split, "libriphrase")
                )
    return rows


def process_voxceleb(data_root: Path, seed: int) -> dict[str, list[dict[str, str]]]:
    vc_root = data_root / "voxceleb2" / "dev" / "aac"
    rows: dict[str, list[dict[str, str]]] = {"train_speaker.csv": [], "dev_speaker.csv": []}
    if not vc_root.exists():
        print(f"  VoxCeleb2 not found at {vc_root}, skipping.")
        return rows

    per_speaker_counts: dict[str, int] = {}
    for spk_dir in sorted(vc_root.iterdir()):
        if not spk_dir.is_dir():
            continue
        speaker_id = spk_dir.name
        split = assign_split(speaker_id, seed=seed)
        manifest_key = "train_speaker.csv" if split == "train" else "dev_speaker.csv"
        for audio_file in sorted(spk_dir.rglob("*.wav")) + sorted(spk_dir.rglob("*.m4a")):
            if per_speaker_counts.get(speaker_id, 0) >= 50:
                break
            try:
                duration = audio_duration(audio_file)
            except Exception:
                continue
            per_speaker_counts[speaker_id] = per_speaker_counts.get(speaker_id, 0) + 1
            rows[manifest_key].append(_row(audio_file, duration, speaker_id, "", split, "voxceleb2"))
    return rows


def _write_full_vocabs(rows: dict[str, list[dict[str, str]]], output_dir: Path) -> None:
    keyword_values = {
        row["keyword_text"]
        for name in ["train_content.csv", "dev_content.csv", "test_kpi.csv"]
        for row in rows.get(name, [])
        if row["keyword_text"]
    }
    speaker_values = {
        row["speaker_id"]
        for name in ["train_speaker.csv", "dev_speaker.csv", "train_content.csv", "dev_content.csv"]
        for row in rows[name]
        if row["speaker_id"]
    }
    source_values = {row["source_dataset"] for name in rows for row in rows[name]}
    _write_json(output_dir / "keyword_vocab.json", _vocab_from(keyword_values))
    _write_json(output_dir / "gsc_vocab.json", {word: idx for idx, word in enumerate(GSC_CLASSES)})
    _write_json(output_dir / "speaker_vocab.json", _vocab_from(speaker_values))
    _write_json(output_dir / "source_vocab.json", _vocab_from(source_values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare training manifests")
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--output-dir", default="data/manifests", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        write_smoke_manifests(args.output_dir)
        print(f"Smoke manifests written to {args.output_dir}/")
        return

    print(f"Processing datasets from {args.data_root}...")
    all_rows: dict[str, list[dict[str, str]]] = {name: [] for name in REQUIRED_MANIFESTS}
    for process_fn in [process_gsc, process_libriphrase, process_voxceleb]:
        for name, data in process_fn(args.data_root, args.seed).items():
            all_rows.setdefault(name, []).extend(data)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_full_vocabs(all_rows, args.output_dir)
    _write_manifests(all_rows, args.output_dir)
    _write_stats(all_rows, args.output_dir, smoke=False)

    print("\nManifest summary:")
    for name in REQUIRED_MANIFESTS:
        print(f"  {name}: {len(all_rows[name]):,} rows")
    print(f"\nOutput written to {args.output_dir}/")


if __name__ == "__main__":
    main()
