"""Build LibriPhrase clips from phrase segment metadata.

This script consumes segment metadata produced from the pinned MFA alignment
recipe documented in ``data/README.md``. The input CSV must contain:
``file_path,start_s,end_s,duration_s,phrase,speaker_id``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import soundfile as sf

from solospeak.utils.audio import load_audio_segment


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "phrase"


def _segment_key(file_path: str, start_s: float, end_s: float) -> str:
    raw = f"{file_path}:{start_s:.6f}:{end_s:.6f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def prepare_libriphrase(
    segments_csv: Path,
    output_root: Path = Path("data/raw/libriphrase"),
    max_phrases: int = 1500,
    max_per_phrase_speaker: int = 10,
) -> None:
    with open(segments_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    filtered = [
        row
        for row in rows
        if 0.4 <= float(row["duration_s"]) <= 1.4
        and row["phrase"].strip()
        and row["speaker_id"].strip()
    ]
    phrase_speakers: dict[str, set[str]] = defaultdict(set)
    phrase_counts: dict[str, int] = defaultdict(int)
    for row in filtered:
        phrase = row["phrase"].strip().lower()
        phrase_speakers[phrase].add(row["speaker_id"])
        phrase_counts[phrase] += 1

    eligible = [phrase for phrase, speakers in phrase_speakers.items() if len(speakers) >= 5]
    eligible.sort(key=lambda p: (-len(phrase_speakers[p]), -phrase_counts[p], p))
    keep = set(eligible[:max_phrases])

    per_pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    written = 0
    for row in filtered:
        phrase = row["phrase"].strip().lower()
        speaker_id = row["speaker_id"].strip()
        if phrase not in keep:
            continue
        pair = (phrase, speaker_id)
        if per_pair_counts[pair] >= max_per_phrase_speaker:
            continue
        start_s = float(row["start_s"])
        end_s = float(row["end_s"])
        wav = load_audio_segment(row["file_path"], start_s, end_s)
        dest = output_root / speaker_id / _slug(phrase) / f"{_segment_key(row['file_path'], start_s, end_s)}.wav"
        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dest), wav, 16000)
        per_pair_counts[pair] += 1
        written += 1

    print(f"Wrote {written:,} LibriPhrase clips to {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LibriPhrase clips from MFA segments")
    parser.add_argument("--segments-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/libriphrase"))
    parser.add_argument("--max-phrases", type=int, default=1500)
    parser.add_argument("--max-per-phrase-speaker", type=int, default=10)
    args = parser.parse_args()
    prepare_libriphrase(
        args.segments_csv,
        args.output_root,
        args.max_phrases,
        args.max_per_phrase_speaker,
    )


if __name__ == "__main__":
    main()
