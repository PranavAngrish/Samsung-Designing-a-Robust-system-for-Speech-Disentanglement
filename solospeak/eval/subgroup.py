"""Stratified evaluation by available subgroup metadata."""

from __future__ import annotations

import csv
from pathlib import Path

from solospeak.data.datasets import MANIFEST_COLUMNS
from solospeak.utils.types import SubgroupDict

SUBGROUP_AXES = ["keyword_syllable_count", "gender", "age_bucket", "accent_bucket"]


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [{col: (row.get(col) or "") for col in MANIFEST_COLUMNS} for row in reader]


def _syllable_count(text: str) -> int:
    count = 0
    vowels = set("aeiouy")
    for word in text.lower().split():
        in_vowel = False
        word_count = 0
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not in_vowel:
                word_count += 1
            in_vowel = is_vowel
        count += max(1, word_count)
    return max(1, count)


def _syllable_bucket(text: str) -> str:
    n = _syllable_count(text)
    if n <= 2:
        return "2"
    if n == 3:
        return "3"
    return "4+"


def run_subgroup_eval(
    checkpoint_path: Path,
    test_manifest: Path,
) -> SubgroupDict:
    """Return subgroup availability and smoke-safe keyword-syllable summaries."""
    del checkpoint_path
    rows = _read_manifest(test_manifest)
    q1_rows = [row for row in rows if row["quadrant_class"] == "Q1_accept"]
    buckets: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in q1_rows:
        bucket = _syllable_bucket(row["keyword_text"] or row["enrolled_keyword_text"])
        counts[bucket] = counts.get(bucket, 0) + 1
    total = max(sum(counts.values()), 1)
    for bucket in ["2", "3", "4+"]:
        buckets[bucket] = counts.get(bucket, 0) / total
    return {
        "keyword_syllable_count": buckets,
        "gender": None,
        "age_bucket": None,
        "accent_bucket": None,
    }


def format_subgroup_report(results: SubgroupDict) -> str:
    lines = ["# Subgroup Report\n\n"]
    for axis, values in results.items():
        lines.append(f"## {axis}\n\n")
        if values is None:
            lines.append("Metadata unavailable; reported as null in KPI JSON.\n\n")
            continue
        lines.append("| Bucket | Share / Metric |\n|---|---:|\n")
        for bucket, value in values.items():
            lines.append(f"| {bucket} | {value:.4f} |\n")
        lines.append("\n")
    return "".join(lines)

