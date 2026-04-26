"""Pack manifest audio into a deterministic LMDB store."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import lmdb
import numpy as np

from scripts.prepare_manifests import MANIFEST_COLUMNS, REQUIRED_MANIFESTS
from solospeak.utils.audio import load_audio_segment, make_lmdb_key
from solospeak.utils.types import FloatArray

LMDB_PREFIX = "lmdb://"


def _audio_bytes(wav: FloatArray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, wav.astype(np.float32, copy=False), allow_pickle=False)
    return buf.getvalue()


def _checksum(wav: FloatArray) -> str:
    return hashlib.sha256(wav.astype(np.float32, copy=False).tobytes()).hexdigest()


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [{col: (row.get(col) or "") for col in MANIFEST_COLUMNS} for row in reader]


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _manifest_paths(manifests_dir: Path) -> list[Path]:
    paths = [manifests_dir / name for name in REQUIRED_MANIFESTS]
    paths.extend(sorted(manifests_dir.glob("stage5_*quadrants.csv")))
    return [path for path in paths if path.exists()]


def _lmdb_uri(lmdb_path: Path, key: str) -> str:
    return f"{LMDB_PREFIX}{lmdb_path.as_posix()}/{key}"


def pack_manifests(
    manifests_dir: Path = Path("data/manifests"),
    lmdb_path: Path = Path("data/processed/audio.lmdb"),
    index_path: Path = Path("data/processed/audio_lmdb_index.json"),
    map_size_gb: float = 64.0,
) -> None:
    lmdb_path.parent.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict[str, str | float]] = {}
    if index_path.exists():
        index = json.loads(index_path.read_text())

    env = lmdb.open(str(lmdb_path), map_size=int(map_size_gb * 1024**3), subdir=True)
    try:
        with env.begin(write=True) as txn:
            for manifest_path in _manifest_paths(manifests_dir):
                rows = _read_manifest(manifest_path)
                rewritten: list[dict[str, str]] = []
                for row in rows:
                    file_path = row["file_path"]
                    duration = float(row["duration_s"] or 0.0)
                    if file_path.startswith(LMDB_PREFIX):
                        rewritten.append(row)
                        continue

                    start_s = float(row["start_s"] or 0.0)
                    end_s = float(row["end_s"] or duration)
                    key = make_lmdb_key(file_path, start_s, end_s)
                    wav = load_audio_segment(file_path, start_s, end_s, target_sr=16000)
                    checksum = _checksum(wav)

                    existing = txn.get(key.encode())
                    if existing is None or index.get(key, {}).get("checksum") != checksum:
                        txn.put(key.encode(), _audio_bytes(wav))

                    packed_duration = len(wav) / 16000.0
                    index[key] = {
                        "source_file_path": file_path,
                        "source_start_s": start_s,
                        "source_end_s": end_s,
                        "duration_s": packed_duration,
                        "checksum": checksum,
                    }
                    packed = dict(row)
                    packed["file_path"] = _lmdb_uri(lmdb_path, key)
                    packed["start_s"] = "0.000000"
                    packed["end_s"] = f"{packed_duration:.6f}"
                    packed["duration_s"] = f"{packed_duration:.6f}"
                    rewritten.append(packed)
                _write_manifest(manifest_path, rewritten)
    finally:
        env.close()

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"Packed audio LMDB: {lmdb_path}")
    print(f"Index: {index_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack manifest audio into LMDB")
    parser.add_argument("--manifests-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--lmdb-path", type=Path, default=Path("data/processed/audio.lmdb"))
    parser.add_argument("--index-path", type=Path, default=Path("data/processed/audio_lmdb_index.json"))
    parser.add_argument("--map-size-gb", type=float, default=64.0)
    args = parser.parse_args()
    pack_manifests(args.manifests_dir, args.lmdb_path, args.index_path, args.map_size_gb)


if __name__ == "__main__":
    main()
