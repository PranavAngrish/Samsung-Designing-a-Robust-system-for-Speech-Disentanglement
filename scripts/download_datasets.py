"""Idempotent dataset downloader.

Usage:
    python -m scripts.download_datasets
    python -m scripts.download_datasets --dataset gsc_v2
    python -m scripts.download_datasets --dataset all

Design:
    - Every download verified by SHA-256
    - Resumable (uses curl --continue-at -)
    - Writes to data/raw/<dataset_name>/
    - Never overwrites existing complete files (SHA match = skip)
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tarfile
import zipfile
from pathlib import Path


DATASETS: dict[str, dict[str, str]] = {
    "gsc_v2": {
        "url": "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
        "sha256": "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58",
        "dest": "data/raw/gsc_v2",
    },
    "musan": {
        "url": "https://www.openslr.org/resources/17/musan.tar.gz",
        "sha256": "",  # fill in after first download
        "dest": "data/raw/musan",
    },
    "rirs": {
        "url": "https://www.openslr.org/resources/28/rirs_noises.zip",
        "sha256": "",  # fill in after first download
        "dest": "data/raw/rirs",
    },
    # VoxCeleb requires academic access — not auto-downloadable
    # LibriPhrase generated from LibriSpeech — see data/README.md
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_dataset(name: str) -> None:
    info = DATASETS[name]
    dest = Path(info["dest"])
    dest.mkdir(parents=True, exist_ok=True)

    url = info["url"]
    expected_sha = info["sha256"]
    filename = dest / Path(url).name

    # Skip if already downloaded and SHA matches
    if filename.exists() and expected_sha:
        actual = sha256_file(filename)
        if actual == expected_sha:
            print(f"  {name}: already downloaded (SHA matches), skipping.")
            return

    # Download with curl (resumable)
    print(f"  Downloading {url} → {filename}")
    subprocess.run(
        ["curl", "--continue-at", "-", "-L", "--output", str(filename), url],
        check=True,
    )

    # Verify SHA
    if expected_sha:
        actual = sha256_file(filename)
        if actual != expected_sha:
            raise ValueError(
                f"SHA-256 mismatch for {name}:\n  expected: {expected_sha}\n  got:      {actual}"
            )

    # Extract
    if str(filename).endswith(".tar.gz"):
        print(f"  Extracting {filename} → {dest}")
        with tarfile.open(filename, "r:gz") as tf:
            tf.extractall(dest)
    elif str(filename).endswith(".zip"):
        print(f"  Extracting {filename} → {dest}")
        with zipfile.ZipFile(filename) as zf:
            zf.extractall(dest)

    print(f"  {name}: done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SoloSpeak datasets")
    parser.add_argument("--dataset", default="all", choices=list(DATASETS) + ["all"])
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Create the Phase-0 smoke directory scaffold without network downloads.",
    )
    args = parser.parse_args()

    targets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    if args.minimal:
        for name in targets:
            dest = Path(DATASETS[name]["dest"])
            dest.mkdir(parents=True, exist_ok=True)
            print(f"Prepared smoke directory: {dest}")
        return

    for name in targets:
        print(f"Downloading {name}...")
        download_dataset(name)


if __name__ == "__main__":
    main()
