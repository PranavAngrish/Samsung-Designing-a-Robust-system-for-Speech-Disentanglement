"""Precompute one ECAPA-TDNN speaker embedding per speaker."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.prepare_manifests import MANIFEST_COLUMNS
from solospeak.data.datasets import _load_row_audio
from solospeak.utils.config import AudioConfig
from solospeak.utils.types import FloatArray


def _first_row_per_speaker(manifest: Path, max_speakers: int | None) -> list[dict[str, str]]:
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    with open(manifest, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {col: (raw.get(col) or "") for col in MANIFEST_COLUMNS}
            speaker_id = row["speaker_id"]
            if speaker_id == "" or speaker_id in seen:
                continue
            seen.add(speaker_id)
            rows.append(row)
            if max_speakers is not None and len(rows) >= max_speakers:
                break
    return rows


def precompute(
    manifest: Path = Path("data/manifests/train_speaker.csv"),
    output_dir: Path = Path("data/processed"),
    device: str = "cpu",
    max_speakers: int | None = None,
) -> None:
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError as exc:
        raise RuntimeError(
            "speechbrain is required for speaker embedding precomputation. "
            "Install project dependencies from pyproject.toml."
        ) from exc

    rows = _first_row_per_speaker(manifest, max_speakers)
    classifier: Any = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="data/processed/speechbrain_spkrec_ecapa_voxceleb",
        run_opts={"device": device},
    )

    embeddings: list[FloatArray] = []
    speaker_ids: list[str] = []
    audio_config = AudioConfig()
    for row in rows:
        wav = _load_row_audio(row, audio_config).to(device)
        with torch.no_grad():
            raw_emb = classifier.encode_batch(wav.unsqueeze(0)).squeeze().detach().cpu().numpy()
        emb = np.asarray(raw_emb, dtype=np.float32)
        emb /= max(float(np.linalg.norm(emb)), 1e-8)
        embeddings.append(emb)
        speaker_ids.append(row["speaker_id"])

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.stack(embeddings, axis=0) if embeddings else np.zeros((0, 192), dtype=np.float32)
    np.save(output_dir / "speaker_embeddings.npy", matrix.astype(np.float32))
    (output_dir / "speaker_embedding_ids.json").write_text(json.dumps(speaker_ids, indent=2) + "\n")
    print(f"Wrote {len(speaker_ids)} speaker embeddings to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute ECAPA speaker embeddings")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/train_speaker.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-speakers", type=int, default=None)
    args = parser.parse_args()
    precompute(args.manifest, args.output_dir, args.device, args.max_speakers)


if __name__ == "__main__":
    main()
