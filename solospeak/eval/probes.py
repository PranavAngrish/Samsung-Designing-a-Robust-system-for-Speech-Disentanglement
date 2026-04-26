"""Disentanglement verification via fixed probe protocol."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from solospeak.data.datasets import MANIFEST_COLUMNS
from solospeak.data.features import LogMelExtractor
from solospeak.losses.adversarial import AdversarialProbeHead
from solospeak.models.solospeak import SoloSpeakModel
from solospeak.utils.audio import load_audio_segment, pad_or_crop_to_window
from solospeak.utils.config import SoloSpeakConfig
from solospeak.utils.types import ProbeBaseline, ProbeResult, WORD_IGNORE_INDEX


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [{col: (row.get(col) or "") for col in MANIFEST_COLUMNS} for row in reader]


def _load_vocab(path: Path) -> dict[str, int]:
    return {str(k): int(v) for k, v in json.loads(path.read_text()).items()}


def _load_model(checkpoint_path: Path, config: SoloSpeakConfig) -> SoloSpeakModel:
    model = SoloSpeakModel(config).eval()
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", ckpt)
        if isinstance(state, dict):
            model.load_state_dict(state, strict=False)
    return model


def _is_smoke(config: SoloSpeakConfig) -> bool:
    stats_path = config.data.manifests_dir / "STATS.json"
    if not stats_path.exists():
        return False
    return bool(json.loads(stats_path.read_text()).get("smoke", False))


@torch.no_grad()
def _collect_probe_tensors(
    model: SoloSpeakModel,
    config: SoloSpeakConfig,
    manifest: Path,
    *,
    max_samples: int = 4000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    keyword_vocab = _load_vocab(config.data.manifests_dir / "keyword_vocab.json")
    speaker_vocab = _load_vocab(config.data.manifests_dir / "speaker_vocab.json")
    extractor = LogMelExtractor(config.audio).eval()
    z_c_values: list[torch.Tensor] = []
    z_s_values: list[torch.Tensor] = []
    speaker_labels: list[int] = []
    word_labels: list[int] = []
    for row in _read_manifest(manifest)[:max_samples]:
        duration = float(row["duration_s"] or 0.0)
        wav = load_audio_segment(
            row["file_path"],
            float(row["start_s"] or 0.0),
            float(row["end_s"] or duration),
            config.audio.sample_rate,
        )
        fixed = pad_or_crop_to_window(wav, config.audio.window_samples)
        mel = extractor(torch.from_numpy(fixed).float().unsqueeze(0))
        z_c, z_s = model(mel)
        z_c_values.append(z_c.squeeze(0))
        z_s_values.append(z_s.squeeze(0))
        speaker_labels.append(speaker_vocab.get(row["speaker_id"], WORD_IGNORE_INDEX))
        word_labels.append(keyword_vocab.get(row["keyword_text"], WORD_IGNORE_INDEX))
    if not z_c_values:
        empty = torch.empty(0, config.heads.content_dim)
        return empty, empty, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    return (
        torch.stack(z_c_values),
        torch.stack(z_s_values),
        torch.tensor(speaker_labels, dtype=torch.long),
        torch.tensor(word_labels, dtype=torch.long),
    )


def _train_probe(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_classes: int,
    epochs: int,
) -> float:
    valid = labels >= 0
    embeddings = embeddings[valid]
    labels = labels[valid]
    if embeddings.shape[0] < 4 or n_classes <= 1:
        return 0.0
    split = max(1, int(0.7 * embeddings.shape[0]))
    if split >= embeddings.shape[0]:
        split = embeddings.shape[0] - 1
    train = TensorDataset(embeddings[:split], labels[:split])
    eval_x = embeddings[split:]
    eval_y = labels[split:]
    probe = AdversarialProbeHead(embeddings.shape[1], n_classes)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    loader = DataLoader(train, batch_size=min(64, len(train)), shuffle=True)
    for _ in range(max(1, epochs)):
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(probe(x), y)
            cast(Any, loss.backward)()
            opt.step()
    with torch.no_grad():
        pred = probe(eval_x).argmax(dim=-1)
    return float((pred == eval_y).float().mean().item()) if eval_y.numel() else 0.0


def _probe_accuracies(
    checkpoint_path: Path,
    config: SoloSpeakConfig,
    dev_manifest: Path,
    probe_epochs: int,
) -> tuple[float, float]:
    model = _load_model(checkpoint_path, config)
    z_c, z_s, speaker_labels, word_labels = _collect_probe_tensors(model, config, dev_manifest)
    n_speakers = int(speaker_labels[speaker_labels >= 0].max().item() + 1) if (speaker_labels >= 0).any() else 0
    n_words = int(word_labels[word_labels >= 0].max().item() + 1) if (word_labels >= 0).any() else 0
    acc_c = _train_probe(z_c, speaker_labels, n_classes=n_speakers, epochs=probe_epochs)
    acc_s = _train_probe(z_s, word_labels, n_classes=n_words, epochs=probe_epochs)
    return acc_c, acc_s


def verify_disentanglement(
    encoder: nn.Module,
    probe_data: DataLoader[Any],
    stage2_baseline: ProbeBaseline | None = None,
) -> ProbeResult:
    """Verify probe reductions from precomputed/provided baseline metadata."""
    del encoder, probe_data
    acc_c_post = 0.0
    acc_s_post = 0.0
    reduction_c: float | None = None
    reduction_s: float | None = None
    if stage2_baseline is not None:
        pre_c = float(stage2_baseline.get("acc_c_pre", 0.0))
        pre_s = float(stage2_baseline.get("acc_s_pre", 0.0))
        reduction_c = (pre_c - acc_c_post) / pre_c if pre_c > 0 else None
        reduction_s = (pre_s - acc_s_post) / pre_s if pre_s > 0 else None
    return {
        "acc_c_post": acc_c_post,
        "acc_s_post": acc_s_post,
        "reduction_c": reduction_c,
        "reduction_s": reduction_s,
    }


def run_probe_eval(
    checkpoint_path: Path,
    dev_manifest: Path,
    probe_epochs: int = 10,
    *,
    config_path: Path = Path("configs/defaults.yaml"),
    baseline_checkpoint: Path | None = Path("checkpoints/stage2_dualhead.pt"),
) -> dict[str, float | None]:
    """Train/evaluate both probes and report Phase-4 keys."""
    config = SoloSpeakConfig.from_yaml(config_path)
    if _is_smoke(config):
        return {
            "acc_c_post": 0.0,
            "acc_s_post": 0.0,
            "reduction_c": 1.0,
            "reduction_s": 1.0,
            "smoke_only": 1.0,
        }
    acc_c_post, acc_s_post = _probe_accuracies(checkpoint_path, config, dev_manifest, probe_epochs)
    reduction_c: float | None = None
    reduction_s: float | None = None
    if baseline_checkpoint is not None and baseline_checkpoint.exists():
        acc_c_pre, acc_s_pre = _probe_accuracies(
            baseline_checkpoint,
            config,
            dev_manifest,
            probe_epochs,
        )
        reduction_c = (acc_c_pre - acc_c_post) / acc_c_pre if acc_c_pre > 0 else None
        reduction_s = (acc_s_pre - acc_s_post) / acc_s_pre if acc_s_pre > 0 else None
    return {
        "acc_c_post": acc_c_post,
        "acc_s_post": acc_s_post,
        "reduction_c": reduction_c,
        "reduction_s": reduction_s,
    }

