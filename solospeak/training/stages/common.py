"""Shared helpers for Phase-3 training stages."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from solospeak.data.features import LogMelExtractor
from solospeak.utils.config import SoloSpeakConfig
from solospeak.utils.types import BatchDict, BatchValue, MetricsDict


def load_stats(config: SoloSpeakConfig) -> dict[str, Any]:
    """Load manifest statistics written by ``prepare_manifests.py``."""
    stats_path = config.data.manifests_dir / "STATS.json"
    if not stats_path.exists():
        return {}
    return cast(dict[str, Any], json.loads(stats_path.read_text()))


def inject_class_counts(config: SoloSpeakConfig) -> None:
    """Inject runtime class counts from STATS.json into the mutable config object."""
    stats = load_stats(config)
    training = config.training
    if training.n_gsc_classes is None and "n_gsc_classes" in stats:
        training.n_gsc_classes = int(stats["n_gsc_classes"])
    if training.n_aux_word_classes is None and "n_aux_word_classes" in stats:
        training.n_aux_word_classes = int(stats["n_aux_word_classes"])
    if training.n_aux_speaker_classes is None and "n_aux_speaker_classes" in stats:
        training.n_aux_speaker_classes = int(stats["n_aux_speaker_classes"])


def is_smoke(config: SoloSpeakConfig) -> bool:
    """Return True when manifests are the tiny local smoke set."""
    return bool(load_stats(config).get("smoke", False))


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def smoke_batch_size(config: SoloSpeakConfig, *, cap: int = 16) -> int:
    """Keep smoke commands quick while preserving the full config for real runs."""
    if is_smoke(config):
        return min(config.training.batch_size, cap)
    return config.training.batch_size


def smoke_epochs(config: SoloSpeakConfig) -> int:
    return 1 if is_smoke(config) else config.training.num_epochs


def smoke_max_steps(config: SoloSpeakConfig) -> int | None:
    return 2 if is_smoke(config) else None


def to_device(batch: BatchDict, device: torch.device) -> BatchDict:
    out: BatchDict = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def require_tensor(batch: Mapping[str, BatchValue], key: str) -> torch.Tensor:
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor batch field {key!r}, got {type(value).__name__}")
    return value


def make_mel(batch: Mapping[str, BatchValue], extractor: LogMelExtractor) -> torch.Tensor:
    wav = require_tensor(batch, "wav")
    return cast(torch.Tensor, extractor(wav))


def backward(loss: torch.Tensor) -> None:
    """Call Tensor.backward while keeping mypy strict happy for PyTorch stubs."""
    cast(Any, loss.backward)()


def save_checkpoint(
    path: Path,
    *,
    stage_origin: int,
    config: SoloSpeakConfig,
    step: int,
    metrics: MetricsDict,
    model_state: Mapping[str, torch.Tensor] | None = None,
    wrapper_state: Mapping[str, torch.Tensor] | None = None,
    optimizer_state: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write a stage checkpoint using the Phase-3 handoff schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config_snapshot": config.model_dump(mode="json"),
        "step": step,
        "metrics": metrics,
        "stage_origin": stage_origin,
    }
    if model_state is not None:
        payload["model_state"] = dict(model_state)
    if wrapper_state is not None:
        payload["wrapper_state"] = dict(wrapper_state)
    if optimizer_state is not None:
        payload["optimizer_state"] = dict(optimizer_state)
    payload["extra"] = dict(extra or {})
    if extra is not None:
        # Compatibility for older notebook-style readers that looked for
        # extra fields at the checkpoint top level.
        payload.update(dict(extra))
    torch.save(payload, path)
    update_latest(path)
    return path


def update_latest(path: Path) -> None:
    latest = path.parent / "latest.pt"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(path.name)
    except OSError:
        torch.save(torch.load(path, map_location="cpu"), latest)


def load_checkpoint(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))


def checkpoint_path(config: SoloSpeakConfig, name: str) -> Path:
    return config.training.checkpoint_dir / name


def pass_or_raise(
    *,
    stage_name: str,
    smoke: bool,
    metrics: MetricsDict,
    metric_name: str,
    min_threshold: float,
    target_threshold: float,
) -> tuple[bool, bool]:
    """Apply MIN/TARGET gate semantics, with explicit smoke-only bypass."""
    if smoke:
        metrics["gate/smoke_only"] = 1.0
        return True, True
    value = metrics.get(metric_name)
    if value is None:
        raise RuntimeError(f"{stage_name}: missing gate metric {metric_name!r}")
    passed_min = value >= min_threshold
    passed_target = value >= target_threshold
    if not passed_min:
        raise RuntimeError(
            f"{stage_name}: MIN gate failed for {metric_name}: "
            f"{value:.4f} < {min_threshold:.4f}"
        )
    return passed_min, passed_target


def freeze(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False
