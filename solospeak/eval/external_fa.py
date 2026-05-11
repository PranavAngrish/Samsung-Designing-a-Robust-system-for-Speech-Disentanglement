"""External false-accept discovery, trial building, scoring, and summaries.

This is the source-code counterpart of the Stage-7 external FA cells in
``ta-test.ipynb``. It keeps the trial counts, row schema, and output score
format in one reusable place while requiring dataset roots from config or CLI.
"""

from __future__ import annotations

import csv
import os
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from solospeak.data.features import LogMelExtractor
from solospeak.eval.internal_quadrants import load_profile_npz
from solospeak.training.stages.common import make_mel
from solospeak.utils.audio import load_audio_segment, pad_or_crop_to_window
from solospeak.utils.config import SoloSpeakConfig


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
TRIALS_PER_COMMON_VOICE_FILE = 3
TRIALS_PER_LIBRISPEECH_FILE = 2
TRIALS_PER_BACKGROUND_FILE = 3
TRIALS_PER_URBANSOUND_FILE = 3
MAX_TOTAL_EXTERNAL_TRIALS = 40000
SCORE_BATCH_SIZE = 128
DISCOVERY_SEED = 123
TRIAL_SEED = 999


@dataclass(frozen=True)
class ExternalFaRoots:
    common_voice: tuple[Path, ...] = ()
    librispeech: tuple[Path, ...] = ()
    background_noise: tuple[Path, ...] = ()
    urbansound8k: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ExternalFaSpec:
    roots: ExternalFaRoots = field(default_factory=ExternalFaRoots)
    max_common_voice_files: int = 8000
    max_librispeech_files: int = 5000
    max_background_files: int = 1500
    max_urbansound_files: int = 3000
    trials_per_common_voice_file: int = TRIALS_PER_COMMON_VOICE_FILE
    trials_per_librispeech_file: int = TRIALS_PER_LIBRISPEECH_FILE
    trials_per_background_file: int = TRIALS_PER_BACKGROUND_FILE
    trials_per_urbansound_file: int = TRIALS_PER_URBANSOUND_FILE
    max_total_external_trials: int = MAX_TOTAL_EXTERNAL_TRIALS
    discovery_seed: int = DISCOVERY_SEED
    trial_seed: int = TRIAL_SEED


def external_fa_spec_from_config(config: SoloSpeakConfig) -> ExternalFaSpec:
    """Build external FA discovery settings from config-provided roots."""

    roots = ExternalFaRoots(
        common_voice=tuple(config.data.external_common_voice_roots),
        librispeech=tuple(config.data.external_librispeech_roots),
        background_noise=tuple(config.data.external_background_noise_roots),
        urbansound8k=tuple(config.data.external_urbansound_roots),
    )
    return ExternalFaSpec(
        roots=roots,
        max_total_external_trials=(
            config.stage7.max_external_fa_trials
            if hasattr(config, "stage7")
            else config.data.external_fa_max_total_trials
        ),
    )


def find_audio_files_fast(roots: Sequence[Path], max_files: int | None = None) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                name
                for name in dirnames
                if not name.startswith(".") and name not in {"__MACOSX", ".git", "metadata"}
            ]
            for filename in filenames:
                path = Path(dirpath) / filename
                if path.suffix.lower() in AUDIO_EXTS:
                    files.append(path)
                    if max_files is not None and len(files) >= max_files:
                        return files
    return files


def discover_external_fa_audio(spec: ExternalFaSpec = ExternalFaSpec()) -> dict[str, list[Path]]:
    rng = random.Random(spec.discovery_seed)
    files_by_source = {
        "common_voice": find_audio_files_fast(
            spec.roots.common_voice, spec.max_common_voice_files
        ),
        "librispeech": find_audio_files_fast(spec.roots.librispeech, spec.max_librispeech_files),
        "background_noise": find_audio_files_fast(
            spec.roots.background_noise, spec.max_background_files
        ),
        "urbansound8k": find_audio_files_fast(spec.roots.urbansound8k, spec.max_urbansound_files),
    }
    limits = {
        "common_voice": spec.max_common_voice_files,
        "librispeech": spec.max_librispeech_files,
        "background_noise": spec.max_background_files,
        "urbansound8k": spec.max_urbansound_files,
    }
    for source, files in files_by_source.items():
        rng.shuffle(files)
        files_by_source[source] = files[: limits[source]]
    return files_by_source


def _make_rows_for_source(
    *,
    files: Sequence[Path],
    source: str,
    trials_per_file: int,
    profiles: Sequence[Mapping[str, Any]],
    duration_s: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in files:
        chosen_profiles = rng.sample(list(profiles), k=min(trials_per_file, len(profiles)))
        for profile in chosen_profiles:
            rows.append(
                {
                    "file_path": str(path),
                    "start_s": "0.0",
                    "duration_s": str(duration_s),
                    "speaker_id": f"external_{source}",
                    "keyword_text": "__external_fa__",
                    "source_dataset": source,
                    "quadrant_class": "EXTERNAL_FA",
                    "profile_id": profile["profile_id"],
                    "profile_path": profile["profile_path"],
                    "enrolled_user_id": profile["enrolled_speaker_id"],
                    "enrolled_keyword_text": profile["enrolled_keyword_text"],
                    "trial_source": "external_fa",
                }
            )
    return rows


def make_external_fa_rows(
    *,
    files_by_source: Mapping[str, Sequence[Path]],
    profiles: Sequence[Mapping[str, Any]],
    config: SoloSpeakConfig,
    spec: ExternalFaSpec = ExternalFaSpec(),
) -> list[dict[str, Any]]:
    if not profiles:
        raise ValueError("External FA rows require at least one enrolled profile.")
    rng = random.Random(spec.trial_seed)
    duration_s = float(config.audio.window_samples) / float(config.audio.sample_rate)
    rows: list[dict[str, Any]] = []
    rows += _make_rows_for_source(
        files=files_by_source.get("common_voice", []),
        source="common_voice",
        trials_per_file=spec.trials_per_common_voice_file,
        profiles=profiles,
        duration_s=duration_s,
        rng=rng,
    )
    rows += _make_rows_for_source(
        files=files_by_source.get("librispeech", []),
        source="librispeech",
        trials_per_file=spec.trials_per_librispeech_file,
        profiles=profiles,
        duration_s=duration_s,
        rng=rng,
    )
    rows += _make_rows_for_source(
        files=files_by_source.get("background_noise", []),
        source="background_noise",
        trials_per_file=spec.trials_per_background_file,
        profiles=profiles,
        duration_s=duration_s,
        rng=rng,
    )
    rows += _make_rows_for_source(
        files=files_by_source.get("urbansound8k", []),
        source="urbansound8k",
        trials_per_file=spec.trials_per_urbansound_file,
        profiles=profiles,
        duration_s=duration_s,
        rng=rng,
    )
    rng.shuffle(rows)
    return rows[: spec.max_total_external_trials]


def write_external_fa_manifest(
    rows: Sequence[Mapping[str, Any]],
    path: Path = Path("reports/stage7_external_fa_manifest.csv"),
) -> Path:
    fields = [
        "file_path",
        "start_s",
        "duration_s",
        "speaker_id",
        "keyword_text",
        "source_dataset",
        "quadrant_class",
        "profile_id",
        "profile_path",
        "enrolled_user_id",
        "enrolled_keyword_text",
        "trial_source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])
    return path


def read_external_fa_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def load_external_wav(row: Mapping[str, Any], config: SoloSpeakConfig) -> np.ndarray:
    fp = str(row["file_path"])
    if not Path(fp).exists():
        raise FileNotFoundError(fp)
    default_duration = float(config.audio.window_samples) / float(config.audio.sample_rate)
    start_s = float(row.get("start_s") or 0.0)
    duration = float(row.get("duration_s") or default_duration)
    wav = load_audio_segment(fp, start_s, duration, config.audio.sample_rate)
    wav = pad_or_crop_to_window(wav, config.audio.window_samples)
    return np.clip(wav, -1.0, 1.0).astype(np.float32)


@torch.no_grad()
def score_external_fa_rows(
    *,
    model: torch.nn.Module,
    extractor: LogMelExtractor,
    config: SoloSpeakConfig,
    rows: Sequence[Mapping[str, Any]],
    tau: float,
    output_path: Path = Path("reports/stage7_external_fa_scores.csv"),
    device: torch.device | None = None,
    batch_size: int = SCORE_BATCH_SIZE,
) -> dict[str, Any]:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    extractor.eval()

    profile_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    good_rows: list[Mapping[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    all_probs: list[np.ndarray] = []
    all_sc: list[np.ndarray] = []
    all_ss: list[np.ndarray] = []

    for start in range(0, len(rows), batch_size):
        raw_batch = rows[start : start + batch_size]
        batch_rows: list[Mapping[str, Any]] = []
        wavs: list[np.ndarray] = []
        for row in raw_batch:
            try:
                wavs.append(load_external_wav(row, config))
                batch_rows.append(row)
            except Exception as exc:
                failed = dict(row)
                failed["error"] = repr(exc)
                failed_rows.append(failed)
        if not batch_rows:
            continue

        wav = torch.from_numpy(np.stack(wavs, axis=0)).float().to(device)
        mel = make_mel({"wav": wav}, extractor).float()
        z_c, z_s = model(mel)
        z_c = F.normalize(z_c.float(), p=2, dim=-1)
        z_s = F.normalize(z_s.float(), p=2, dim=-1)

        ct_list: list[np.ndarray] = []
        st_list: list[np.ndarray] = []
        for row in batch_rows:
            profile_path = str(row["profile_path"])
            if profile_path not in profile_cache:
                profile_cache[profile_path] = load_profile_npz(profile_path)
            ct, st = profile_cache[profile_path]
            ct_list.append(ct)
            st_list.append(st)

        ct = F.normalize(torch.from_numpy(np.stack(ct_list)).float().to(device), p=2, dim=-1)
        st = F.normalize(torch.from_numpy(np.stack(st_list)).float().to(device), p=2, dim=-1)
        s_c = (z_c * ct).sum(dim=-1)
        s_s = (z_s * st).sum(dim=-1)
        probs = model.forward_fusion(s_c, s_s).view(-1)

        all_probs.append(probs.detach().cpu().numpy())
        all_sc.append(s_c.detach().cpu().numpy())
        all_ss.append(s_s.detach().cpu().numpy())
        good_rows.extend(batch_rows)

    probs = np.concatenate(all_probs) if all_probs else np.array([], dtype=np.float32)
    sc = np.concatenate(all_sc) if all_sc else np.array([], dtype=np.float32)
    ss = np.concatenate(all_ss) if all_ss else np.array([], dtype=np.float32)
    accept = probs >= tau

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "file_path",
        "source_dataset",
        "profile_id",
        "enrolled_user_id",
        "prob",
        "s_c",
        "s_s",
        "accepted",
        "tau",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row, prob, content_score, speaker_score, accepted in zip(
            good_rows, probs, sc, ss, accept, strict=False
        ):
            writer.writerow(
                {
                    "file_path": row["file_path"],
                    "source_dataset": row["source_dataset"],
                    "profile_id": row["profile_id"],
                    "enrolled_user_id": row["enrolled_user_id"],
                    "prob": float(prob),
                    "s_c": float(content_score),
                    "s_s": float(speaker_score),
                    "accepted": bool(accepted),
                    "tau": float(tau),
                }
            )

    summary = summarize_external_fa_arrays(
        rows=good_rows,
        probs=probs,
        accept=accept,
        duration_s=float(config.audio.window_samples) / float(config.audio.sample_rate),
    )
    summary["score_file"] = str(output_path)
    summary["failed_rows"] = int(len(failed_rows))
    summary["source_counts"] = dict(Counter(str(r["source_dataset"]) for r in good_rows))
    return summary


def summarize_external_fa_arrays(
    *,
    rows: Sequence[Mapping[str, Any]],
    probs: np.ndarray,
    accept: np.ndarray,
    duration_s: float,
) -> dict[str, Any]:
    def summarize_source(source_name: str) -> dict[str, Any]:
        mask = np.array([row["source_dataset"] == source_name for row in rows], dtype=bool)
        n = int(mask.sum())
        if n == 0:
            return {
                "rows": 0,
                "false_accepts": 0,
                "fa_rate": 0.0,
                "fa_hours": 0.0,
                "fa_per_hour_trial_level": 0.0,
                "fa_per_hour_per_profile": 0.0,
                "unique_profiles": 0,
                "mean_prob": 0.0,
                "p95_prob": 0.0,
                "p99_prob": 0.0,
                "max_prob": 0.0,
            }
        source_probs = probs[mask]
        source_accept = accept[mask]
        source_rows = [row for row, keep in zip(rows, mask, strict=False) if keep]
        unique_profiles = len({row["profile_id"] for row in source_rows})
        total_hours = n * duration_s / 3600.0
        false_accepts = int(source_accept.sum())
        return {
            "rows": n,
            "false_accepts": false_accepts,
            "fa_rate": float(source_accept.mean()),
            "fa_hours": float(total_hours),
            "fa_per_hour_trial_level": float(false_accepts / max(total_hours, 1e-9)),
            "fa_per_hour_per_profile": float(
                false_accepts / max(total_hours, 1e-9) / max(unique_profiles, 1)
            ),
            "unique_profiles": int(unique_profiles),
            "mean_prob": float(np.mean(source_probs)),
            "p95_prob": float(np.percentile(source_probs, 95)),
            "p99_prob": float(np.percentile(source_probs, 99)),
            "max_prob": float(np.max(source_probs)),
        }

    sources = sorted({str(row["source_dataset"]) for row in rows})
    by_source = {source: summarize_source(source) for source in sources}
    overall_profiles = len({row["profile_id"] for row in rows})
    overall_hours = len(rows) * duration_s / 3600.0
    overall_false_accepts = int(accept.sum())
    overall = {
        "rows": int(len(rows)),
        "false_accepts": overall_false_accepts,
        "fa_rate": float(accept.mean()) if len(accept) else 0.0,
        "fa_hours": float(overall_hours),
        "fa_per_hour_trial_level": float(overall_false_accepts / max(overall_hours, 1e-9)),
        "fa_per_hour_per_profile": float(
            overall_false_accepts / max(overall_hours, 1e-9) / max(overall_profiles, 1)
        ),
        "unique_profiles": int(overall_profiles),
        "mean_prob": float(np.mean(probs)) if len(probs) else 0.0,
        "p95_prob": float(np.percentile(probs, 95)) if len(probs) else 0.0,
        "p99_prob": float(np.percentile(probs, 99)) if len(probs) else 0.0,
        "max_prob": float(np.max(probs)) if len(probs) else 0.0,
    }
    return {"overall": overall, "by_source": by_source}
