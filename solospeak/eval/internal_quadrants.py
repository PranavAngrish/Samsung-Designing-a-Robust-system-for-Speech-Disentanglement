"""GSC internal Q1/Q2/Q3/Q4 profile generation and scoring.

This module is the source-code form of the final Stage-7 verification cells in
``ta-test.ipynb``. It rebuilds enrolled GSC wake-word profiles from a
speaker-disjoint test split, creates the four internal quadrants, and scores
them with the fusion head.
"""

from __future__ import annotations

import csv
import hashlib
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from solospeak.data.features import LogMelExtractor
from solospeak.training.stages.common import make_mel
from solospeak.utils.audio import load_audio_segment, pad_or_crop_to_window
from solospeak.utils.config import SoloSpeakConfig


WAKE_WORD = "zero"
N_ENROLL = 3
MAX_PROFILES = 300
PROFILE_SEED = 777
SPLIT_SEED = 505
Q1_PER_PROFILE = 2
Q2_PER_PROFILE = 6
Q3_PER_PROFILE = 2
Q4_PER_PROFILE = 2
EMBED_BATCH_SIZE = 128


@dataclass(frozen=True)
class InternalQuadrantSpec:
    """Notebook-authoritative internal verification settings."""

    wake_word: str = WAKE_WORD
    n_enroll: int = N_ENROLL
    max_profiles: int = MAX_PROFILES
    profile_seed: int = PROFILE_SEED
    split_seed: int = SPLIT_SEED
    q1_per_profile: int = Q1_PER_PROFILE
    q2_per_profile: int = Q2_PER_PROFILE
    q3_per_profile: int = Q3_PER_PROFILE
    q4_per_profile: int = Q4_PER_PROFILE
    require_wrong_word_for_profile: bool = True
    profile_prefix: str = "stage7_verify_profile"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def is_gsc_row(row: dict[str, Any]) -> bool:
    src = str(row.get("source_dataset", "")).lower()
    fp = str(row.get("file_path", "")).lower()
    spk = str(row.get("speaker_id", ""))
    kw = str(row.get("keyword_text", ""))
    if not fp or not spk or not kw:
        return False
    return (
        "gsc" in src
        or "speech_commands" in src
        or "speech-commands" in src
        or "speech-commands-v2" in fp
        or spk.startswith("gsc_")
    )


def stable_hash_int(text: str, seed: int = 42) -> int:
    return int(hashlib.sha256(f"{seed}:{text}".encode()).hexdigest(), 16)


def split_by_speaker(speaker_id: str, seed: int = SPLIT_SEED) -> str:
    h = stable_hash_int(speaker_id, seed) % 10000
    if h < 7000:
        return "train"
    if h < 8500:
        return "dev"
    return "test"


def norm_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    n = float(np.linalg.norm(x))
    return x / max(n, 1e-8)


def save_profile_npz(
    profile_dir: Path,
    profile_id: str,
    content_template: np.ndarray,
    speaker_template: np.ndarray,
) -> str:
    profile_dir.mkdir(parents=True, exist_ok=True)
    path = profile_dir / f"{profile_id}.npz"
    np.savez_compressed(
        path,
        content_template=norm_np(content_template),
        speaker_template=norm_np(speaker_template),
    )
    return str(path)


def load_profile_npz(profile_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(profile_path, allow_pickle=False) as arr:
        content = arr["content_template"].astype(np.float32)
        speaker = arr["speaker_template"].astype(np.float32)
    return norm_np(content), norm_np(speaker)


def load_fixed_wav(row: dict[str, Any], config: SoloSpeakConfig) -> np.ndarray:
    fp = str(row.get("file_path", ""))
    if not fp or not Path(fp).exists():
        raise FileNotFoundError(f"Missing audio file: {fp}")

    default_duration = float(config.audio.window_samples) / float(config.audio.sample_rate)
    start_s = float(row.get("start_s") or 0.0)
    if row.get("end_s"):
        end_s = float(row.get("end_s") or default_duration)
        duration = max(0.01, end_s - start_s)
    elif row.get("duration_s"):
        duration = max(0.01, float(row.get("duration_s") or default_duration))
    else:
        duration = default_duration

    wav = load_audio_segment(fp, start_s, duration, config.audio.sample_rate)
    wav = pad_or_crop_to_window(wav, config.audio.window_samples)
    return np.clip(wav, -1.0, 1.0).astype(np.float32)


def _with_progress(items: Iterable[int], desc: str | None) -> Iterable[int]:
    if desc is None:
        return items
    try:
        from tqdm.auto import tqdm

        return tqdm(items, desc=desc)
    except Exception:
        return items


@torch.no_grad()
def embed_rows(
    *,
    model: torch.nn.Module,
    extractor: LogMelExtractor,
    rows: Sequence[dict[str, Any]],
    config: SoloSpeakConfig,
    device: torch.device,
    batch_size: int = EMBED_BATCH_SIZE,
    desc: str | None = "embedding GSC rows",
) -> tuple[torch.Tensor, torch.Tensor]:
    all_zc: list[torch.Tensor] = []
    all_zs: list[torch.Tensor] = []
    model.eval()
    extractor.eval()

    starts = range(0, len(rows), batch_size)
    for start in _with_progress(starts, desc):
        batch_rows = rows[start : start + batch_size]
        wavs = [load_fixed_wav(r, config) for r in batch_rows]
        wav = torch.from_numpy(np.stack(wavs, axis=0)).float().to(device)
        mel = make_mel({"wav": wav}, extractor).float()
        z_c, z_s = model(mel)
        z_c = F.normalize(z_c.float(), p=2, dim=-1)
        z_s = F.normalize(z_s.float(), p=2, dim=-1)
        all_zc.append(z_c.cpu())
        all_zs.append(z_s.cpu())

    if not all_zc:
        raise ValueError("No rows were embedded.")
    return torch.cat(all_zc, dim=0), torch.cat(all_zs, dim=0)


def collect_gsc_test_rows(
    manifests_dir: Path,
    *,
    spec: InternalQuadrantSpec = InternalQuadrantSpec(),
    manifest_names: Sequence[str] = ("train_content.csv", "dev_content.csv"),
    check_missing_sample: bool = True,
) -> list[dict[str, str]]:
    all_rows: list[dict[str, str]] = []
    for name in manifest_names:
        path = manifests_dir / name
        if path.exists():
            all_rows.extend(read_csv_rows(path))

    all_gsc = [r for r in all_rows if is_gsc_row(r)]
    dedup: dict[str, dict[str, str]] = {}
    for row in all_gsc:
        fp = row.get("file_path", "")
        if fp:
            dedup[fp] = row
    all_gsc = list(dedup.values())

    if check_missing_sample and all_gsc:
        sample_n = min(1000, len(all_gsc))
        sample = random.Random(42).sample(all_gsc, sample_n)
        missing_sample = [r["file_path"] for r in sample if not Path(r["file_path"]).exists()]
        if missing_sample:
            joined = "\n".join(f"  {p}" for p in missing_sample[:10])
            raise FileNotFoundError(
                "Raw GSC dataset paths are missing. Example missing paths:\n" + joined
            )

    for row in all_gsc:
        row["_stage5_split"] = split_by_speaker(row["speaker_id"], spec.split_seed)
    return [r for r in all_gsc if r["_stage5_split"] == "test"]


def _build_indices(
    rows: Sequence[dict[str, Any]],
) -> tuple[
    dict[tuple[str, str], list[int]],
    dict[str, list[int]],
    dict[str, list[int]],
]:
    by_spk_kw: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_spk: dict[str, list[int]] = defaultdict(list)
    by_kw: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        spk = str(row["speaker_id"])
        kw = str(row["keyword_text"])
        by_spk_kw[(spk, kw)].append(idx)
        by_spk[spk].append(idx)
        by_kw[kw].append(idx)
    return by_spk_kw, by_spk, by_kw


def _candidate_speakers(
    rows: Sequence[dict[str, Any]],
    by_spk_kw: dict[tuple[str, str], list[int]],
    by_spk: dict[str, list[int]],
    spec: InternalQuadrantSpec,
) -> list[str]:
    candidates: list[str] = []
    for (spk, kw), idxs in by_spk_kw.items():
        if kw != spec.wake_word or len(idxs) < spec.n_enroll + 1:
            continue
        if spec.require_wrong_word_for_profile:
            same_spk_wrong_word = [
                i for i in by_spk[spk] if rows[i]["keyword_text"] != spec.wake_word
            ]
            if not same_spk_wrong_word:
                continue
        candidates.append(spk)

    candidates = sorted(set(candidates))
    rng = random.Random(spec.profile_seed)
    rng.shuffle(candidates)
    return candidates[: spec.max_profiles]


def build_gsc_internal_examples(
    *,
    model: torch.nn.Module,
    extractor: LogMelExtractor,
    config: SoloSpeakConfig,
    manifests_dir: Path | None = None,
    profile_dir: Path = Path("data/stage7_verify_profiles"),
    spec: InternalQuadrantSpec = InternalQuadrantSpec(),
    device: torch.device | None = None,
    batch_size: int = EMBED_BATCH_SIZE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build enrolled profiles and Q1/Q2/Q3/Q4 examples from GSC test speakers."""

    if device is None:
        device = next(model.parameters()).device
    manifests_dir = manifests_dir or config.data.manifests_dir
    rows = collect_gsc_test_rows(manifests_dir, spec=spec)
    if not rows:
        raise ValueError("No GSC test rows were found in train_content/dev_content manifests.")

    by_spk_kw, by_spk, by_kw = _build_indices(rows)
    candidate_speakers = _candidate_speakers(rows, by_spk_kw, by_spk, spec)
    if not candidate_speakers:
        raise ValueError(f"No candidate GSC speakers found for wake word {spec.wake_word!r}.")

    zc_all, zs_all = embed_rows(
        model=model,
        extractor=extractor,
        rows=rows,
        config=config,
        device=device,
        batch_size=batch_size,
        desc="embedding GSC test rows",
    )

    rng = random.Random(spec.profile_seed)
    profiles: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    def add_trial(
        idx: int,
        profile_id: str,
        profile_path: str,
        enrolled_spk: str,
        quadrant: str,
        label: int,
        ct_t: torch.Tensor,
        st_t: torch.Tensor,
    ) -> None:
        row = rows[idx]
        examples.append(
            {
                "s_c": float(torch.dot(zc_all[idx], ct_t)),
                "s_s": float(torch.dot(zs_all[idx], st_t)),
                "label": float(label),
                "quadrant": quadrant,
                "profile_id": profile_id,
                "profile_path": profile_path,
                "speaker_id": row.get("speaker_id", ""),
                "keyword_text": row.get("keyword_text", ""),
                "file_path": row.get("file_path", ""),
                "source_dataset": row.get("source_dataset", "gsc_internal"),
                "enrolled_user_id": enrolled_spk,
                "enrolled_keyword_text": spec.wake_word,
            }
        )

    for spk in candidate_speakers:
        wake_idxs = list(by_spk_kw[(spk, spec.wake_word)])
        rng.shuffle(wake_idxs)
        enroll_idxs = wake_idxs[: spec.n_enroll]
        q1_pool = wake_idxs[spec.n_enroll :]
        if not q1_pool:
            continue

        ct = (
            F.normalize(zc_all[enroll_idxs].mean(dim=0, keepdim=True), p=2, dim=-1)
            .squeeze(0)
            .numpy()
        )
        st = (
            F.normalize(zs_all[enroll_idxs].mean(dim=0, keepdim=True), p=2, dim=-1)
            .squeeze(0)
            .numpy()
        )
        profile_id = (
            f"{spec.profile_prefix}_{spec.wake_word}_"
            f"{hashlib.sha1(spk.encode()).hexdigest()[:16]}"
        )
        profile_path = save_profile_npz(profile_dir, profile_id, ct, st)
        profiles.append(
            {
                "profile_id": profile_id,
                "profile_path": profile_path,
                "enrolled_speaker_id": spk,
                "enrolled_keyword_text": spec.wake_word,
                "n_enroll": spec.n_enroll,
            }
        )

        ct_t = torch.from_numpy(norm_np(ct))
        st_t = torch.from_numpy(norm_np(st))

        rng.shuffle(q1_pool)
        for idx in q1_pool[: spec.q1_per_profile]:
            add_trial(idx, profile_id, profile_path, spk, "Q1_accept", 1, ct_t, st_t)

        q2_pool = [i for i in by_kw[spec.wake_word] if rows[i]["speaker_id"] != spk]
        rng.shuffle(q2_pool)
        for idx in q2_pool[: spec.q2_per_profile]:
            add_trial(idx, profile_id, profile_path, spk, "Q2_imposter", 0, ct_t, st_t)

        q3_pool = [i for i in by_spk[spk] if rows[i]["keyword_text"] != spec.wake_word]
        rng.shuffle(q3_pool)
        for idx in q3_pool[: spec.q3_per_profile]:
            add_trial(idx, profile_id, profile_path, spk, "Q3_wrong_word", 0, ct_t, st_t)

        q4_pool = [
            i
            for i, row in enumerate(rows)
            if row["speaker_id"] != spk and row["keyword_text"] != spec.wake_word
        ]
        rng.shuffle(q4_pool)
        for idx in q4_pool[: spec.q4_per_profile]:
            add_trial(idx, profile_id, profile_path, spk, "Q4_background", 0, ct_t, st_t)

    if not profiles or not examples:
        raise ValueError("Internal profile/example build produced no rows.")
    return profiles, examples


@torch.no_grad()
def fusion_probs_from_examples(
    model: torch.nn.Module,
    examples: Sequence[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    all_probs: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        x = torch.tensor(
            [[float(e["s_c"]), float(e["s_s"])] for e in batch],
            dtype=torch.float32,
            device=device,
        )
        probs = model.forward_fusion(x[:, 0], x[:, 1]).view(-1)
        all_probs.append(probs.detach().cpu().numpy())
    return np.concatenate(all_probs, axis=0) if all_probs else np.array([], dtype=np.float32)


def internal_metrics_from_probs(
    examples: Sequence[dict[str, Any]],
    probs: np.ndarray,
    *,
    tau: float,
) -> dict[str, Any]:
    accept = probs >= tau
    quadrants = np.array([e["quadrant"] for e in examples], dtype=object)
    q1 = quadrants == "Q1_accept"
    q2 = quadrants == "Q2_imposter"
    q3 = quadrants == "Q3_wrong_word"
    q4 = quadrants == "Q4_background"

    def frac(mask: np.ndarray, values: np.ndarray) -> float:
        denom = int(mask.sum())
        if denom <= 0:
            return 0.0
        return float(values[mask].mean())

    metrics = {
        "tau": float(tau),
        "n_rows": int(len(examples)),
        "n_q1": int(q1.sum()),
        "n_q2": int(q2.sum()),
        "n_q3": int(q3.sum()),
        "n_q4": int(q4.sum()),
        "ta_clean": frac(q1, accept.astype(np.float32)),
        "q2_rejection": frac(q2, (~accept).astype(np.float32)),
        "q3_rejection": frac(q3, (~accept).astype(np.float32)),
        "q4_rejection": frac(q4, (~accept).astype(np.float32)),
        "accepted_rate": float(accept.mean()) if len(accept) else 0.0,
        "mean_prob": float(np.mean(probs)) if len(probs) else 0.0,
        "p95_prob": float(np.percentile(probs, 95)) if len(probs) else 0.0,
        "p99_prob": float(np.percentile(probs, 99)) if len(probs) else 0.0,
        "max_prob": float(np.max(probs)) if len(probs) else 0.0,
        "counts": dict(Counter(quadrants.tolist())),
    }
    vals = [
        metrics["ta_clean"],
        metrics["q2_rejection"],
        metrics["q3_rejection"],
        metrics["q4_rejection"],
    ]
    metrics["quadrant_accuracy_min"] = float(min(vals))
    metrics["quadrant_accuracy_mean"] = float(sum(vals) / len(vals))
    return metrics


def write_internal_scores_csv(
    path: Path,
    examples: Sequence[dict[str, Any]],
    probs: np.ndarray,
    *,
    tau: float,
) -> Path:
    accept = probs >= tau
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "quadrant",
        "label",
        "s_c",
        "s_s",
        "prob",
        "accepted",
        "tau",
        "profile_id",
        "speaker_id",
        "keyword_text",
        "file_path",
        "source_dataset",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for example, prob, accepted in zip(examples, probs, accept, strict=False):
            writer.writerow(
                {
                    "quadrant": example.get("quadrant", ""),
                    "label": example.get("label", ""),
                    "s_c": float(example.get("s_c", 0.0)),
                    "s_s": float(example.get("s_s", 0.0)),
                    "prob": float(prob),
                    "accepted": bool(accepted),
                    "tau": float(tau),
                    "profile_id": example.get("profile_id", ""),
                    "speaker_id": example.get("speaker_id", ""),
                    "keyword_text": example.get("keyword_text", ""),
                    "file_path": example.get("file_path", ""),
                    "source_dataset": example.get("source_dataset", ""),
                }
            )
    return path


def score_internal_examples_to_csv(
    *,
    model: torch.nn.Module,
    examples: Sequence[dict[str, Any]],
    tau: float,
    output_path: Path = Path("reports/stage7_final_internal_scores.csv"),
    device: torch.device | None = None,
) -> dict[str, Any]:
    if device is None:
        device = next(model.parameters()).device
    probs = fusion_probs_from_examples(model, examples, device=device)
    write_internal_scores_csv(output_path, examples, probs, tau=tau)
    metrics = internal_metrics_from_probs(examples, probs, tau=tau)
    metrics["score_file"] = str(output_path)
    return metrics
