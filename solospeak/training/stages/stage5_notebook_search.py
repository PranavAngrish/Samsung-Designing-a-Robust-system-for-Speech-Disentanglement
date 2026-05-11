"""Notebook-authoritative Stage-5 fusion search.

This module ports the final ``clean-the-mess.ipynb`` Stage-5 cell into source
code. It searches GSC wake-word profile data variants and fusion-head training
variants, then writes the recoverable and strict Stage-5 checkpoints.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from solospeak.data.datasets import MANIFEST_COLUMNS
from solospeak.data.features import LogMelExtractor
from solospeak.eval.internal_quadrants import (
    embed_rows,
    is_gsc_row,
    norm_np,
    read_csv_rows,
    save_profile_npz,
    split_by_speaker,
)
from solospeak.models.solospeak import SoloSpeakModel
from solospeak.training.stages.common import default_device, save_checkpoint
from solospeak.utils.config import SoloSpeakConfig
from solospeak.utils.types import MetricsDict


@dataclass(frozen=True)
class DataVariant:
    name: str
    wake_word: str
    n_enroll: int
    q1_per_profile: int
    q2_per_profile: int
    q3_per_profile: int
    q4_per_profile: int
    max_train_profiles: int | None = None
    max_dev_profiles: int | None = None
    max_test_profiles: int | None = None


@dataclass(frozen=True)
class FusionVariant:
    name: str
    lr: float
    epochs: int
    patience: int
    batch_size: int
    weight_decay: float
    q1_weight: float
    q2_weight: float
    q3_weight: float
    q4_weight: float
    pos_weight_scale: float


DATA_VARIANTS = [
    DataVariant("zero_e2_product", "zero", 2, 2, 4, 2, 2),
    DataVariant("zero_e2_q2heavy", "zero", 2, 2, 6, 2, 2),
    DataVariant("zero_e3_product", "zero", 3, 2, 4, 2, 2),
    DataVariant("yes_e2_check", "yes", 2, 2, 4, 2, 2),
]

FUSION_VARIANTS = [
    FusionVariant("balanced", 1e-3, 500, 80, 256, 1e-4, 1.4, 1.8, 0.4, 0.4, 1.0),
    FusionVariant("q2_strong", 8e-4, 600, 90, 256, 1e-4, 1.3, 2.8, 0.25, 0.25, 0.9),
    FusionVariant("q2_very_strong", 6e-4, 700, 100, 256, 1e-4, 1.2, 4.0, 0.2, 0.2, 0.8),
]

EXPECTED_STAGE5_BEST_SIGNATURE = {
    "stage4_candidate": "stage4d_hardq2_mining_balanced",
    "data_variant": "zero_e3_product",
    "fusion_variant": "q2_very_strong",
}
EXPECTED_STAGE5_WINNER_CONFIG = {
    "stage4_candidate": {
        "name": "stage4d_hardq2_mining_balanced",
        "checkpoint": "checkpoints/stage4d_hardq2_mining_balanced.pt",
    },
    "data_variant": {
        "name": "zero_e3_product",
        "wake_word": "zero",
        "n_enroll": 3,
        "q1_per_profile": 2,
        "q2_per_profile": 4,
        "q3_per_profile": 2,
        "q4_per_profile": 2,
    },
    "fusion_variant": {
        "name": "q2_very_strong",
        "lr": 0.0006,
        "epochs": 700,
        "patience": 100,
        "batch_size": 256,
        "weight_decay": 0.0001,
        "q1_weight": 1.2,
        "q2_weight": 4.0,
        "q3_weight": 0.2,
        "q4_weight": 0.2,
        "pos_weight_scale": 0.8,
    },
}


def stage5_result_signature(result: dict[str, Any]) -> dict[str, str]:
    """Return the three fields that define the winning notebook Stage-5 path."""

    data_variant = result.get("data_variant", {})
    fusion_variant = result.get("fusion_variant", {})
    if not isinstance(data_variant, dict):
        data_name = str(data_variant)
    else:
        data_name = str(data_variant.get("name", ""))
    if not isinstance(fusion_variant, dict):
        fusion_name = str(fusion_variant)
    else:
        fusion_name = str(fusion_variant.get("name", ""))
    return {
        "stage4_candidate": str(result.get("stage4_candidate", "")),
        "data_variant": data_name,
        "fusion_variant": fusion_name,
    }


def _build_index(
    rows: list[dict[str, Any]],
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


def _rows_by_split(manifests_dir: Path) -> dict[str, list[dict[str, str]]]:
    train_content = read_csv_rows(manifests_dir / "train_content.csv")
    dev_content = read_csv_rows(manifests_dir / "dev_content.csv")
    all_gsc = [r for r in train_content + dev_content if is_gsc_row(r)]
    dedup: dict[str, dict[str, str]] = {}
    for row in all_gsc:
        fp = str(row.get("file_path", ""))
        if fp:
            dedup[fp] = row
    all_gsc = list(dedup.values())
    if not all_gsc:
        raise ValueError("No GSC rows found in train_content/dev_content manifests.")
    for row in all_gsc:
        row["_stage5_split"] = split_by_speaker(row["speaker_id"])
    rows_by_split = {
        "train": [r for r in all_gsc if r["_stage5_split"] == "train"],
        "dev": [r for r in all_gsc if r["_stage5_split"] == "dev"],
        "test": [r for r in all_gsc if r["_stage5_split"] == "test"],
    }
    split_speakers = {name: {r["speaker_id"] for r in rows} for name, rows in rows_by_split.items()}
    if not split_speakers["train"].isdisjoint(split_speakers["dev"]):
        raise RuntimeError("Stage5 train/dev speaker split is not disjoint.")
    if not split_speakers["train"].isdisjoint(split_speakers["test"]):
        raise RuntimeError("Stage5 train/test speaker split is not disjoint.")
    if not split_speakers["dev"].isdisjoint(split_speakers["test"]):
        raise RuntimeError("Stage5 dev/test speaker split is not disjoint.")
    return rows_by_split


def extract_model_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model_state")
    if state is not None:
        return dict(state)
    wrapper_state = checkpoint.get("wrapper_state")
    if wrapper_state is None:
        raise ValueError("Stage4 checkpoint has no model_state or wrapper_state.")
    return {
        key.replace("model.", "", 1): value
        for key, value in wrapper_state.items()
        if key.startswith("model.")
    }


def _stage4_candidates(config: SoloSpeakConfig) -> list[dict[str, Any]]:
    if config.training.resume_from is not None:
        resume_path = config.training.resume_from
        resume_name = resume_path.name.lower()
        kind = "hardq2" if "hardq2" in resume_name or "stage4d" in resume_name else "stage4c"
        return [{"name": resume_path.stem, "kind": kind, "path": resume_path}]

    if config.stage5.reproduce_winner_only:
        return [
            {
                "name": config.stage5.stage4_candidate_name,
                "kind": "hardq2",
                "path": config.stage5.stage4_candidate_checkpoint,
            }
        ]

    candidates = [
        {
            "name": "stage4c_production",
            "kind": "stage4c",
            "path": (
                config.training.resume_from
                or config.training.checkpoint_dir / "stage4_robust.pt"
            ),
        }
    ]
    challenger = config.training.checkpoint_dir / "stage4d_hardq2_mining_balanced.pt"
    if challenger.exists():
        candidates.append({"name": challenger.stem, "kind": "hardq2", "path": challenger})
    return candidates


def verify_stage4_checkpoint(candidate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(candidate["path"])
    obj = torch.load(path, map_location="cpu", weights_only=False)
    metrics = obj.get("metrics", {}) or {}
    if obj.get("stage_origin") != 4:
        raise ValueError(f"{candidate['name']} is not stage_origin=4: {path}")
    if "model_state" not in obj and "wrapper_state" not in obj:
        raise ValueError(f"{candidate['name']} has no model_state/wrapper_state: {path}")
    if candidate["kind"] == "stage4c" and "dev/accepted_for_stage5" in metrics:
        if float(metrics.get("dev/accepted_for_stage5", 0.0)) < 1.0:
            raise ValueError(f"{candidate['name']} is not accepted_for_stage5.")
    elif candidate["kind"] == "hardq2":
        ta = float(metrics.get("dev/stage4d_hardq2_ta_clean", 0.0))
        q2 = float(metrics.get("dev/stage4d_hardq2_q2_rejection", 0.0))
        q3 = float(metrics.get("dev/stage4d_hardq2_q3_rejection", 0.0))
        qmin = float(metrics.get("dev/stage4d_hardq2_quadrant_min", 0.0))
        if ta < 0.90 or q2 < 0.90 or q3 < 0.88 or qmin < 0.88:
            raise ValueError(f"{candidate['name']} failed hard-Q2 challenger prechecks.")
    return obj, metrics


def _make_manifest_row(
    base: dict[str, Any],
    profile_id: str,
    profile_path: str,
    enrolled_spk: str,
    enrolled_kw: str,
    quadrant: str,
) -> dict[str, Any]:
    row = dict(base)
    row["profile_id"] = profile_id
    row["profile_path"] = profile_path
    row["enrolled_user_id"] = enrolled_spk
    row["enrolled_keyword_text"] = enrolled_kw
    row["quadrant_class"] = quadrant
    row["trial_source"] = "real"
    row["q3_gate_eligible"] = "true" if quadrant == "Q3_wrong_word" else "false"
    row["split"] = "test" if row.get("_stage5_split") == "test" else row.get("_stage5_split", "")
    for column in MANIFEST_COLUMNS:
        row.setdefault(column, "")
    return row


def build_quadrant_dataset(
    *,
    rows: list[dict[str, Any]],
    zc: torch.Tensor,
    zs: torch.Tensor,
    data_variant: DataVariant,
    split: str,
    profile_dir: Path,
    candidate_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    by_spk_kw, by_spk, by_kw = _build_index(rows)
    seed = 1000 + stable_candidate_seed(candidate_name + data_variant.name + split) % 100000
    rng = random.Random(seed)
    candidate_speakers: list[str] = []
    for (spk, kw), idxs in by_spk_kw.items():
        if kw != data_variant.wake_word:
            continue
        wrong_same_spk = [
            i for i in by_spk[spk] if rows[i]["keyword_text"] != data_variant.wake_word
        ]
        if len(idxs) >= data_variant.n_enroll + 1 and wrong_same_spk:
            candidate_speakers.append(spk)
    candidate_speakers = sorted(set(candidate_speakers))
    rng.shuffle(candidate_speakers)
    max_profiles = {
        "train": data_variant.max_train_profiles,
        "dev": data_variant.max_dev_profiles,
        "test": data_variant.max_test_profiles,
    }[split]
    if max_profiles is not None:
        candidate_speakers = candidate_speakers[:max_profiles]

    examples: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    profiles: dict[str, dict[str, Any]] = {}

    for spk in candidate_speakers:
        wake_idxs = list(by_spk_kw[(spk, data_variant.wake_word)])
        rng.shuffle(wake_idxs)
        enroll_idxs = wake_idxs[: data_variant.n_enroll]
        q1_pool = wake_idxs[data_variant.n_enroll :]
        if not q1_pool:
            continue

        ct = (
            F.normalize(zc[enroll_idxs].mean(dim=0, keepdim=True), p=2, dim=-1)
            .squeeze(0)
            .numpy()
        )
        st = (
            F.normalize(zs[enroll_idxs].mean(dim=0, keepdim=True), p=2, dim=-1)
            .squeeze(0)
            .numpy()
        )
        profile_id = (
            f"profile_{candidate_name}_{data_variant.name}_{split}_"
            f"{hashlib.sha1((spk + data_variant.wake_word).encode()).hexdigest()[:16]}"
        )
        profile_path = save_profile_npz(profile_dir, profile_id, ct, st)
        profiles[profile_id] = {
            "speaker_id": spk,
            "wake_word": data_variant.wake_word,
            "profile_path": profile_path,
            "enroll_idxs": enroll_idxs,
        }
        ct_t = torch.from_numpy(norm_np(ct))
        st_t = torch.from_numpy(norm_np(st))

        def add_trial(idx: int, quadrant: str, label: int) -> None:
            examples.append(
                {
                    "s_c": float(torch.dot(zc[idx], ct_t)),
                    "s_s": float(torch.dot(zs[idx], st_t)),
                    "label": float(label),
                    "quadrant": quadrant,
                    "profile_id": profile_id,
                    "row_index": idx,
                }
            )
            manifest_rows.append(
                _make_manifest_row(
                    rows[idx],
                    profile_id,
                    profile_path,
                    spk,
                    data_variant.wake_word,
                    quadrant,
                )
            )

        rng.shuffle(q1_pool)
        for idx in q1_pool[: data_variant.q1_per_profile]:
            add_trial(idx, "Q1_accept", 1)

        q2_pool = [i for i in by_kw[data_variant.wake_word] if rows[i]["speaker_id"] != spk]
        rng.shuffle(q2_pool)
        for idx in q2_pool[: data_variant.q2_per_profile]:
            add_trial(idx, "Q2_imposter", 0)

        q3_pool = [i for i in by_spk[spk] if rows[i]["keyword_text"] != data_variant.wake_word]
        rng.shuffle(q3_pool)
        for idx in q3_pool[: data_variant.q3_per_profile]:
            add_trial(idx, "Q3_wrong_word", 0)

        q4_pool = [
            i
            for i, row in enumerate(rows)
            if row["speaker_id"] != spk and row["keyword_text"] != data_variant.wake_word
        ]
        rng.shuffle(q4_pool)
        for idx in q4_pool[: data_variant.q4_per_profile]:
            add_trial(idx, "Q4_background", 0)

    return examples, manifest_rows, profiles


def stable_candidate_seed(text: str) -> int:
    return int(hashlib.sha256(f"17:{text}".encode()).hexdigest(), 16)


def _tensorize_examples(
    examples: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    x = torch.tensor([[e["s_c"], e["s_s"]] for e in examples], dtype=torch.float32)
    y = torch.tensor([e["label"] for e in examples], dtype=torch.float32)
    q = np.array([e["quadrant"] for e in examples], dtype=object)
    return x, y, q


def _fusion_forward_from_x(
    model: SoloSpeakModel,
    x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return model.forward_fusion(x[:, 0].to(device), x[:, 1].to(device)).view(-1)


@torch.no_grad()
def eval_fusion(
    model: SoloSpeakModel,
    examples: list[dict[str, Any]],
    *,
    device: torch.device,
    tau: float | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    x, _, q = _tensorize_examples(examples)
    probs = []
    for start in range(0, len(x), 2048):
        p = _fusion_forward_from_x(model, x[start : start + 2048], device)
        probs.append(p.detach().cpu())
    prob_np = torch.cat(probs).numpy()

    def compute_at_tau(threshold: float) -> dict[str, float]:
        accept = prob_np >= threshold
        q1 = q == "Q1_accept"
        q2 = q == "Q2_imposter"
        q3 = q == "Q3_wrong_word"
        q4 = q == "Q4_background"

        def frac(mask: np.ndarray, values: np.ndarray) -> float:
            denom = int(mask.sum())
            if denom <= 0:
                return 0.0
            return float(values[mask].mean())

        ta = frac(q1, accept.astype(np.float32))
        q2r = frac(q2, (~accept).astype(np.float32))
        q3r = frac(q3, (~accept).astype(np.float32))
        q4r = frac(q4, (~accept).astype(np.float32))
        q1q2_min = min(ta, q2r)
        quad_min = min(ta, q2r, q3r, q4r)
        quad_mean = (ta + q2r + q3r + q4r) / 4.0
        return {
            "tau": float(threshold),
            "ta_clean": float(ta),
            "q2_rejection": float(q2r),
            "q3_rejection": float(q3r),
            "q4_rejection": float(q4r),
            "quadrant_accuracy_min": float(quad_min),
            "quadrant_accuracy_mean": float(quad_mean),
            "q1q2_min": float(q1q2_min),
            "accepted_rate": float(accept.mean()),
            "selection_score": float(6.0 * q1q2_min + 2.0 * quad_min + quad_mean),
            "mean_prob": float(prob_np.mean()),
        }

    if tau is None:
        best: dict[str, float] | None = None
        for threshold in np.linspace(0.05, 0.95, 181):
            candidate = compute_at_tau(float(threshold))
            if best is None or candidate["selection_score"] > best["selection_score"]:
                best = candidate
        if best is None:
            raise ValueError("No Stage5 threshold candidates evaluated.")
        return best, prob_np
    return compute_at_tau(float(tau)), prob_np


def _model_from_state(
    config: SoloSpeakConfig,
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[SoloSpeakModel, list[torch.nn.Parameter]]:
    model = SoloSpeakModel(config).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected Stage4 state keys: {unexpected[:20]}")
    if missing:
        raise RuntimeError(f"Missing Stage4 state keys: {missing[:20]}")
    for param in model.parameters():
        param.requires_grad = False
    params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if "fusion" in name.lower():
            param.requires_grad = True
            params.append(param)
    if not params:
        raise RuntimeError("No fusion parameters found for Stage5.")
    return model, params


def train_fusion_variant(
    *,
    config: SoloSpeakConfig,
    device: torch.device,
    source_state: dict[str, torch.Tensor],
    source_stage4_path: Path,
    candidate_name: str,
    data_variant: DataVariant,
    fusion_variant: FusionVariant,
    train_examples: list[dict[str, Any]],
    dev_examples: list[dict[str, Any]],
    test_examples: list[dict[str, Any]],
) -> dict[str, Any]:
    model, params = _model_from_state(config, source_state, device)
    optimizer = torch.optim.AdamW(
        params,
        lr=fusion_variant.lr,
        weight_decay=fusion_variant.weight_decay,
    )
    x_train, y_train, _ = _tensorize_examples(train_examples)
    weights = np.ones(len(train_examples), dtype=np.float32)
    for idx, example in enumerate(train_examples):
        if example["quadrant"] == "Q1_accept":
            weights[idx] = fusion_variant.q1_weight
        elif example["quadrant"] == "Q2_imposter":
            weights[idx] = fusion_variant.q2_weight
        elif example["quadrant"] == "Q3_wrong_word":
            weights[idx] = fusion_variant.q3_weight
        elif example["quadrant"] == "Q4_background":
            weights[idx] = fusion_variant.q4_weight
    weights_t = torch.tensor(weights, dtype=torch.float32)
    pos_count = float((y_train == 1).sum().item())
    neg_count = float((y_train == 0).sum().item())
    pos_weight = torch.tensor(
        [max(1e-6, neg_count / max(pos_count, 1.0)) * fusion_variant.pos_weight_scale],
        dtype=torch.float32,
        device=device,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_tau: float | None = None
    best_epoch = 0
    best_score = -1e9
    no_improve = 0
    indices = np.arange(len(train_examples))

    for epoch in range(1, fusion_variant.epochs + 1):
        model.train()
        np.random.shuffle(indices)
        for start in range(0, len(indices), fusion_variant.batch_size):
            idx = indices[start : start + fusion_variant.batch_size]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            wb = weights_t[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            probs = _fusion_forward_from_x(model, xb, device).clamp(1e-6, 1.0 - 1e-6)
            logits = torch.logit(probs)
            loss_raw = F.binary_cross_entropy_with_logits(
                logits,
                yb,
                pos_weight=pos_weight,
                reduction="none",
            )
            loss = (loss_raw * wb).mean()
            loss.backward()
            optimizer.step()

        dev_metrics, _ = eval_fusion(model, dev_examples, device=device)
        score = (
            10.0 * dev_metrics["q1q2_min"]
            + 3.0 * dev_metrics["quadrant_accuracy_min"]
            + dev_metrics["quadrant_accuracy_mean"]
            + 2.0 * dev_metrics["q2_rejection"]
            + dev_metrics["ta_clean"]
        )
        if score > best_score:
            best_score = score
            best_tau = dev_metrics["tau"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= fusion_variant.patience:
            break

    if best_state is None or best_tau is None:
        raise RuntimeError("Stage5 fusion variant did not produce a best state.")
    model.load_state_dict(best_state, strict=False)
    dev_metrics, _ = eval_fusion(model, dev_examples, device=device, tau=best_tau)
    test_metrics, _ = eval_fusion(model, test_examples, device=device, tau=best_tau)
    gate_passed = (
        dev_metrics["quadrant_accuracy_min"] >= 0.85
        and dev_metrics["q2_rejection"] >= 0.85
        and dev_metrics["ta_clean"] >= 0.85
    )
    target_passed = dev_metrics["quadrant_accuracy_min"] >= 0.95
    selection_score = (
        1000.0 * float(gate_passed)
        + 100.0 * float(target_passed)
        + 10.0 * dev_metrics["quadrant_accuracy_min"]
        + 5.0 * dev_metrics["q1q2_min"]
        + 2.0 * test_metrics["quadrant_accuracy_min"]
        + 2.0 * test_metrics["q2_rejection"]
        + test_metrics["ta_clean"]
    )
    return {
        "stage4_candidate": candidate_name,
        "source_stage4_checkpoint": str(source_stage4_path),
        "data_variant": asdict(data_variant),
        "fusion_variant": asdict(fusion_variant),
        "best_epoch": int(best_epoch),
        "best_tau": float(best_tau),
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "gate_passed": bool(gate_passed),
        "target_passed": bool(target_passed),
        "selection_score": float(selection_score),
        "model": model,
    }


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    preferred = [
        "file_path",
        "start_s",
        "end_s",
        "duration_s",
        "speaker_id",
        "speaker_label",
        "keyword_text",
        "keyword_label",
        "source_dataset",
        "split",
        "quadrant_class",
        "profile_id",
        "profile_path",
        "enrolled_user_id",
        "enrolled_keyword_text",
        "trial_source",
        "q3_gate_eligible",
    ]
    fieldnames = [k for k in preferred if k in all_keys] + sorted(
        all_keys - set(preferred) - {"_stage5_split"}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _copy_manifest_profiles(
    rows: list[dict[str, Any]],
    profile_dir: Path,
) -> list[dict[str, Any]]:
    profile_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        profile_path = str(row.get("profile_path", ""))
        if profile_path:
            source = Path(profile_path)
            if source.exists():
                target = profile_dir / source.name
                if profile_path not in copied:
                    shutil.copyfile(source, target)
                    copied[profile_path] = str(target)
                new_row["profile_path"] = copied[profile_path]
        out.append(new_row)
    return out


def _metrics_from_result(result: dict[str, Any]) -> MetricsDict:
    dev = result["dev_metrics"]
    test = result["test_metrics"]
    tau = float(result["best_tau"])
    return {
        "dev/best_epoch": float(result["best_epoch"]),
        "dev/tau_stage5": tau,
        "dev/selection_score": float(result["selection_score"]),
        "dev/gate_min_threshold": 0.85,
        "dev/gate_target_threshold": 0.95,
        "dev/gate_passed": float(result["gate_passed"]),
        "dev/target_passed": float(result["target_passed"]),
        "dev/ta_clean": float(dev["ta_clean"]),
        "dev/q2_rejection": float(dev["q2_rejection"]),
        "dev/q3_rejection": float(dev["q3_rejection"]),
        "dev/q4_rejection": float(dev["q4_rejection"]),
        "dev/accepted_rate": float(dev["accepted_rate"]),
        "dev/tau": float(dev["tau"]),
        "dev/quadrant_accuracy_min": float(dev["quadrant_accuracy_min"]),
        "dev/quadrant_accuracy_mean": float(dev["quadrant_accuracy_mean"]),
        "dev/q1q2_min": float(dev["q1q2_min"]),
        "test/ta_clean": float(test["ta_clean"]),
        "test/q2_rejection": float(test["q2_rejection"]),
        "test/q3_rejection": float(test["q3_rejection"]),
        "test/q4_rejection": float(test["q4_rejection"]),
        "test/accepted_rate": float(test["accepted_rate"]),
        "test/tau": float(test["tau"]),
        "test/quadrant_accuracy_min": float(test["quadrant_accuracy_min"]),
        "test/quadrant_accuracy_mean": float(test["quadrant_accuracy_mean"]),
        "test/q1q2_min": float(test["q1q2_min"]),
    }


def run_stage5_notebook_search(config: SoloSpeakConfig) -> Path:
    device = default_device()
    extractor = LogMelExtractor(config.audio).to(device)
    checkpoint_dir = config.training.checkpoint_dir
    reports_dir = Path("reports")
    stage5_dir = config.data.root / "stage5"
    profile_root = stage5_dir / "profiles"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    profile_root.mkdir(parents=True, exist_ok=True)

    rows_by_split = _rows_by_split(config.data.manifests_dir)
    all_results: list[dict[str, Any]] = []
    built_by_checkpoint: dict[str, dict[str, Any]] = {}

    for candidate in _stage4_candidates(config):
        candidate_name = str(candidate["name"])
        candidate_path = Path(candidate["path"])
        stage4_obj, _ = verify_stage4_checkpoint(candidate)
        source_state = extract_model_state(stage4_obj)
        config.training.resume_from = candidate_path

        base_model = SoloSpeakModel(config).to(device).eval()
        missing, unexpected = base_model.load_state_dict(source_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Bad Stage4 state for Stage5: missing={missing[:20]}, unexpected={unexpected[:20]}"
            )

        split_embeddings = {}
        for split in ["train", "dev", "test"]:
            zc, zs = embed_rows(
                model=base_model,
                extractor=extractor,
                rows=rows_by_split[split],
                config=config,
                device=device,
                desc=f"{candidate_name}: embedding {split}",
            )
            split_embeddings[split] = (zc, zs)

        data_variants = DATA_VARIANTS
        fusion_variants = FUSION_VARIANTS
        if config.stage5.reproduce_winner_only:
            data_variants = [v for v in DATA_VARIANTS if v.name == config.stage5.data_variant_name]
            fusion_variants = [
                v for v in FUSION_VARIANTS if v.name == config.stage5.fusion_variant_name
            ]
            if not data_variants or not fusion_variants:
                raise ValueError("Stage5 reproduce-winner config does not match search variants.")

        for data_variant in data_variants:
            built: dict[str, Any] = {}
            for split in ["train", "dev", "test"]:
                examples, manifest_rows, profiles = build_quadrant_dataset(
                    rows=rows_by_split[split],
                    zc=split_embeddings[split][0],
                    zs=split_embeddings[split][1],
                    data_variant=data_variant,
                    split=split,
                    profile_dir=profile_root / candidate_name,
                    candidate_name=candidate_name,
                )
                if not examples:
                    raise ValueError(
                        f"No {split} examples for {candidate_name}/{data_variant.name}."
                    )
                built[split] = {
                    "examples": examples,
                    "manifest_rows": manifest_rows,
                    "profiles": profiles,
                    "counts": dict(Counter(e["quadrant"] for e in examples)),
                }

            for fusion_variant in fusion_variants:
                try:
                    result = train_fusion_variant(
                        config=config,
                        device=device,
                        source_state=source_state,
                        source_stage4_path=candidate_path,
                        candidate_name=candidate_name,
                        data_variant=data_variant,
                        fusion_variant=fusion_variant,
                        train_examples=built["train"]["examples"],
                        dev_examples=built["dev"]["examples"],
                        test_examples=built["test"]["examples"],
                    )
                    model_result = result.pop("model")
                    ckpt_path = (
                        checkpoint_dir
                        / f"stage5_{candidate_name}_{data_variant.name}_{fusion_variant.name}.pt"
                    )
                    metrics = _metrics_from_result(result)
                    save_checkpoint(
                        ckpt_path,
                        stage_origin=5,
                        config=config,
                        step=int(result["best_epoch"]),
                        metrics=metrics,
                        model_state=model_result.state_dict(),
                        optimizer_state={},
                        extra={
                            "gate_checked": True,
                            "gate_passed": bool(result["gate_passed"]),
                            "target_passed": bool(result["target_passed"]),
                            "stage4_candidate": candidate_name,
                            "source_stage4_checkpoint": str(candidate_path),
                            "wake_word": data_variant.wake_word,
                            "n_enroll": data_variant.n_enroll,
                            "best_tau": float(result["best_tau"]),
                            "data_variant": asdict(data_variant),
                            "fusion_variant": asdict(fusion_variant),
                        },
                    )
                    result["stage5_checkpoint"] = str(ckpt_path)
                    result["metrics"] = metrics
                    result["built_counts"] = {
                        split: built[split]["counts"] for split in ["train", "dev", "test"]
                    }
                    all_results.append(result)
                    built_by_checkpoint[str(ckpt_path)] = built
                except Exception as exc:
                    all_results.append(
                        {
                            "stage4_candidate": candidate_name,
                            "source_stage4_checkpoint": str(candidate_path),
                            "data_variant": asdict(data_variant),
                            "fusion_variant": asdict(fusion_variant),
                            "selection_score": -999.0,
                            "gate_passed": False,
                            "error": repr(exc),
                        }
                    )
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    valid = [r for r in all_results if "stage5_checkpoint" in r]
    if not valid:
        raise RuntimeError("No Stage5 variant produced a checkpoint.")
    accepted = [r for r in valid if r.get("gate_passed", False)]
    best = max(accepted or valid, key=lambda r: r["selection_score"])
    best_ckpt = Path(best["stage5_checkpoint"])

    (reports_dir / "stage5_dual_search_results.json").write_text(
        json.dumps(all_results, indent=2) + "\n"
    )
    shutil.copyfile(best_ckpt, checkpoint_dir / "stage5_last.pt")
    if best["gate_passed"]:
        shutil.copyfile(best_ckpt, checkpoint_dir / "stage5_fusion.pt")

    final_built = built_by_checkpoint.get(str(best_ckpt))
    if final_built is not None:
        final_profile_dir = Path("data/stage5/profiles/final_best")
        final_train_rows = _copy_manifest_profiles(
            final_built["train"]["manifest_rows"],
            final_profile_dir,
        )
        final_dev_rows = _copy_manifest_profiles(
            final_built["dev"]["manifest_rows"],
            final_profile_dir,
        )
        final_test_rows = _copy_manifest_profiles(
            final_built["test"]["manifest_rows"],
            final_profile_dir,
        )
        _write_manifest(
            stage5_dir / "stage5_final_train_quadrants.csv",
            final_train_rows,
        )
        _write_manifest(
            stage5_dir / "stage5_final_dev_quadrants.csv",
            final_dev_rows,
        )
        _write_manifest(
            stage5_dir / "stage5_final_test_quadrants.csv",
            final_test_rows,
        )
        _write_manifest(config.data.manifests_dir / "test_kpi.csv", final_test_rows)
        _write_manifest(
            config.data.manifests_dir / "test_fa.csv",
            [row for row in final_test_rows if row.get("quadrant_class") != "Q1_accept"],
        )

    summary = {
        "best": best,
        "best_checkpoint": str(checkpoint_dir / "stage5_last.pt"),
        "strict_checkpoint": str(checkpoint_dir / "stage5_fusion.pt")
        if best["gate_passed"]
        else None,
    }
    (reports_dir / "stage5_dual_search_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (reports_dir / "stage5_dual_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return checkpoint_dir / ("stage5_fusion.pt" if best["gate_passed"] else "stage5_last.pt")
