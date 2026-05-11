"""Stage 4D hard-Q2 mining from the final Kaggle notebook.

The production notebook did not simply continue the compact Stage-4 robustness
loop. It loaded the accepted Stage4C checkpoint, built speaker-disjoint GSC
episodes around the wake word ``zero``, mined the highest-scoring Q2 imposters
inside each step, and fine-tuned the embedding model with a tiny learning rate.
This module keeps that behavior in source form so Stage 5 can compare Stage4C
against the hard-Q2 challenger checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from solospeak.data.features import LogMelExtractor
from solospeak.eval.internal_quadrants import (
    is_gsc_row,
    load_fixed_wav,
    read_csv_rows,
    split_by_speaker,
)
from solospeak.losses.orthogonality import orthogonality_loss
from solospeak.models.solospeak import SoloSpeakModel
from solospeak.training.stages.common import (
    default_device,
    inject_class_counts,
    is_smoke,
    load_checkpoint,
    make_mel,
    save_checkpoint,
)
from solospeak.training.stages.stage5_notebook_search import extract_model_state
from solospeak.utils.config import SoloSpeakConfig
from solospeak.utils.types import MetricsDict


@dataclass(frozen=True)
class HardQ2Variant:
    name: str
    steps: int
    lr: float
    episodes_per_step: int
    q2_candidate_pool: int
    q2_per_episode: int
    q3_per_episode: int
    q4_per_episode: int
    snr_range: tuple[int, int]
    logit_scale: float
    speaker_center: float
    content_center: float
    q1_clean_weight: float
    q1_noisy_weight: float
    q2_weight: float
    q3_weight: float
    q4_weight: float
    speaker_rank_weight: float
    content_rank_weight: float
    noisy_consistency_weight: float
    orthogonality_weight: float
    speaker_margin: float
    content_margin: float
    eval_every: int


HARD_Q2_VARIANTS = [
    HardQ2Variant(
        name="stage4d_hardq2_mining_balanced",
        steps=2200,
        lr=4e-6,
        episodes_per_step=8,
        q2_candidate_pool=48,
        q2_per_episode=8,
        q3_per_episode=2,
        q4_per_episode=2,
        snr_range=(5, 25),
        logit_scale=40.0,
        speaker_center=0.90,
        content_center=0.74,
        q1_clean_weight=1.4,
        q1_noisy_weight=1.1,
        q2_weight=5.5,
        q3_weight=1.0,
        q4_weight=0.7,
        speaker_rank_weight=6.5,
        content_rank_weight=2.0,
        noisy_consistency_weight=0.15,
        orthogonality_weight=0.08,
        speaker_margin=0.15,
        content_margin=0.08,
        eval_every=440,
    ),
    HardQ2Variant(
        name="stage4d_hardq2_mining_strict",
        steps=2400,
        lr=3.5e-6,
        episodes_per_step=8,
        q2_candidate_pool=64,
        q2_per_episode=10,
        q3_per_episode=2,
        q4_per_episode=2,
        snr_range=(8, 25),
        logit_scale=42.0,
        speaker_center=0.90,
        content_center=0.74,
        q1_clean_weight=1.4,
        q1_noisy_weight=0.9,
        q2_weight=6.5,
        q3_weight=1.0,
        q4_weight=0.7,
        speaker_rank_weight=7.5,
        content_rank_weight=2.0,
        noisy_consistency_weight=0.10,
        orthogonality_weight=0.08,
        speaker_margin=0.17,
        content_margin=0.08,
        eval_every=480,
    ),
]


def _build_indices(
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


def _eligible_speakers(
    rows: list[dict[str, Any]],
    by_spk_kw: dict[tuple[str, str], list[int]],
    by_spk: dict[str, list[int]],
    *,
    wake_word: str,
    n_enroll: int,
) -> list[str]:
    speakers: list[str] = []
    for (spk, kw), idxs in by_spk_kw.items():
        if kw != wake_word:
            continue
        wrong_word = [i for i in by_spk[spk] if rows[i]["keyword_text"] != wake_word]
        if len(idxs) >= n_enroll + 1 and wrong_word:
            speakers.append(spk)
    return sorted(set(speakers))


def _rows_by_split(manifests_dir: Path) -> dict[str, list[dict[str, str]]]:
    train_content = read_csv_rows(manifests_dir / "train_content.csv")
    dev_content = read_csv_rows(manifests_dir / "dev_content.csv")
    all_gsc = [row for row in train_content + dev_content if is_gsc_row(row)]
    dedup: dict[str, dict[str, str]] = {}
    for row in all_gsc:
        fp = str(row.get("file_path", ""))
        if fp:
            dedup[fp] = row
    all_gsc = list(dedup.values())
    if not all_gsc:
        raise ValueError("No GSC rows found in train_content/dev_content manifests.")

    for row in all_gsc:
        row["_stage4d_split"] = split_by_speaker(row["speaker_id"])
    rows_by_split = {
        "train": [row for row in all_gsc if row["_stage4d_split"] == "train"],
        "dev": [row for row in all_gsc if row["_stage4d_split"] == "dev"],
        "test": [row for row in all_gsc if row["_stage4d_split"] == "test"],
    }
    speakers = {name: {row["speaker_id"] for row in rows} for name, rows in rows_by_split.items()}
    if not speakers["train"] or not speakers["dev"] or not speakers["test"]:
        raise RuntimeError("Stage4D requires non-empty train/dev/test GSC splits.")
    if not speakers["train"].isdisjoint(speakers["dev"]):
        raise RuntimeError("Stage4D train/dev speaker split leaked.")
    if not speakers["train"].isdisjoint(speakers["test"]):
        raise RuntimeError("Stage4D train/test speaker split leaked.")
    if not speakers["dev"].isdisjoint(speakers["test"]):
        raise RuntimeError("Stage4D dev/test speaker split leaked.")
    return rows_by_split


def _add_white_noise_snr(
    wav: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    noise = rng.standard_normal(wav.shape).astype(np.float32)
    wav_rms = np.sqrt(np.mean(wav**2) + 1e-8)
    noise_rms = np.sqrt(np.mean(noise**2) + 1e-8)
    scale = wav_rms / (noise_rms * (10.0 ** (snr_db / 20.0)))
    return np.clip(wav + noise * scale, -1.0, 1.0).astype(np.float32)


def _random_gain_shift(wav: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gain = float(rng.uniform(0.80, 1.20))
    shift = int(rng.integers(-600, 601))
    out = wav * gain
    out = np.roll(out, shift)
    if shift > 0:
        out[:shift] = 0.0
    elif shift < 0:
        out[shift:] = 0.0
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _augment_wav(
    wav: np.ndarray,
    rng: np.random.Generator,
    snr_range: tuple[int, int],
) -> np.ndarray:
    snr = float(rng.uniform(snr_range[0], snr_range[1]))
    out = _add_white_noise_snr(wav.copy(), snr, rng)
    if rng.random() < 0.35:
        out = _random_gain_shift(out, rng)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _bce_from_score(
    score: torch.Tensor,
    label: float,
    center: float,
    logit_scale: float,
) -> torch.Tensor:
    logit = (score - center) * logit_scale
    target = torch.full_like(logit, float(label))
    return F.binary_cross_entropy_with_logits(logit, target)


def _with_progress(items: range, desc: str) -> Any:
    try:
        from tqdm.auto import tqdm

        return tqdm(items, desc=desc, unit="step")
    except Exception:
        return items


class Stage4D:
    """Run the notebook-authoritative hard-Q2 mining search."""

    stage_id = 4
    stage_name = "stage4d_hardq2_mining"

    def __init__(self, config: SoloSpeakConfig) -> None:
        self.config = config
        inject_class_counts(self.config)
        self.device = default_device()
        self.extractor = LogMelExtractor(config.audio).to(self.device).eval()
        self.rows_by_split: dict[str, list[dict[str, str]]] = {}
        self.indices: dict[str, tuple[Any, Any, Any]] = {}
        self.speakers: dict[str, list[str]] = {}
        self.start_checkpoint = (
            self.config.training.resume_from
            or self.config.training.checkpoint_dir / "stage4_robust.pt"
        )
        self.stage4_obj: dict[str, Any] | None = None

    def _load_start_model(self) -> SoloSpeakModel:
        model = SoloSpeakModel(self.config).to(self.device)
        ckpt = load_checkpoint(self.start_checkpoint)
        if ckpt is None:
            if is_smoke(self.config):
                return model
            raise FileNotFoundError(f"Missing Stage4C checkpoint: {self.start_checkpoint}")
        self.stage4_obj = ckpt
        if ckpt.get("stage_origin") != 4 and not is_smoke(self.config):
            raise ValueError(f"Stage4D expected stage_origin=4: {self.start_checkpoint}")
        state = extract_model_state(ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or (missing and not is_smoke(self.config)):
            raise RuntimeError(
                f"Bad Stage4C state for Stage4D: missing={missing[:20]}, "
                f"unexpected={unexpected[:20]}"
            )
        return model

    def _prepare_rows(self) -> None:
        self.rows_by_split = _rows_by_split(self.config.data.manifests_dir)
        for split in ("train", "dev"):
            rows = self.rows_by_split[split]
            by_spk_kw, by_spk, by_kw = _build_indices(rows)
            self.indices[split] = (by_spk_kw, by_spk, by_kw)
            self.speakers[split] = _eligible_speakers(
                rows,
                by_spk_kw,
                by_spk,
                wake_word=self.config.stage4d.wake_word,
                n_enroll=self.config.stage4d.n_enroll,
            )
            if not self.speakers[split]:
                raise RuntimeError(f"No Stage4D eligible {split} speakers.")

    @torch.no_grad()
    def _embed_wavs(
        self,
        model: SoloSpeakModel,
        wavs_np: list[np.ndarray],
        *,
        batch_size: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model.eval()
        zc_all: list[torch.Tensor] = []
        zs_all: list[torch.Tensor] = []
        for start in range(0, len(wavs_np), batch_size):
            chunk = wavs_np[start : start + batch_size]
            wav = torch.from_numpy(np.stack(chunk, axis=0)).float().to(self.device)
            mel = make_mel({"wav": wav}, self.extractor).float()
            z_c, z_s = model(mel)
            zc_all.append(F.normalize(z_c.float(), p=2, dim=-1))
            zs_all.append(F.normalize(z_s.float(), p=2, dim=-1))
        return torch.cat(zc_all, dim=0), torch.cat(zs_all, dim=0)

    def _sample_episode(
        self,
        *,
        rows: list[dict[str, Any]],
        by_spk_kw: dict[tuple[str, str], list[int]],
        by_spk: dict[str, list[int]],
        by_kw: dict[str, list[int]],
        speakers: list[str],
        variant: HardQ2Variant,
        rng: random.Random,
        np_rng: np.random.Generator,
    ) -> dict[str, Any]:
        wake_word = self.config.stage4d.wake_word
        n_enroll = self.config.stage4d.n_enroll
        spk = rng.choice(speakers)
        wake_idxs = list(by_spk_kw[(spk, wake_word)])
        rng.shuffle(wake_idxs)
        enroll_idxs = wake_idxs[:n_enroll]
        q1_idx = wake_idxs[n_enroll]

        q2_pool = [i for i in by_kw[wake_word] if rows[i]["speaker_id"] != spk]
        q3_pool = [i for i in by_spk[spk] if rows[i]["keyword_text"] != wake_word]
        q4_pool = [
            i
            for i, row in enumerate(rows)
            if row["speaker_id"] != spk and row["keyword_text"] != wake_word
        ]

        rng.shuffle(q2_pool)
        rng.shuffle(q3_pool)
        rng.shuffle(q4_pool)
        q2_candidates = q2_pool[: min(variant.q2_candidate_pool, len(q2_pool))]
        q3_idxs = q3_pool[: variant.q3_per_episode]
        q4_idxs = q4_pool[: variant.q4_per_episode]

        enroll_wavs = [load_fixed_wav(rows[i], self.config) for i in enroll_idxs]
        q1_clean = load_fixed_wav(rows[q1_idx], self.config)
        q1_noisy = _augment_wav(q1_clean, np_rng, variant.snr_range)
        return {
            "speaker_id": spk,
            "enroll_wavs": enroll_wavs,
            "q1_clean": q1_clean,
            "q1_noisy": q1_noisy,
            "q2_candidate_wavs": [load_fixed_wav(rows[i], self.config) for i in q2_candidates],
            "q3_wavs": [load_fixed_wav(rows[i], self.config) for i in q3_idxs],
            "q4_wavs": [load_fixed_wav(rows[i], self.config) for i in q4_idxs],
        }

    def _step_loss(
        self,
        model: SoloSpeakModel,
        episodes: list[dict[str, Any]],
        variant: HardQ2Variant,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        all_wavs: list[np.ndarray] = []
        meta: list[dict[str, Any]] = []
        for episode in episodes:
            start = len(all_wavs)
            all_wavs.extend(episode["enroll_wavs"])
            enroll_range = list(range(start, start + len(episode["enroll_wavs"])))
            q1_clean_i = len(all_wavs)
            all_wavs.append(episode["q1_clean"])
            q1_noisy_i = len(all_wavs)
            all_wavs.append(episode["q1_noisy"])

            q2_candidate_range: list[int] = []
            for wav in episode["q2_candidate_wavs"]:
                q2_candidate_range.append(len(all_wavs))
                all_wavs.append(wav)
            q3_range: list[int] = []
            for wav in episode["q3_wavs"]:
                q3_range.append(len(all_wavs))
                all_wavs.append(wav)
            q4_range: list[int] = []
            for wav in episode["q4_wavs"]:
                q4_range.append(len(all_wavs))
                all_wavs.append(wav)
            meta.append(
                {
                    "enroll": enroll_range,
                    "q1_clean": q1_clean_i,
                    "q1_noisy": q1_noisy_i,
                    "q2_candidates": q2_candidate_range,
                    "q3": q3_range,
                    "q4": q4_range,
                }
            )

        wav = torch.from_numpy(np.stack(all_wavs, axis=0)).float().to(self.device)
        mel = make_mel({"wav": wav}, self.extractor).float()
        z_c, z_s = model(mel)
        z_c = F.normalize(z_c.float(), p=2, dim=-1)
        z_s = F.normalize(z_s.float(), p=2, dim=-1)

        losses: list[torch.Tensor] = []
        debug: defaultdict[str, float] = defaultdict(float)
        for m in meta:
            enroll = torch.tensor(m["enroll"], device=self.device, dtype=torch.long)
            ct = F.normalize(z_c[enroll].mean(dim=0, keepdim=True), p=2, dim=-1)
            st = F.normalize(z_s[enroll].mean(dim=0, keepdim=True), p=2, dim=-1)

            q1_clean = int(m["q1_clean"])
            q1_noisy = int(m["q1_noisy"])
            q1_clean_sc = (z_c[q1_clean : q1_clean + 1] * ct).sum(dim=-1)
            q1_clean_ss = (z_s[q1_clean : q1_clean + 1] * st).sum(dim=-1)
            q1_noisy_sc = (z_c[q1_noisy : q1_noisy + 1] * ct).sum(dim=-1)
            q1_noisy_ss = (z_s[q1_noisy : q1_noisy + 1] * st).sum(dim=-1)

            l_q1_clean = (
                _bce_from_score(q1_clean_sc, 1.0, variant.content_center, variant.logit_scale)
                + _bce_from_score(q1_clean_ss, 1.0, variant.speaker_center, variant.logit_scale)
            ) * variant.q1_clean_weight
            l_q1_noisy = (
                _bce_from_score(q1_noisy_sc, 1.0, variant.content_center, variant.logit_scale)
                + _bce_from_score(q1_noisy_ss, 1.0, variant.speaker_center, variant.logit_scale)
            ) * variant.q1_noisy_weight
            l_cons = (
                1.0
                - (z_s[q1_clean] * z_s[q1_noisy]).sum()
                + 0.35 * (1.0 - (z_c[q1_clean] * z_c[q1_noisy]).sum())
            ) * variant.noisy_consistency_weight
            losses.extend([l_q1_clean, l_q1_noisy, l_cons])

            if m["q2_candidates"]:
                q2_cand_idx = torch.tensor(
                    m["q2_candidates"], device=self.device, dtype=torch.long
                )
                q2_cand_sc = (z_c[q2_cand_idx] * ct).sum(dim=-1)
                q2_cand_ss = (z_s[q2_cand_idx] * st).sum(dim=-1)
                k = min(variant.q2_per_episode, q2_cand_ss.numel())
                _, hard_pos = torch.topk(q2_cand_ss.detach(), k=k, largest=True)
                hard_idx = q2_cand_idx[hard_pos]
                q2_sc = (z_c[hard_idx] * ct).sum(dim=-1)
                q2_ss = (z_s[hard_idx] * st).sum(dim=-1)
                l_q2_content_pos = _bce_from_score(
                    q2_sc,
                    1.0,
                    variant.content_center,
                    variant.logit_scale,
                ) * 0.15
                l_q2_speaker_neg = _bce_from_score(
                    q2_ss,
                    0.0,
                    variant.speaker_center,
                    variant.logit_scale,
                ) * variant.q2_weight
                l_q2_rank_clean = F.relu(
                    variant.speaker_margin - q1_clean_ss.mean() + q2_ss
                ).mean() * variant.speaker_rank_weight
                l_q2_rank_noisy = F.relu(
                    (variant.speaker_margin * 0.70) - q1_noisy_ss.mean() + q2_ss
                ).mean() * (variant.speaker_rank_weight * 0.45)
                losses.extend(
                    [
                        l_q2_content_pos,
                        l_q2_speaker_neg,
                        l_q2_rank_clean,
                        l_q2_rank_noisy,
                    ]
                )
                debug["q2_hard_ss"] += float(q2_ss.mean().detach().cpu())
                debug["q2_cand_ss"] += float(q2_cand_ss.mean().detach().cpu())
                debug["q2_max_ss"] += float(q2_cand_ss.max().detach().cpu())

            if m["q3"]:
                q3_idx = torch.tensor(m["q3"], device=self.device, dtype=torch.long)
                q3_sc = (z_c[q3_idx] * ct).sum(dim=-1)
                q3_ss = (z_s[q3_idx] * st).sum(dim=-1)
                l_q3_content_neg = _bce_from_score(
                    q3_sc,
                    0.0,
                    variant.content_center,
                    variant.logit_scale,
                ) * variant.q3_weight
                l_q3_speaker_pos = _bce_from_score(
                    q3_ss,
                    1.0,
                    variant.speaker_center,
                    variant.logit_scale,
                ) * 0.10
                l_q3_rank = F.relu(
                    variant.content_margin - q1_clean_sc.mean() + q3_sc
                ).mean() * variant.content_rank_weight
                losses.extend([l_q3_content_neg, l_q3_speaker_pos, l_q3_rank])

            if m["q4"]:
                q4_idx = torch.tensor(m["q4"], device=self.device, dtype=torch.long)
                q4_sc = (z_c[q4_idx] * ct).sum(dim=-1)
                q4_ss = (z_s[q4_idx] * st).sum(dim=-1)
                l_q4 = (
                    _bce_from_score(q4_sc, 0.0, variant.content_center, variant.logit_scale)
                    + _bce_from_score(q4_ss, 0.0, variant.speaker_center, variant.logit_scale)
                ) * variant.q4_weight
                losses.append(l_q4)

            debug["q1_clean_ss"] += float(q1_clean_ss.mean().detach().cpu())
            debug["q1_noisy_ss"] += float(q1_noisy_ss.mean().detach().cpu())
            debug["q1_clean_sc"] += float(q1_clean_sc.mean().detach().cpu())
            debug["q1_noisy_sc"] += float(q1_noisy_sc.mean().detach().cpu())

        base_loss = torch.stack([x if x.ndim == 0 else x.mean() for x in losses]).mean()
        ortho = orthogonality_loss(z_c, z_s) * variant.orthogonality_weight
        total = base_loss + ortho
        n = max(len(meta), 1)
        for key in list(debug.keys()):
            debug[key] /= n
        debug["base_loss"] = float(base_loss.detach().cpu())
        debug["ortho"] = float(ortho.detach().cpu())
        debug["total"] = float(total.detach().cpu())
        return total, dict(debug)

    @torch.no_grad()
    def _build_eval_scores(
        self,
        model: SoloSpeakModel,
        *,
        max_profiles: int | None,
    ) -> list[dict[str, float | str]]:
        rows = self.rows_by_split["dev"]
        by_spk_kw, by_spk, by_kw = self.indices["dev"]
        speakers = list(self.speakers["dev"])
        wake_word = self.config.stage4d.wake_word
        n_enroll = self.config.stage4d.n_enroll
        rng = random.Random(999)
        np_rng = np.random.default_rng(999)
        rng.shuffle(speakers)
        if max_profiles is not None:
            speakers = speakers[:max_profiles]

        trial_rows: list[dict[str, Any]] = []
        wavs: list[np.ndarray] = []
        for spk in speakers:
            wake_idxs = list(by_spk_kw[(spk, wake_word)])
            rng.shuffle(wake_idxs)
            if len(wake_idxs) < n_enroll + 1:
                continue
            enroll_idxs = wake_idxs[:n_enroll]
            q1_pool = wake_idxs[n_enroll:]
            q2_pool = [i for i in by_kw[wake_word] if rows[i]["speaker_id"] != spk]
            q3_pool = [i for i in by_spk[spk] if rows[i]["keyword_text"] != wake_word]
            q4_pool = [
                i
                for i, row in enumerate(rows)
                if row["speaker_id"] != spk and row["keyword_text"] != wake_word
            ]
            if not q1_pool or not q2_pool or not q3_pool or not q4_pool:
                continue

            enroll_wavs = [load_fixed_wav(rows[i], self.config) for i in enroll_idxs]
            zc_e, zs_e = self._embed_wavs(model, enroll_wavs)
            ct_np = (
                F.normalize(zc_e.mean(dim=0, keepdim=True), p=2, dim=-1)
                .detach()
                .cpu()
                .numpy()[0]
            )
            st_np = (
                F.normalize(zs_e.mean(dim=0, keepdim=True), p=2, dim=-1)
                .detach()
                .cpu()
                .numpy()[0]
            )
            profile = {"speaker_id": spk, "ct": ct_np, "st": st_np}

            rng.shuffle(q1_pool)
            for idx in q1_pool[:2]:
                wavs.append(load_fixed_wav(rows[idx], self.config))
                trial_rows.append({"quadrant": "Q1_accept", "profile": profile})
            for idx in q1_pool[:2]:
                clean = load_fixed_wav(rows[idx], self.config)
                wavs.append(_augment_wav(clean, np_rng, (-5, 20)))
                trial_rows.append({"quadrant": "Q1_noisy", "profile": profile})

            rng.shuffle(q2_pool)
            for idx in q2_pool[:6]:
                wavs.append(load_fixed_wav(rows[idx], self.config))
                trial_rows.append({"quadrant": "Q2_imposter", "profile": profile})
            rng.shuffle(q3_pool)
            for idx in q3_pool[:2]:
                wavs.append(load_fixed_wav(rows[idx], self.config))
                trial_rows.append({"quadrant": "Q3_wrong_word", "profile": profile})
            rng.shuffle(q4_pool)
            for idx in q4_pool[:2]:
                wavs.append(load_fixed_wav(rows[idx], self.config))
                trial_rows.append({"quadrant": "Q4_background", "profile": profile})

        if not wavs:
            raise RuntimeError("Stage4D eval built no trials.")
        zc, zs = self._embed_wavs(model, wavs)
        scores: list[dict[str, float | str]] = []
        for idx, trial in enumerate(trial_rows):
            ct = F.normalize(
                torch.from_numpy(trial["profile"]["ct"]).float().to(self.device),
                p=2,
                dim=-1,
            )
            st = F.normalize(
                torch.from_numpy(trial["profile"]["st"]).float().to(self.device),
                p=2,
                dim=-1,
            )
            scores.append(
                {
                    "quadrant": str(trial["quadrant"]),
                    "s_c": float((zc[idx] * ct).sum().detach().cpu()),
                    "s_s": float((zs[idx] * st).sum().detach().cpu()),
                }
            )
        return scores

    def _eval_scores_with_thresholds(
        self,
        scores: list[dict[str, float | str]],
    ) -> dict[str, float]:
        q = np.array([score["quadrant"] for score in scores], dtype=object)
        sc = np.array([score["s_c"] for score in scores], dtype=np.float32)
        ss = np.array([score["s_s"] for score in scores], dtype=np.float32)
        q1 = q == "Q1_accept"
        q1n = q == "Q1_noisy"
        q2 = q == "Q2_imposter"
        q3 = q == "Q3_wrong_word"
        q4 = q == "Q4_background"
        content_grid = np.unique(np.round(np.quantile(sc, np.linspace(0.05, 0.95, 50)), 6))
        speaker_grid = np.unique(np.round(np.quantile(ss, np.linspace(0.05, 0.95, 50)), 6))

        def frac(mask: np.ndarray, values: np.ndarray) -> float:
            denom = int(mask.sum())
            if denom <= 0:
                return 0.0
            return float(values[mask].mean())

        best: dict[str, float] | None = None
        for ct in content_grid:
            content_ok = sc >= ct
            for st in speaker_grid:
                accept = content_ok & (ss >= st)
                ta_clean = frac(q1, accept.astype(np.float32))
                ta_noisy = frac(q1n, accept.astype(np.float32))
                q2_rej = frac(q2, (~accept).astype(np.float32))
                q3_rej = frac(q3, (~accept).astype(np.float32))
                q4_rej = frac(q4, (~accept).astype(np.float32))
                vals = [ta_clean, q2_rej, q3_rej, q4_rej]
                qmin = min(vals)
                qmean = sum(vals) / len(vals)
                score = (
                    6.0 * q2_rej
                    + 4.0 * min(ta_clean, q2_rej)
                    + 2.5 * ta_noisy
                    + 1.5 * qmin
                    + 0.5 * qmean
                )
                cur = {
                    "ta_clean": ta_clean,
                    "ta_noisy_macro": ta_noisy,
                    "q2_rejection": q2_rej,
                    "q3_rejection": q3_rej,
                    "q4_rejection": q4_rej,
                    "quadrant_min": qmin,
                    "quadrant_mean": qmean,
                    "content_tau": float(ct),
                    "speaker_tau": float(st),
                    "accepted_rate": float(accept.mean()),
                    "score": float(score),
                    "n_trials": float(len(scores)),
                    "n_q1": float(q1.sum()),
                    "n_q1_noisy": float(q1n.sum()),
                    "n_q2": float(q2.sum()),
                    "n_q3": float(q3.sum()),
                    "n_q4": float(q4.sum()),
                    "mean_q1_ss": float(ss[q1].mean()) if q1.any() else 0.0,
                    "mean_q1n_ss": float(ss[q1n].mean()) if q1n.any() else 0.0,
                    "mean_q2_ss": float(ss[q2].mean()) if q2.any() else 0.0,
                    "mean_q1_sc": float(sc[q1].mean()) if q1.any() else 0.0,
                    "mean_q1n_sc": float(sc[q1n].mean()) if q1n.any() else 0.0,
                    "mean_q2_sc": float(sc[q2].mean()) if q2.any() else 0.0,
                }
                if best is None or cur["score"] > best["score"]:
                    best = cur
        if best is None:
            raise RuntimeError("Stage4D eval found no threshold candidate.")
        return best

    def _eval(self, model: SoloSpeakModel) -> dict[str, float]:
        scores = self._build_eval_scores(
            model,
            max_profiles=self.config.stage4d.max_eval_profiles,
        )
        return self._eval_scores_with_thresholds(scores)

    def _metrics(
        self,
        *,
        best_eval: dict[str, float],
        best_step: int,
        accepted: bool,
        selection_score: float,
    ) -> MetricsDict:
        metrics: MetricsDict = {
            "dev/stage4d_hardq2_best_step": float(best_step),
            "dev/stage4d_hardq2_ta_clean": float(best_eval["ta_clean"]),
            "dev/stage4d_hardq2_ta_noisy_macro": float(best_eval["ta_noisy_macro"]),
            "dev/stage4d_hardq2_q2_rejection": float(best_eval["q2_rejection"]),
            "dev/stage4d_hardq2_q3_rejection": float(best_eval["q3_rejection"]),
            "dev/stage4d_hardq2_q4_rejection": float(best_eval["q4_rejection"]),
            "dev/stage4d_hardq2_quadrant_min": float(best_eval["quadrant_min"]),
            "dev/stage4d_hardq2_quadrant_mean": float(best_eval["quadrant_mean"]),
            "dev/stage4d_hardq2_content_tau": float(best_eval["content_tau"]),
            "dev/stage4d_hardq2_speaker_tau": float(best_eval["speaker_tau"]),
            "dev/stage4d_hardq2_accepted_rate": float(best_eval["accepted_rate"]),
            "dev/stage4d_hardq2_selection_score": float(selection_score),
            "dev/accepted_for_stage5": float(accepted),
        }
        for key, value in best_eval.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                metrics[f"dev/stage4d_hardq2_{key}"] = float(value)
        return metrics

    def _save_variant(
        self,
        *,
        model: SoloSpeakModel,
        variant: HardQ2Variant,
        best_step: int,
        best_eval: dict[str, float],
        accepted: bool,
        selection_score: float,
    ) -> Path:
        path = (
            self.config.stage4d.output_checkpoint
            if variant.name == "stage4d_hardq2_mining_balanced"
            else self.config.training.checkpoint_dir / f"{variant.name}.pt"
        )
        metrics = self._metrics(
            best_eval=best_eval,
            best_step=best_step,
            accepted=accepted,
            selection_score=selection_score,
        )
        return save_checkpoint(
            path,
            stage_origin=4,
            config=self.config,
            step=int(best_step),
            metrics=metrics,
            model_state=model.state_dict(),
            optimizer_state={},
            extra={
                "stage4d_hardq2": True,
                "variant": asdict(variant),
                "source_checkpoint": str(self.start_checkpoint),
                "accepted_for_stage5": bool(accepted),
                "best_eval": best_eval,
                "note": "Online hard-Q2 mining checkpoint.",
            },
        )

    def _train_variant(self, variant: HardQ2Variant) -> dict[str, Any]:
        model = self._load_start_model().to(self.device)
        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=variant.lr, weight_decay=1e-5)
        seed_base = 2027 + int(hashlib.sha256(variant.name.encode()).hexdigest(), 16) % 100000
        rng = random.Random(seed_base)
        np_rng = np.random.default_rng(seed_base + 17)
        by_spk_kw, by_spk, by_kw = self.indices["train"]

        best_state: dict[str, torch.Tensor] | None = None
        best_eval: dict[str, float] | None = None
        best_step = 0
        best_score = -1e9

        for step in _with_progress(range(1, variant.steps + 1), f"hardq2:{variant.name}"):
            model.train()
            episodes = [
                self._sample_episode(
                    rows=self.rows_by_split["train"],
                    by_spk_kw=by_spk_kw,
                    by_spk=by_spk,
                    by_kw=by_kw,
                    speakers=self.speakers["train"],
                    variant=variant,
                    rng=rng,
                    np_rng=np_rng,
                )
                for _ in range(variant.episodes_per_step)
            ]
            optimizer.zero_grad(set_to_none=True)
            loss, _ = self._step_loss(model, episodes, variant)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % variant.eval_every == 0 or step == variant.steps:
                ev = self._eval(model)
                if ev["score"] > best_score:
                    best_score = ev["score"]
                    best_eval = ev
                    best_step = step
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if best_state is None or best_eval is None:
            raise RuntimeError(f"Stage4D variant produced no best state: {variant.name}")
        model.load_state_dict(best_state, strict=False)
        accepted = (
            best_eval["ta_clean"] >= 0.92
            and best_eval["q2_rejection"] >= 0.94
            and best_eval["ta_noisy_macro"] >= 0.68
            and best_eval["quadrant_min"] >= 0.90
            and best_eval["q3_rejection"] >= 0.90
            and best_eval["q4_rejection"] >= 0.99
        )
        selection_score = (
            100.0 * float(accepted)
            + 7.0 * best_eval["q2_rejection"]
            + 4.0 * best_eval["ta_clean"]
            + 3.0 * best_eval["ta_noisy_macro"]
            + 2.0 * best_eval["quadrant_min"]
            + best_eval["quadrant_mean"]
        )
        ckpt_path = self._save_variant(
            model=model,
            variant=variant,
            best_step=best_step,
            best_eval=best_eval,
            accepted=accepted,
            selection_score=selection_score,
        )
        return {
            "name": variant.name,
            "ckpt": str(ckpt_path),
            "score": float(selection_score),
            "accepted_for_stage5": bool(accepted),
            "best_step": int(best_step),
            "metrics": self._metrics(
                best_eval=best_eval,
                best_step=best_step,
                accepted=accepted,
                selection_score=selection_score,
            ),
            "eval": best_eval,
            "variant": asdict(variant),
        }

    def _smoke_checkpoint(self) -> Path:
        model = self._load_start_model().to(self.device)
        metrics: MetricsDict = {
            "dev/stage4d_hardq2_best_step": 0.0,
            "dev/stage4d_hardq2_ta_clean": 1.0,
            "dev/stage4d_hardq2_q2_rejection": 1.0,
            "dev/stage4d_hardq2_q3_rejection": 1.0,
            "dev/stage4d_hardq2_q4_rejection": 1.0,
            "dev/stage4d_hardq2_quadrant_min": 1.0,
            "dev/stage4d_hardq2_quadrant_mean": 1.0,
            "dev/stage4d_hardq2_content_tau": 0.74,
            "dev/stage4d_hardq2_speaker_tau": 0.90,
            "dev/stage4d_hardq2_accepted_rate": 0.25,
            "dev/stage4d_hardq2_selection_score": 111.0,
            "dev/accepted_for_stage5": 1.0,
            "gate/smoke_only": 1.0,
        }
        path = save_checkpoint(
            self.config.stage4d.output_checkpoint,
            stage_origin=4,
            config=self.config,
            step=0,
            metrics=metrics,
            model_state=model.state_dict(),
            optimizer_state={},
            extra={
                "stage4d_hardq2": True,
                "variant": asdict(HARD_Q2_VARIANTS[0]),
                "source_checkpoint": str(self.start_checkpoint),
                "accepted_for_stage5": True,
                "smoke_only": True,
            },
        )
        self.config.stage4d.results_json.parent.mkdir(parents=True, exist_ok=True)
        self.config.stage4d.results_json.write_text(
            json.dumps(
                [
                    {
                        "name": HARD_Q2_VARIANTS[0].name,
                        "ckpt": str(path),
                        "score": 111.0,
                        "accepted_for_stage5": True,
                        "smoke_only": True,
                    }
                ],
                indent=2,
            )
            + "\n"
        )
        shutil.copyfile(path, self.config.stage4d.last_checkpoint)
        shutil.copyfile(path, self.config.stage4d.production_checkpoint)
        return path

    def run(self) -> Path:
        if is_smoke(self.config):
            return self._smoke_checkpoint()
        self._prepare_rows()
        results: list[dict[str, Any]] = []
        for variant in HARD_Q2_VARIANTS:
            try:
                results.append(self._train_variant(variant))
            except Exception as exc:
                results.append(
                    {
                        "name": variant.name,
                        "ckpt": "",
                        "score": -999.0,
                        "accepted_for_stage5": False,
                        "error": repr(exc),
                        "variant": asdict(variant),
                    }
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        valid = [result for result in results if result.get("ckpt")]
        if not valid:
            raise RuntimeError("No Stage4D hard-Q2 variant completed successfully.")
        accepted = [result for result in valid if result["accepted_for_stage5"]]
        best = max(accepted or valid, key=lambda item: float(item["score"]))
        best_ckpt = Path(str(best["ckpt"]))

        self.config.stage4d.results_json.parent.mkdir(parents=True, exist_ok=True)
        self.config.stage4d.results_json.write_text(json.dumps(results, indent=2) + "\n")
        shutil.copyfile(best_ckpt, self.config.stage4d.last_checkpoint)
        if best["accepted_for_stage5"]:
            shutil.copyfile(best_ckpt, self.config.stage4d.production_checkpoint)
            if self.config.stage4d.replace_stage4_robust_on_accept:
                shutil.copyfile(best_ckpt, self.config.training.checkpoint_dir / "stage4_robust.pt")
        return best_ckpt


def run_stage4d_hardq2_mining(config: SoloSpeakConfig) -> Path:
    """Convenience function used by CLI orchestration."""

    return Stage4D(config).run()
