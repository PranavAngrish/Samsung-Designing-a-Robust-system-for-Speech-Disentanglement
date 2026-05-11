"""Stage 7: fusion-only external false-accept hard-negative tuning.

This stage is the source-code version of the final ``ta-test.ipynb`` Stage-7
cells. It rebuilds GSC internal profile examples, mines external false accepts
from the Stage-6 model, fine-tunes only the fusion MLP, evaluates a validation
threshold sweep, and saves ``stage7_fusion.pt``.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from solospeak.data.features import LogMelExtractor
from solospeak.eval.external_fa import (
    discover_external_fa_audio,
    external_fa_spec_from_config,
    make_external_fa_rows,
    score_external_fa_rows,
    write_external_fa_manifest,
)
from solospeak.eval.internal_quadrants import (
    InternalQuadrantSpec,
    build_gsc_internal_examples,
)
from solospeak.models.solospeak import SoloSpeakModel
from solospeak.training.stages.base import TrainingStage
from solospeak.training.stages.common import (
    default_device,
    inject_class_counts,
    is_smoke,
    load_checkpoint,
    save_checkpoint,
)
from solospeak.utils.config import SoloSpeakConfig
from solospeak.utils.types import LossDict, MetricsDict, StageBatch


STAGE7_EPOCHS = 700
STAGE7_PATIENCE = 100
STAGE7_BATCH_SIZE = 256
STAGE7_LR = 5e-4
STAGE7_WEIGHT_DECAY = 1e-4
VAL_FRAC = 0.25
VAL_SPLIT_SEED = 3030
EXTERNAL_NEGATIVE_SEED = 2027
MAX_EASY_EXTERNAL_NEGATIVES = 12000
HARD_NEGATIVE_MULTIPLIER = 6.0
SOFT_EXTERNAL_WEIGHT = 1.5
BASE_TAU_DEFAULT = 0.65
STAGE7_TRAINABLE_FUSION_PARAMS = {
    "fusion_mlp.net.0.weight",
    "fusion_mlp.net.0.bias",
    "fusion_mlp.net.2.weight",
    "fusion_mlp.net.2.bias",
    "fusion_mlp.net.4.weight",
    "fusion_mlp.net.4.bias",
}


def _source_base_weight(source: str) -> float:
    if source == "common_voice":
        return 2.5
    if source == "librispeech":
        return 2.0
    if source == "background_noise":
        return 2.0
    if source == "urbansound8k":
        return 1.5
    return SOFT_EXTERNAL_WEIGHT


def _default_internal_weight(quadrant: str) -> float:
    if quadrant == "Q1_accept":
        return 2.5
    if quadrant == "Q2_imposter":
        return 2.5
    if quadrant == "Q3_wrong_word":
        return 1.5
    if quadrant == "Q4_background":
        return 1.5
    return 1.0


def tensorize_examples(
    examples: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    x = torch.tensor([[e["s_c"], e["s_s"]] for e in examples], dtype=torch.float32)
    y = torch.tensor([e["label"] for e in examples], dtype=torch.float32)
    w = torch.tensor([e.get("weight", 1.0) for e in examples], dtype=torch.float32)
    q = np.array([e["quadrant"] for e in examples], dtype=object)
    src = np.array([e.get("source_dataset", "") for e in examples], dtype=object)
    return x, y, w, q, src


@torch.no_grad()
def eval_stage7_examples(
    model: SoloSpeakModel,
    examples: list[dict[str, Any]],
    *,
    device: torch.device,
    tau: float | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    x, _, _, q, _ = tensorize_examples(examples)
    probs: list[torch.Tensor] = []
    for start in range(0, len(x), 4096):
        xb = x[start : start + 4096].to(device)
        p = model.forward_fusion(xb[:, 0], xb[:, 1]).view(-1)
        probs.append(p.detach().cpu())
    prob_np = torch.cat(probs).numpy() if probs else np.array([], dtype=np.float32)

    def frac(mask: np.ndarray, values: np.ndarray) -> float:
        denom = int(mask.sum())
        if denom <= 0:
            return 0.0
        return float(values[mask].mean())

    def metrics_for_threshold(threshold: float, include_score: bool) -> dict[str, float]:
        accept = prob_np >= threshold
        q1 = q == "Q1_accept"
        q2 = q == "Q2_imposter"
        q3 = q == "Q3_wrong_word"
        q4 = q == "Q4_background"
        ext = np.isin(q, ["EXTERNAL_FA", "EXTERNAL_FA_HARD"])

        ta = frac(q1, accept.astype(np.float32))
        q2r = frac(q2, (~accept).astype(np.float32))
        q3r = frac(q3, (~accept).astype(np.float32))
        q4r = frac(q4, (~accept).astype(np.float32))
        extr = frac(ext, (~accept).astype(np.float32))
        ext_fa_rate = 1.0 - extr if int(ext.sum()) else 0.0
        vals = [ta, q2r, q3r, q4r]
        quad_min = min(vals)
        quad_mean = sum(vals) / 4.0
        out = {
            "tau": float(threshold),
            "ta_clean": float(ta),
            "q2_rejection": float(q2r),
            "q3_rejection": float(q3r),
            "q4_rejection": float(q4r),
            "external_rejection": float(extr),
            "external_fa_rate": float(ext_fa_rate),
            "quadrant_accuracy_min": float(quad_min),
            "quadrant_accuracy_mean": float(quad_mean),
            "accepted_rate": float(accept.mean()) if len(accept) else 0.0,
            "mean_prob": float(prob_np.mean()) if len(prob_np) else 0.0,
        }
        if include_score:
            out["score"] = float(
                10.0 * quad_min
                + 4.0 * ta
                + 4.0 * q2r
                + 3.0 * extr
                - 20.0 * max(0.0, ext_fa_rate - 0.005)
            )
        return out

    if tau is not None:
        return metrics_for_threshold(tau, include_score=False), prob_np

    best: dict[str, float] | None = None
    for threshold in np.linspace(0.30, 0.90, 121):
        candidate = metrics_for_threshold(float(threshold), include_score=True)
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    if best is None:
        raise ValueError("Cannot evaluate Stage7 without examples.")
    return best, prob_np


class Stage7(TrainingStage):
    stage_id = 7
    stage_name = "stage7_fa_tuning"
    min_gate_metric = "dev/stage7_demo_passed"
    min_gate_threshold = 1.0
    target_gate_threshold = 1.0

    def __init__(self, config: SoloSpeakConfig) -> None:
        super().__init__(config)
        inject_class_counts(self.config)
        self.device = default_device()
        self.extractor = LogMelExtractor(config.audio).to(self.device)
        self.model: SoloSpeakModel | None = None
        self.source_checkpoint: dict[str, Any] | None = None

    def prepare_data(self) -> tuple[Any, ...]:
        return ()

    def build_model(self) -> nn.Module:
        self.model = SoloSpeakModel(self.config).to(self.device)
        ckpt = load_checkpoint(self.config.training.resume_from)
        self.source_checkpoint = ckpt
        if ckpt is not None and "model_state" in ckpt:
            missing, unexpected = self.model.load_state_dict(ckpt["model_state"], strict=False)
            if not is_smoke(self.config) and (missing or unexpected):
                raise RuntimeError(
                    "Stage7 expected a clean Stage6 model_state; "
                    f"missing={missing[:20]}, unexpected={unexpected[:20]}"
                )
        for param in self.model.parameters():
            param.requires_grad = False
        for name, param in self.model.named_parameters():
            if name in STAGE7_TRAINABLE_FUSION_PARAMS:
                param.requires_grad = True
        trainable_names = {name for name, p in self.model.named_parameters() if p.requires_grad}
        if trainable_names != STAGE7_TRAINABLE_FUSION_PARAMS:
            raise RuntimeError(
                "Stage7 must tune only the fusion MLP parameters; "
                f"got trainable={sorted(trainable_names)}"
            )
        return self.model

    def compute_loss(self, batches: StageBatch, step: int) -> LossDict:
        raise NotImplementedError("Stage7 uses a notebook-faithful custom run loop.")

    def _base_tau(self) -> float:
        ckpt = self.source_checkpoint or {}
        metrics = ckpt.get("metrics", {}) or {}
        extra = ckpt.get("extra", {}) or {}
        if "stage6/tau_on" in metrics:
            return float(metrics["stage6/tau_on"])
        if "tau_on" in extra:
            return float(extra["tau_on"])
        return float(self.config.fusion.tau_on or BASE_TAU_DEFAULT)

    def _smoke_checkpoint(self) -> Path:
        assert self.model is not None
        metrics: MetricsDict = {
            "dev/best_epoch": 0.0,
            "dev/tau_stage7": float(self.config.fusion.tau_on),
            "dev/stage7_gate_passed": 1.0,
            "dev/stage7_demo_passed": 1.0,
            "dev/ta_clean": 1.0,
            "dev/q2_rejection": 1.0,
            "dev/q3_rejection": 1.0,
            "dev/q4_rejection": 1.0,
            "dev/external_fa_rate": 0.0,
            "dev/external_rejection": 1.0,
            "dev/quadrant_accuracy_min": 1.0,
            "dev/quadrant_accuracy_mean": 1.0,
            "gate/smoke_only": 1.0,
        }
        return save_checkpoint(
            self.config.stage7.stage7_checkpoint,
            stage_origin=self.stage_id,
            config=self.config,
            step=0,
            metrics=metrics,
            model_state=self.model.state_dict(),
            optimizer_state={},
            extra={
                "source_stage6_checkpoint": str(self.config.training.resume_from or ""),
                "base_tau": float(self.config.fusion.tau_on),
                "stage7_tau": float(self.config.fusion.tau_on),
                "wake_word": self.config.stage7.wake_word,
                "n_enroll": self.config.stage7.n_enroll,
                "stage7_gate_passed": True,
                "stage7_demo_passed": True,
                "smoke_only": True,
            },
        )

    def _build_training_examples(self, base_tau: float) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        int,
        int,
    ]:
        assert self.model is not None
        stage7_dir = self.config.data.root / "stage7"
        reports_dir = Path("reports")
        profile_dir = stage7_dir / "profiles"
        stage7_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        internal_spec = InternalQuadrantSpec(
            wake_word=self.config.stage7.wake_word,
            n_enroll=self.config.stage7.n_enroll,
            max_profiles=self.config.stage7.max_profiles,
            profile_seed=777,
            q1_per_profile=2,
            q2_per_profile=6,
            q3_per_profile=2,
            q4_per_profile=2,
            require_wrong_word_for_profile=False,
            profile_prefix="stage7_profile",
        )
        profiles, internal_examples = build_gsc_internal_examples(
            model=self.model,
            extractor=self.extractor,
            config=self.config,
            manifests_dir=self.config.data.manifests_dir,
            profile_dir=profile_dir,
            spec=internal_spec,
            device=self.device,
        )

        external_spec = external_fa_spec_from_config(self.config)
        files_by_source = discover_external_fa_audio(external_spec)
        if not any(files_by_source.values()):
            raise FileNotFoundError(
                "No external FA audio found. Attach Common Voice, LibriSpeech, "
                "background noise, or UrbanSound8K datasets before Stage7."
            )

        external_rows = make_external_fa_rows(
            files_by_source=files_by_source,
            profiles=profiles,
            config=self.config,
            spec=external_spec,
        )
        if not external_rows:
            raise ValueError("External FA trial build produced no rows.")
        write_external_fa_manifest(external_rows, self.config.stage7.external_manifest)

        stage6_scores = score_external_fa_rows(
            model=self.model,
            extractor=self.extractor,
            config=self.config,
            rows=external_rows,
            tau=base_tau,
            output_path=self.config.stage7.stage6_external_scores,
            device=self.device,
        )

        # score_external_fa_rows writes the CSV, but Stage7 training needs the
        # in-memory scores. Re-read the score rows to exactly mirror cell 14.
        import csv

        score_rows: list[dict[str, str]] = []
        with open(stage6_scores["score_file"], newline="") as f:
            score_rows = [dict(row) for row in csv.DictReader(f)]

        external_negative_examples: list[dict[str, Any]] = []
        for row in score_rows:
            source = str(row["source_dataset"])
            accepted = str(row["accepted"]).lower() == "true"
            base_weight = _source_base_weight(source)
            weight = base_weight * (HARD_NEGATIVE_MULTIPLIER if accepted else 1.0)
            external_negative_examples.append(
                {
                    "s_c": float(row["s_c"]),
                    "s_s": float(row["s_s"]),
                    "label": 0.0,
                    "quadrant": "EXTERNAL_FA_HARD" if accepted else "EXTERNAL_FA",
                    "source_dataset": source,
                    "profile_id": row["profile_id"],
                    "stage6_prob": float(row["prob"]),
                    "weight": float(weight),
                }
            )

        rng = random.Random(EXTERNAL_NEGATIVE_SEED)
        hard_ext = [e for e in external_negative_examples if e["quadrant"] == "EXTERNAL_FA_HARD"]
        easy_ext = [e for e in external_negative_examples if e["quadrant"] == "EXTERNAL_FA"]
        rng.shuffle(easy_ext)
        easy_ext = easy_ext[: self.config.stage7.max_easy_external_negatives]

        stage7_examples = list(internal_examples)
        stage7_examples += hard_ext
        stage7_examples += easy_ext
        for example in stage7_examples:
            if "weight" not in example:
                example["weight"] = _default_internal_weight(str(example["quadrant"]))

        rng.shuffle(stage7_examples)
        pos = sum(1 for e in stage7_examples if float(e["label"]) == 1.0)
        neg = sum(1 for e in stage7_examples if float(e["label"]) == 0.0)
        if pos <= 0 or neg <= 0:
            raise ValueError(
                f"Stage7 needs positive and negative examples, got pos={pos}, neg={neg}."
            )
        if not hard_ext:
            raise ValueError(
                "No hard external negatives were found. The notebook only allowed "
                "continuing here when Stage6 already had zero external FA."
            )
        return stage7_examples, profiles, external_rows, len(hard_ext), len(easy_ext)

    def go_no_go_check(self, metrics: MetricsDict) -> tuple[bool, bool]:
        if is_smoke(self.config):
            metrics["gate/smoke_only"] = 1.0
            return True, True
        demo = bool(metrics.get("dev/stage7_demo_passed", 0.0) >= 1.0)
        prod = bool(metrics.get("dev/stage7_gate_passed", 0.0) >= 1.0)
        if not demo:
            raise RuntimeError("stage7_fa_tuning: MIN gate failed; demo gate did not pass.")
        return demo, prod

    def run(self) -> Path:
        self.prepare_data()
        model = self.build_model()
        assert isinstance(model, SoloSpeakModel)
        base_tau = self._base_tau()
        self.config.fusion.tau_on = base_tau
        if is_smoke(self.config):
            return self._smoke_checkpoint()

        (
            stage7_examples,
            profiles,
            external_rows,
            hard_ext_count,
            easy_ext_count,
        ) = self._build_training_examples(base_tau)

        rng = random.Random(VAL_SPLIT_SEED)
        rng.shuffle(stage7_examples)
        n_val = max(1, int(len(stage7_examples) * VAL_FRAC))
        val_examples = stage7_examples[:n_val]
        train_examples = stage7_examples[n_val:]

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("Stage7 found no trainable fusion parameters.")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.training.lr or STAGE7_LR,
            weight_decay=self.config.training.weight_decay or STAGE7_WEIGHT_DECAY,
        )

        x_train, y_train, w_train, _, _ = tensorize_examples(train_examples)
        pos_count = float((y_train == 1).sum().item())
        neg_count = float((y_train == 0).sum().item())
        pos_weight = torch.tensor(
            [max(1.0, neg_count / max(pos_count, 1.0)) * 0.35],
            dtype=torch.float32,
            device=self.device,
        )

        best_state: dict[str, torch.Tensor] | None = None
        best_val: dict[str, float] | None = None
        best_tau: float | None = None
        best_epoch = 0
        best_score = -1e9
        no_improve = 0
        indices = np.arange(len(train_examples))
        epochs = self.config.training.num_epochs or STAGE7_EPOCHS
        batch_size = self.config.training.batch_size or STAGE7_BATCH_SIZE

        for epoch in range(1, epochs + 1):
            model.train()
            np.random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                idx = indices[start : start + batch_size]
                xb = x_train[idx].to(self.device)
                yb = y_train[idx].to(self.device)
                wb = w_train[idx].to(self.device)

                optimizer.zero_grad(set_to_none=True)
                probs = model.forward_fusion(xb[:, 0], xb[:, 1]).view(-1).clamp(1e-6, 1.0 - 1e-6)
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

            val_metrics, _ = eval_stage7_examples(model, val_examples, device=self.device)
            score = val_metrics["score"]
            if score > best_score:
                best_score = score
                best_val = val_metrics
                best_tau = val_metrics["tau"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= STAGE7_PATIENCE:
                break

        if best_state is None or best_tau is None or best_val is None:
            raise RuntimeError("Stage7 training did not produce a best checkpoint.")
        model.load_state_dict(best_state, strict=False)
        val_metrics, _ = eval_stage7_examples(model, val_examples, device=self.device, tau=best_tau)
        all_train_metrics, _ = eval_stage7_examples(
            model, stage7_examples, device=self.device, tau=best_tau
        )
        external_summary = score_external_fa_rows(
            model=model,
            extractor=self.extractor,
            config=self.config,
            rows=external_rows,
            tau=best_tau,
            output_path=self.config.stage7.stage7_external_scores,
            device=self.device,
        )
        self.config.stage7.external_summary.parent.mkdir(parents=True, exist_ok=True)
        self.config.stage7.external_summary.write_text(
            json.dumps(external_summary, indent=2, sort_keys=True) + "\n"
        )

        stage7_gate_passed = (
            val_metrics["ta_clean"] >= 0.90
            and val_metrics["q2_rejection"] >= 0.95
            and val_metrics["q3_rejection"] >= 0.90
            and val_metrics["q4_rejection"] >= 0.99
            and val_metrics["external_fa_rate"] <= 0.005
        )
        stage7_demo_passed = (
            val_metrics["ta_clean"] >= 0.90
            and val_metrics["q2_rejection"] >= 0.90
            and val_metrics["q3_rejection"] >= 0.90
            and val_metrics["q4_rejection"] >= 0.95
            and val_metrics["external_fa_rate"] <= 0.01
        )

        metrics: MetricsDict = {
            "dev/best_epoch": float(best_epoch),
            "dev/tau_stage7": float(best_tau),
            "dev/stage7_gate_passed": float(stage7_gate_passed),
            "dev/stage7_demo_passed": float(stage7_demo_passed),
            "dev/ta_clean": float(val_metrics["ta_clean"]),
            "dev/q2_rejection": float(val_metrics["q2_rejection"]),
            "dev/q3_rejection": float(val_metrics["q3_rejection"]),
            "dev/q4_rejection": float(val_metrics["q4_rejection"]),
            "dev/external_fa_rate": float(val_metrics["external_fa_rate"]),
            "dev/external_rejection": float(val_metrics["external_rejection"]),
            "dev/quadrant_accuracy_min": float(val_metrics["quadrant_accuracy_min"]),
            "dev/quadrant_accuracy_mean": float(val_metrics["quadrant_accuracy_mean"]),
            "dev/accepted_rate": float(val_metrics["accepted_rate"]),
            "dev/mean_prob": float(val_metrics["mean_prob"]),
            "train_all/ta_clean": float(all_train_metrics["ta_clean"]),
            "train_all/q2_rejection": float(all_train_metrics["q2_rejection"]),
            "train_all/q3_rejection": float(all_train_metrics["q3_rejection"]),
            "train_all/q4_rejection": float(all_train_metrics["q4_rejection"]),
            "train_all/external_fa_rate": float(all_train_metrics["external_fa_rate"]),
            "train_all/quadrant_accuracy_min": float(
                all_train_metrics["quadrant_accuracy_min"]
            ),
            "external/stage7_false_accepts": float(
                external_summary["overall"]["false_accepts"]
            ),
            "external/stage7_trials": float(external_summary["overall"]["rows"]),
            "external/stage7_fa_rate": float(external_summary["overall"]["fa_rate"]),
        }

        # Preserve the notebook behavior of writing the artifact with diagnostics.
        path = save_checkpoint(
            self.config.stage7.stage7_checkpoint,
            stage_origin=self.stage_id,
            config=self.config,
            step=int(best_epoch),
            metrics=metrics,
            model_state=model.state_dict(),
            optimizer_state={},
            extra={
                "source_stage6_checkpoint": str(self.config.training.resume_from or ""),
                "base_tau": float(base_tau),
                "stage7_tau": float(best_tau),
                "wake_word": self.config.stage7.wake_word,
                "n_enroll": self.config.stage7.n_enroll,
                "profiles": len(profiles),
                "external_fa_summary": external_summary,
                "stage7_gate_passed": bool(stage7_gate_passed),
                "stage7_demo_passed": bool(stage7_demo_passed),
                "hard_external_negatives": hard_ext_count,
                "easy_external_negatives_used": easy_ext_count,
                "trainable_params": sorted(STAGE7_TRAINABLE_FUSION_PARAMS),
                "counts_by_quadrant": dict(Counter(e["quadrant"] for e in stage7_examples)),
                "counts_by_source": dict(
                    Counter(e.get("source_dataset", "unknown") for e in stage7_examples)
                ),
            },
        )
        self.go_no_go_check(metrics)
        return path
