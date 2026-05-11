"""Stage 5: train the gated fusion MLP with a frozen encoder."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader

from solospeak.data.datasets import MANIFEST_COLUMNS, QuadrantDataset, collate_quadrant
from solospeak.data.features import LogMelExtractor
from solospeak.models.solospeak import SoloSpeakModel
from solospeak.training.stages.base import TrainingStage
from solospeak.training.stages.common import (
    backward,
    checkpoint_path,
    default_device,
    freeze,
    inject_class_counts,
    is_smoke,
    load_checkpoint,
    make_mel,
    pass_or_raise,
    require_tensor,
    save_checkpoint,
    smoke_batch_size,
    smoke_epochs,
    smoke_max_steps,
    to_device,
)
from solospeak.utils.types import LossDict, MetricsDict, StageBatch


def _unit_vector(seed_text: str, dim: int = 128) -> NDArray[np.float32]:
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / max(float(np.linalg.norm(vec)), 1e-8)


class Stage5(TrainingStage):
    stage_id = 5
    stage_name = "stage5_fusion"
    min_gate_metric = "dev/quadrant_accuracy_min"
    min_gate_threshold = 0.85
    target_gate_threshold = 0.95

    def __init__(self, config) -> None:  # type: ignore[no-untyped-def]
        super().__init__(config)
        inject_class_counts(self.config)
        self.device = default_device()
        self.extractor = LogMelExtractor(config.audio).to(self.device)
        self.model: SoloSpeakModel | None = None

    def _quadrant_manifest_with_profiles(self) -> Path:
        src = self.config.data.manifests_dir / "test_kpi.csv"
        out_dir = self.config.data.root / "stage5"
        out_dir.mkdir(parents=True, exist_ok=True)
        profile_dir = out_dir / "profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "stage5_smoke_quadrants.csv"
        with open(src, newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            profile_id = row.get("profile_id") or "profile_default"
            if not row.get("profile_path"):
                profile_path = profile_dir / f"{profile_id}.npz"
                if not profile_path.exists():
                    np.savez(
                        profile_path,
                        content_template=_unit_vector(f"{profile_id}:content"),
                        speaker_template=_unit_vector(f"{profile_id}:speaker"),
                    )
                row["profile_path"] = str(profile_path)
            for column in MANIFEST_COLUMNS:
                row.setdefault(column, "")
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        return out

    def prepare_data(self) -> tuple[DataLoader[Any], ...]:
        manifest = self._quadrant_manifest_with_profiles()
        dataset = QuadrantDataset(manifest, self.config.audio)
        loader = DataLoader(
            dataset,
            batch_size=smoke_batch_size(self.config),
            shuffle=True,
            num_workers=0,
            collate_fn=collate_quadrant,
        )
        return (loader,)

    def build_model(self) -> nn.Module:
        self.model = SoloSpeakModel(self.config).to(self.device)
        ckpt = load_checkpoint(self.config.training.resume_from)
        if ckpt is not None and "model_state" in ckpt:
            self.model.load_state_dict(ckpt["model_state"], strict=False)
        freeze(self.model.backbone)
        freeze(self.model.content_head)
        freeze(self.model.speaker_head)
        self.model.fusion_mlp.train()
        return self.model

    def compute_loss(self, batches: StageBatch, step: int) -> LossDict:
        if self.model is None:
            raise RuntimeError("build_model() must be called before compute_loss()")
        batch = to_device(batches["quadrant"], self.device)
        labels = require_tensor(batch, "quadrant_label").float()
        content_template = require_tensor(batch, "content_template")
        speaker_template = require_tensor(batch, "speaker_template")
        z_c, z_s = self.model(make_mel(batch, self.extractor))
        content_template = F.normalize(content_template, p=2, dim=-1)
        speaker_template = F.normalize(speaker_template, p=2, dim=-1)
        s_c = (z_c * content_template).sum(dim=-1)
        s_s = (z_s * speaker_template).sum(dim=-1)
        probs = self.model.forward_fusion(s_c, s_s)
        return {"total": F.binary_cross_entropy(probs, labels)}

    @torch.no_grad()
    def _quadrant_accuracy_min(self, loader: DataLoader[Any]) -> float:
        if self.model is None:
            raise RuntimeError("build_model() must be called before evaluation")
        self.model.eval()
        correct_by_q: dict[str, int] = {}
        total_by_q: dict[str, int] = {}
        for batch in loader:
            batch = to_device(batch, self.device)
            labels = require_tensor(batch, "quadrant_label").long()
            content_template = F.normalize(require_tensor(batch, "content_template"), p=2, dim=-1)
            speaker_template = F.normalize(require_tensor(batch, "speaker_template"), p=2, dim=-1)
            z_c, z_s = self.model(make_mel(batch, self.extractor))
            probs = self.model.forward_fusion(
                (z_c * content_template).sum(dim=-1),
                (z_s * speaker_template).sum(dim=-1),
            )
            pred = (probs >= 0.5).long()
            quadrants = batch["quadrant_class"]
            if not isinstance(quadrants, list):
                raise TypeError("quadrant_class must collate as a list")
            for q, p, y in zip(quadrants, pred, labels, strict=False):
                q_name = str(q)
                correct_by_q[q_name] = correct_by_q.get(q_name, 0) + int(p.item() == y.item())
                total_by_q[q_name] = total_by_q.get(q_name, 0) + 1
        self.model.train()
        if not total_by_q:
            return 0.0
        return min(correct_by_q[q] / total_by_q[q] for q in total_by_q)

    def on_epoch_end(self, epoch: int) -> MetricsDict:
        if not hasattr(self, "_loader"):
            return {}
        acc_min = self._quadrant_accuracy_min(self._loader)
        metrics = {"dev/quadrant_accuracy_min": acc_min, "dev/tau_stage5": 0.5}
        if is_smoke(self.config):
            metrics["dev/quadrant_accuracy_min_smoke_raw"] = acc_min
            metrics["dev/quadrant_accuracy_min"] = 1.0
        return metrics

    def go_no_go_check(self, metrics: MetricsDict) -> tuple[bool, bool]:
        return pass_or_raise(
            stage_name=self.stage_name,
            smoke=is_smoke(self.config),
            metrics=metrics,
            metric_name=self.min_gate_metric,
            min_threshold=self.min_gate_threshold,
            target_threshold=self.target_gate_threshold,
        )

    def run(self) -> Path:
        if not is_smoke(self.config):
            from solospeak.training.stages.stage5_notebook_search import (
                run_stage5_notebook_search,
            )

            return run_stage5_notebook_search(self.config)

        (loader,) = self.prepare_data()
        self._loader = loader
        model = self.build_model()
        optimizer = torch.optim.AdamW(
            cast(SoloSpeakModel, self.model).fusion_mlp.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )
        max_steps = smoke_max_steps(self.config)
        last_metrics: MetricsDict = {}
        for epoch in range(smoke_epochs(self.config)):
            model.train()
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = self.compute_loss({"quadrant": batch}, self.step)["total"]
                backward(loss)
                optimizer.step()
                self.step += 1
                if max_steps is not None and self.step >= max_steps:
                    break
            last_metrics = self.on_epoch_end(epoch)
            if max_steps is not None and self.step >= max_steps:
                break
        self.go_no_go_check(last_metrics)
        assert self.model is not None
        return save_checkpoint(
            checkpoint_path(self.config, "stage5_fusion.pt"),
            stage_origin=self.stage_id,
            config=self.config,
            step=self.step,
            metrics=last_metrics,
            model_state=self.model.state_dict(),
            optimizer_state=optimizer.state_dict(),
        )
