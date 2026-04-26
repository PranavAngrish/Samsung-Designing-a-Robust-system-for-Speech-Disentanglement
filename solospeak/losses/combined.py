"""Stage-aware weighted loss aggregator.

Stage 1: CE only
Stage 2: SupCon_c + SupCon_s + aux CE
Stage 3: + Orthogonality + Adversarial (ramped)
Stage 4: same as Stage 3 with heavy augmentation
Stage 5: binary CE on fusion head (isolated)
Stage 6: same as Stage 3 with QAT-aware forward
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from solospeak.losses.adversarial import adversarial_lambda
from solospeak.losses.orthogonality import orthogonality_loss
from solospeak.losses.supcon import SupConLoss
from solospeak.utils.config import LossWeights
from solospeak.utils.types import BatchDict, BatchValue, WORD_IGNORE_INDEX


@dataclass
class LossComponents:
    total: torch.Tensor
    supcon_content: float = 0.0
    supcon_speaker: float = 0.0
    orthogonality: float = 0.0
    adversarial: float = 0.0
    ce_aux: float = 0.0
    ce_fusion: float = 0.0
    distill: float = 0.0


OutputStream = Mapping[str, torch.Tensor]


def _require_tensor(batch: BatchDict, key: str) -> torch.Tensor:
    value: BatchValue = batch[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor batch field {key!r}, got {type(value).__name__}")
    return value


def _first_tensor(stream: OutputStream, *keys: str) -> torch.Tensor:
    for key in keys:
        if key in stream:
            return stream[key]
    raise KeyError(f"Missing any of output keys: {keys}")


class CombinedLoss(nn.Module):
    """Stage-aware loss aggregator over logical data streams."""

    def __init__(
        self,
        stage: int,
        weights: LossWeights,
        supcon_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.stage = stage
        self.weights = weights
        self.supcon = SupConLoss(supcon_temperature)

    def forward(
        self,
        outputs: Mapping[str, OutputStream],
        batches: Mapping[str, BatchDict],
        step: int = 0,
    ) -> LossComponents:
        if self.stage == 1:
            gsc_out = outputs["gsc"]
            gsc_batch = batches["gsc"]
            logits = _first_tensor(gsc_out, "logits", "gsc_logits")
            labels = _require_tensor(gsc_batch, "keyword_label").long()
            return compute_stage_loss(
                self.stage,
                step,
                self.weights,
                ce_backbone=F.cross_entropy(logits, labels),
            )

        if self.stage == 5:
            quadrant_out = outputs["quadrant"]
            quadrant_batch = batches["quadrant"]
            labels = _require_tensor(quadrant_batch, "quadrant_label").float()
            if "fusion_logits" in quadrant_out:
                ce_fusion = F.binary_cross_entropy_with_logits(
                    quadrant_out["fusion_logits"].squeeze(-1),
                    labels,
                )
            else:
                probs = _first_tensor(quadrant_out, "fusion_prob", "prob", "probs").squeeze(-1)
                ce_fusion = F.binary_cross_entropy(probs, labels)
            return compute_stage_loss(self.stage, step, self.weights, ce_fusion=ce_fusion)

        if self.stage == 6 and "distill" in outputs:
            distill_out = outputs["distill"]
            if "student_z_c" in distill_out and "teacher_z_c" in distill_out:
                distill_loss = F.mse_loss(
                    distill_out["student_z_c"],
                    distill_out["teacher_z_c"].detach(),
                )
                distill_loss = distill_loss + F.mse_loss(
                    distill_out["student_z_s"],
                    distill_out["teacher_z_s"].detach(),
                )
            else:
                student = _first_tensor(distill_out, "student", "student_logits")
                teacher = _first_tensor(distill_out, "teacher", "teacher_logits").detach()
                distill_loss = F.mse_loss(student, teacher)

            total = distill_loss
            ce_fusion_value = 0.0
            if "quadrant" in outputs and "quadrant" in batches:
                quadrant_out = outputs["quadrant"]
                quadrant_batch = batches["quadrant"]
                labels = _require_tensor(quadrant_batch, "quadrant_label").float()
                probs = _first_tensor(quadrant_out, "fusion_prob", "prob", "probs").squeeze(-1)
                ce_fusion = F.binary_cross_entropy(probs, labels)
                total = total + ce_fusion
                ce_fusion_value = ce_fusion.item()
            return LossComponents(
                total=total,
                ce_fusion=ce_fusion_value,
                distill=distill_loss.item(),
            )

        content_out = outputs["content"]
        speaker_out = outputs["speaker"]
        content_batch = batches["content"]
        speaker_batch = batches["speaker"]

        content_word_labels = _require_tensor(content_batch, "keyword_label").long()
        speaker_labels = _require_tensor(speaker_batch, "speaker_label").long()
        z_c_content = _first_tensor(content_out, "z_c", "content_embedding")
        z_s_speaker = _first_tensor(speaker_out, "z_s", "speaker_embedding")

        supcon_c = self.supcon(z_c_content, content_word_labels)
        supcon_s = self.supcon(z_s_speaker, speaker_labels)

        ce_aux = None
        if "word_logits" in content_out and "speaker_logits" in speaker_out:
            ce_word = F.cross_entropy(
                content_out["word_logits"],
                content_word_labels,
                ignore_index=WORD_IGNORE_INDEX,
            )
            ce_speaker = F.cross_entropy(speaker_out["speaker_logits"], speaker_labels)
            ce_aux = ce_word + ce_speaker

        ortho = None
        adv = None
        if self.stage >= 3:
            ortho_terms = [
                orthogonality_loss(stream["z_c"], stream["z_s"])
                for stream in (content_out, speaker_out)
                if "z_c" in stream and "z_s" in stream
            ]
            if ortho_terms:
                ortho = torch.stack(ortho_terms).mean()

            adv_terms: list[torch.Tensor] = []
            if "speaker_from_content_logits" in speaker_out:
                adv_terms.append(
                    F.cross_entropy(speaker_out["speaker_from_content_logits"], speaker_labels)
                )
            if "word_from_speaker_logits" in content_out:
                adv_terms.append(
                    F.cross_entropy(
                        content_out["word_from_speaker_logits"],
                        content_word_labels,
                        ignore_index=WORD_IGNORE_INDEX,
                    )
                )
            if adv_terms:
                adv = torch.stack(adv_terms).sum()

        return compute_stage_loss(
            self.stage,
            step,
            self.weights,
            supcon_c=supcon_c,
            supcon_s=supcon_s,
            ortho=ortho,
            adv=adv,
            ce_aux=ce_aux,
        )


def compute_stage_loss(
    stage: int,
    step: int,
    weights: LossWeights,
    *,
    supcon_c: torch.Tensor | None = None,
    supcon_s: torch.Tensor | None = None,
    ortho: torch.Tensor | None = None,
    adv: torch.Tensor | None = None,
    ce_aux: torch.Tensor | None = None,
    ce_fusion: torch.Tensor | None = None,
    ce_backbone: torch.Tensor | None = None,
) -> LossComponents:
    """Aggregate losses according to the active stage."""
    _zero = torch.tensor(0.0)

    if stage == 1:
        assert ce_backbone is not None
        return LossComponents(total=ce_backbone)

    if stage == 5:
        assert ce_fusion is not None
        return LossComponents(total=ce_fusion, ce_fusion=ce_fusion.item())

    # Stages 2–4, 6
    assert supcon_c is not None and supcon_s is not None
    loss = weights.supcon_content * supcon_c + weights.supcon_speaker * supcon_s
    components = LossComponents(
        total=_zero,
        supcon_content=supcon_c.item(),
        supcon_speaker=supcon_s.item(),
    )

    if ce_aux is not None:
        loss = loss + weights.ce_aux * ce_aux
        components.ce_aux = ce_aux.item()

    if stage >= 3:
        if ortho is not None:
            loss = loss + weights.orthogonality * ortho
            components.orthogonality = ortho.item()
        if adv is not None:
            lam = adversarial_lambda(step, weights.adversarial_ramp_steps, weights.adversarial)
            loss = loss + lam * adv
            components.adversarial = adv.item()

    components.total = loss
    return components
