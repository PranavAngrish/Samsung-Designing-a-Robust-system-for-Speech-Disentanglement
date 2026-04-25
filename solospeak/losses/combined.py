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

import torch

from solospeak.losses.adversarial import adversarial_lambda
from solospeak.utils.config import LossWeights


@dataclass
class LossComponents:
    total: torch.Tensor
    supcon_content: float = 0.0
    supcon_speaker: float = 0.0
    orthogonality: float = 0.0
    adversarial: float = 0.0
    ce_aux: float = 0.0
    ce_fusion: float = 0.0


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
