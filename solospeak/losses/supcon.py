"""Supervised Contrastive Loss (Khosla et al. 2020).

For a batch of embeddings z ∈ R^(B, D) (L2-normalized) and labels y ∈ {0,...,K}^B:
    L = -1/|P(i)| * sum_{p ∈ P(i)} log( exp(z_i · z_p / τ) / sum_{a≠i} exp(z_i · z_a / τ) )
where P(i) = {j ≠ i : y_j = y_i}.

Used twice:
    Content SupCon: positives share the same keyword class, ignoring speaker identity.
    Speaker SupCon: positives share the same speaker ID, ignoring keyword content.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised contrastive loss.

    📋 CONTRACT
        embeddings: (B, D), already L2-normalized
        labels:     (B,) long tensor — class indices
        returns:    scalar loss
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        valid = labels >= 0
        embeddings = embeddings[valid]
        labels = labels[valid]
        bsz = embeddings.shape[0]
        if bsz <= 1:
            warnings.warn("SupConLoss has no anchors with positives.", stacklevel=2)
            return embeddings.sum() * 0.0

        device = embeddings.device

        sim = embeddings @ embeddings.T / self.temperature
        not_self = ~torch.eye(bsz, dtype=torch.bool, device=device)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & not_self
        pos_counts = pos_mask.sum(dim=1)
        valid_anchors = pos_counts > 0

        missing_frac = 1.0 - valid_anchors.float().mean().item()
        if missing_frac > 0.25:
            warnings.warn(
                f"SupConLoss: {missing_frac:.1%} of anchors have no positives.",
                stacklevel=2,
            )
        if not bool(valid_anchors.any()):
            return embeddings.sum() * 0.0

        sim = sim.masked_fill(~not_self, float("-inf"))
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        log_prob = torch.where(not_self, log_prob, torch.zeros_like(log_prob))
        anchor_loss = -(pos_mask.float() * log_prob).sum(dim=1) / pos_counts.clamp(min=1)
        return anchor_loss[valid_anchors].mean()
