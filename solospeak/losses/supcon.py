"""Supervised Contrastive Loss (Khosla et al. 2020).

For a batch of embeddings z ∈ R^(B, D) (L2-normalized) and labels y ∈ {0,...,K}^B:
    L = -1/|P(i)| * sum_{p ∈ P(i)} log( exp(z_i · z_p / τ) / sum_{a≠i} exp(z_i · z_a / τ) )
where P(i) = {j ≠ i : y_j = y_i}.

Used twice:
    Content SupCon: positives share the same keyword class, ignoring speaker identity.
    Speaker SupCon: positives share the same speaker ID, ignoring keyword content.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        B = embeddings.shape[0]
        device = embeddings.device

        # Pairwise cosine similarities — already normalized so just dot products
        sim = embeddings @ embeddings.T / self.temperature  # (B, B)

        # Mask: same label, different sample
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        not_self = ~torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = label_eq & not_self                          # (B, B)

        # For numerical stability, subtract max per row (logsumexp trick)
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim)
        exp_sim_no_self = exp_sim * not_self.float()

        log_prob = sim - torch.log(exp_sim_no_self.sum(dim=1, keepdim=True) + 1e-8)

        # Mean over positives per anchor
        n_pos = pos_mask.sum(dim=1).float().clamp(min=1)
        loss = -(pos_mask.float() * log_prob).sum(dim=1) / n_pos

        return loss.mean()
