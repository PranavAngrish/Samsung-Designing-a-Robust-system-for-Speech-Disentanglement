"""Frobenius-norm cross-covariance penalty.

Forces E[z_c · z_s^T] → 0, decorrelating content and speaker embedding spaces.
"""

from __future__ import annotations

import torch


def orthogonality_loss(z_c: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
    """Compute cross-covariance Frobenius-norm penalty.

    📋 CONTRACT
        z_c: (B, D) content embeddings, L2-normalized
        z_s: (B, D) speaker embeddings, L2-normalized
        returns: scalar loss (0 when spaces are perfectly orthogonal)
    """
    B = z_c.shape[0]
    z_c_c = z_c - z_c.mean(dim=0, keepdim=True)   # center
    z_s_c = z_s - z_s.mean(dim=0, keepdim=True)
    C = z_c_c.T @ z_s_c / B                        # (D, D) cross-covariance
    return (C**2).sum()
