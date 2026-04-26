"""361-parameter MLP that converts (s_c, s_s) to accept/reject probability.

Input features (6-dim):
    s_c          — cosine similarity, content head vs template
    s_s          — cosine similarity, speaker head vs template
    s_c * s_s    — product (both-must-match indicator)
    |s_c - s_s|  — disagreement between heads
    s_c ** 2     — nonlinearity on content score
    s_s ** 2     — nonlinearity on speaker score

Architecture: Linear(6,20)+ReLU → Linear(20,10)+ReLU → Linear(10,1)+Sigmoid
Total params: 140 + 210 + 11 = 361
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from solospeak.utils.config import FusionConfig


class GatedFusionMLP(nn.Module):
    """Learned AND-like gate over (content_sim, speaker_sim).

    📋 CONTRACT
        input:  s_c (B,) and s_s (B,) — cosine similarities in [-1, 1]
        output: (B,) probability in [0, 1]
    """

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        dims = [6] + config.hidden_dims + [1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, s_c: torch.Tensor, s_s: torch.Tensor) -> torch.Tensor:
        features = torch.stack(
            [s_c, s_s, s_c * s_s, (s_c - s_s).abs(), s_c**2, s_s**2],
            dim=-1,
        )                                   # (B, 6)
        return cast(torch.Tensor, self.net(features)).squeeze(-1)  # (B,)


def naive_fusion_min(s_c: torch.Tensor, s_s: torch.Tensor) -> torch.Tensor:
    """Baseline fusion: min(s_c, s_s). Used in ablation study."""
    return torch.minimum(s_c, s_s)


def naive_fusion_product(s_c: torch.Tensor, s_s: torch.Tensor) -> torch.Tensor:
    """Baseline fusion: s_c * s_s. Used in ablation study."""
    return s_c * s_s
