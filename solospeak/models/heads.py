"""Dual orthogonal heads: content (what was said) and speaker (who said it).

Each head:
    Input:  backbone feature map (B, C, 1, T')
    Output: Embedding128, L2-normalized, shape (B, 128)

Architecture (identical for both heads — weights are what makes them orthogonal):
    AdaptiveAvgPool2d → (B, C)
    Linear(C, H) + GELU + Dropout
    Linear(H, 128)
    F.normalize(dim=-1)
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingHead(nn.Module):
    """Single embedding head (used for both content and speaker).

    📋 CONTRACT
        input:  (B, C, H, W) feature map from backbone
        output: (B, output_dim) L2-normalized embedding
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(input_channels, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).flatten(1)               # (B, C)
        x = self.drop(self.act(self.fc1(x)))       # (B, H)
        x = self.fc2(x)                            # (B, output_dim)
        return F.normalize(x, p=2, dim=-1)         # L2-normalized


class AuxiliaryHeads(nn.Module):
    """Training-only auxiliary classification heads. Stripped at ONNX export."""

    def __init__(
        self, embed_dim: int = 128, n_words: int = 1000, n_speakers: int = 7000
    ) -> None:
        super().__init__()
        self.aux_word = nn.Linear(embed_dim, n_words)
        self.aux_speaker = nn.Linear(embed_dim, n_speakers)

    def forward(self, z_c: torch.Tensor, z_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return cast(torch.Tensor, self.aux_word(z_c)), cast(torch.Tensor, self.aux_speaker(z_s))

    def forward_word(self, z_c: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.aux_word(z_c))

    def forward_speaker(self, z_s: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.aux_speaker(z_s))
