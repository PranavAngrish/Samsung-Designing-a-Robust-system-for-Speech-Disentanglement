"""Full SoloSpeak model: backbone + content head + speaker head + fusion."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from solospeak.models.fusion import GatedFusionMLP
from solospeak.models.backbones.bcresnet import BCResNet
from solospeak.models.heads import EmbeddingHead
from solospeak.utils.config import SoloSpeakConfig


def _build_backbone(config: SoloSpeakConfig) -> BCResNet:
    """Instantiate backbone from config. Raises if variant unknown."""
    return BCResNet(config.backbone.variant)


class SoloSpeakModel(nn.Module):
    """Full model: backbone + content head + speaker head + fusion.

    📋 CONTRACT
        forward(mel):
            input:  mel MelSpectrogram of shape (B, 1, 80, T')
            output: (z_c: Embedding128, z_s: Embedding128) — both L2-normalized

        forward_fusion(s_c, s_s):
            input:  cosine similarities (B,) each
            output: fusion probability (B,)

    The fusion MLP is a separate submodule — backbone + heads are frozen in Stage 5,
    only the fusion MLP is trained.
    """

    def __init__(self, config: SoloSpeakConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = _build_backbone(config)
        channels = self.backbone.output_channels()
        self.content_head = EmbeddingHead(
            channels, config.heads.hidden_dim, config.heads.content_dim, config.heads.dropout
        )
        self.speaker_head = EmbeddingHead(
            channels, config.heads.hidden_dim, config.heads.speaker_dim, config.heads.dropout
        )
        self.fusion_mlp = GatedFusionMLP(config.fusion)

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward — returns content and speaker embeddings."""
        feat = self.backbone(mel)
        z_c = self.content_head(feat)
        z_s = self.speaker_head(feat)
        return z_c, z_s

    def forward_fusion(self, s_c: torch.Tensor, s_s: torch.Tensor) -> torch.Tensor:
        """Apply fusion head to cosine similarities."""
        return cast(torch.Tensor, self.fusion_mlp(s_c, s_s))

    def freeze_backbone_and_heads(self) -> None:
        """Freeze backbone + both heads for Stage 5 fusion training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.content_head.parameters():
            param.requires_grad = False
        for param in self.speaker_head.parameters():
            param.requires_grad = False

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts per submodule."""
        return {
            "backbone": sum(p.numel() for p in self.backbone.parameters()),
            "content_head": sum(p.numel() for p in self.content_head.parameters()),
            "speaker_head": sum(p.numel() for p in self.speaker_head.parameters()),
            "fusion": sum(p.numel() for p in self.fusion_mlp.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
