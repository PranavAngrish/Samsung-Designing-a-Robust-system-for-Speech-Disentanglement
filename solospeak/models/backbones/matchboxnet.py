"""MatchboxNet backbone — fallback if BC-ResNet-8 fails to converge.

Reference: "MatchboxNet: 1D Time-Channel Separable Convolutional Neural Network
            Architecture for Speech Commands Recognition"
            Majumdar & Ginsburg, Interspeech 2020, NVIDIA.

Approx. 140 K parameters. Apache-2.0.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MatchboxNet(nn.Module):
    """1D Time-Channel Separable CNN for keyword spotting.

    📋 CONTRACT
        input:  mel (B, 1, 80, T)
        output: feature map (B, C, 1, T')
    """

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("Implement in Phase 2 only if BC-ResNet-8 fails GO/NO-GO")

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement in Phase 2 only if BC-ResNet-8 fails GO/NO-GO")

    def output_channels(self) -> int:
        raise NotImplementedError("Implement in Phase 2 only if BC-ResNet-8 fails GO/NO-GO")
