"""BC-ResNet parametric family.

Reference: "Broadcasted Residual Learning for Efficient Keyword Spotting"
           Kim et al., Interspeech 2021, Samsung Research.

SUPPORTED VARIANTS:
    bcresnet1  — ~80 K params   (Galaxy Watch)
    bcresnet5  — ~500 K params  (Galaxy Buds)
    bcresnet8  — ~1.0 M params  (default, flagship phone)
    bcresnet10 — ~1.5 M params  (high-end phone / tablet)
    bcresnet16 — ~3.0 M params  (server fallback)
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Channel width and depth per variant
VARIANTS: dict[str, dict[str, object]] = {
    "bcresnet1":  {"base_channels": 8,  "num_stages": [1, 1, 1, 1]},
    "bcresnet5":  {"base_channels": 20, "num_stages": [2, 2, 2, 2]},
    "bcresnet8":  {"base_channels": 32, "num_stages": [2, 2, 4, 4]},
    "bcresnet10": {"base_channels": 40, "num_stages": [2, 2, 4, 4]},
    "bcresnet16": {"base_channels": 64, "num_stages": [2, 2, 4, 4]},
}


class NormalBlock(nn.Module):
    """BC-ResNet normal block with f2-path and broadcasted residual."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        raise NotImplementedError("Implement in Phase 2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement in Phase 2")


class TransitionBlock(nn.Module):
    """BC-ResNet transition block: doubles channels, halves frequency dimension."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        raise NotImplementedError("Implement in Phase 2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement in Phase 2")


class BCResNet(nn.Module):
    """Broadcasted Residual Network for keyword spotting.

    📋 CONTRACT
        input:  mel (B, 1, 80, T)
        output: feature map (B, C, 1, T') where T' ≈ T/8
    """

    def __init__(self, variant: str = "bcresnet8") -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from {list(VARIANTS)}")
        self.variant = variant
        self.cfg = VARIANTS[variant]
        raise NotImplementedError("Implement in Phase 2")

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """(B, 1, 80, T) → (B, C, 1, T')"""
        raise NotImplementedError("Implement in Phase 2")

    def output_channels(self) -> int:
        """Return the number of output channels for this variant."""
        raise NotImplementedError("Implement in Phase 2")
