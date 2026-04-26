"""SoloSpeakResNet backbone.

A 4-stage residual CNN. Plain Conv2d blocks, no depthwise separable convolutions and
no broadcasted residual paths. The config aliases remain ``bcresnet*`` for continuity,
but this is a project-specific architecture rather than an implementation of the
BC-ResNet paper.

INPUT:  (B, 1, 80, T)        - log-mel, 80 mel bands
OUTPUT: (B, C_out, 1, T/8)   - frequency collapsed by AdaptiveAvgPool
"""

from __future__ import annotations

from typing import Any, TypedDict, cast

import torch
from torch import nn


class _VariantSpec(TypedDict):
    base: int
    num_blocks: list[int]


VARIANTS: dict[str, _VariantSpec] = {
    "bcresnet1": {"base": 8, "num_blocks": [1, 1, 1, 1]},
    "bcresnet5": {"base": 16, "num_blocks": [2, 2, 3, 3]},
    "bcresnet8": {"base": 24, "num_blocks": [2, 2, 3, 2]},
    "bcresnet10": {"base": 32, "num_blocks": [2, 2, 2, 2]},
    "bcresnet16": {"base": 40, "num_blocks": [2, 2, 3, 3]},
}


def _fuse_modules(module: nn.Module, modules_to_fuse: list[list[str]]) -> None:
    fuse = cast(Any, torch.ao.quantization.fuse_modules)
    fuse(module, modules_to_fuse, inplace=True)


class NormalBlock(nn.Module):
    """Two 3x3 convs with a residual connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return cast(torch.Tensor, self.relu(out + identity))

    def fuse_model(self) -> None:
        """Fuse Conv+BN pairs for export/quantization."""
        _fuse_modules(self, [["conv1", "bn1"], ["conv2", "bn2"]])


class TransitionBlock(nn.Module):
    """Halve frequency and time dimensions while changing channel count."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return cast(torch.Tensor, self.relu(out + identity))

    def fuse_model(self) -> None:
        """Fuse Conv+BN pairs for export/quantization."""
        _fuse_modules(self, [["conv1", "bn1"], ["conv2", "bn2"]])
        _fuse_modules(self.shortcut, [["0", "1"]])


class SoloSpeakResNet(nn.Module):
    """SoloSpeak residual CNN backbone.

    Stride schedule for input ``(B, 1, 80, 160)``:
        stem:  ``(B, c0, 40, 160)``
        trans1 ``(B, c1, 20, 80)``
        trans2 ``(B, c2, 10, 40)``
        trans3 ``(B, c3, 5, 20)``
        pool:  ``(B, c3, 1, 20)``
    """

    def __init__(self, variant: str = "bcresnet8") -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}. Choose from {list(VARIANTS)}.")

        spec = VARIANTS[variant]
        base = spec["base"]
        num_blocks = spec["num_blocks"]
        channels = [base * (i + 1) for i in range(4)]
        self.variant = variant
        self._output_channels = channels[-1]

        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=5, stride=(2, 1), padding=2, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        stages: list[nn.Module] = []
        for i, n_blocks in enumerate(num_blocks):
            if i > 0:
                stages.append(TransitionBlock(channels[i - 1], channels[i]))
            for _ in range(n_blocks):
                stages.append(NormalBlock(channels[i]))
        self.stages = nn.Sequential(*stages)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

    @property
    def output_channels(self) -> int:
        """Channels at the backbone output."""
        return self._output_channels

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Map ``(B, 1, 80, T)`` to ``(B, C_out, 1, T/8)``."""
        x = self.stem(mel)
        x = self.stages(x)
        return cast(torch.Tensor, self.freq_pool(x))

    def fuse_model(self) -> None:
        """Fuse all supported Conv+BN patterns before export/quantization."""
        self.eval()
        _fuse_modules(self.stem, [["0", "1", "2"]])
        for block in self.stages:
            if hasattr(block, "fuse_model"):
                block.fuse_model()


BCResNet = SoloSpeakResNet
