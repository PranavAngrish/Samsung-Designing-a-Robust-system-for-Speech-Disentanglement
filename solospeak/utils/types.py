"""Type definitions used across SoloSpeak modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# Tensor shape aliases (for documentation only — not enforced at runtime)
# ---------------------------------------------------------------------------

# AudioWaveform: shape (B, T) or (T,), float32, range [-1, 1]
# MelSpectrogram: shape (B, 1, n_mels, T'), float32
# Embedding128: shape (B, 128), float32, L2-normalized
AudioWaveform: TypeAlias = "torch.Tensor"
MelSpectrogram: TypeAlias = "torch.Tensor"
Embedding128: TypeAlias = "torch.Tensor"

# ---------------------------------------------------------------------------
# Backbone variants — must match BC-ResNet ladder from architecture
# ---------------------------------------------------------------------------

BackboneVariant = Literal[
    "bcresnet1", "bcresnet5", "bcresnet8", "bcresnet10", "bcresnet16", "matchboxnet"
]

# ---------------------------------------------------------------------------
# 4-class decision matrix labels
# ---------------------------------------------------------------------------

QuadrantLabel = Literal["Q1_accept", "Q2_imposter", "Q3_wrong_word", "Q4_background"]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UserProfile:
    """On-device user profile. Serialised to ~1.1 KB per user."""

    user_id: str
    keyword_text: str
    content_template: np.ndarray  # shape (128,), float32, L2-normalized
    speaker_template: np.ndarray  # shape (128,), float32, L2-normalized
    tau: float  # decision threshold
    model_version: str  # e.g. "solospeak-v1.0.0"

    def __post_init__(self) -> None:
        assert self.content_template.shape == (128,), "content_template must be (128,)"
        assert self.speaker_template.shape == (128,), "speaker_template must be (128,)"
        assert 0.0 < self.tau < 1.0, "tau must be in (0, 1)"


@dataclass
class WakeEvent:
    """Single detection event from the streaming inference loop."""

    user_id: str
    keyword_text: str
    timestamp_ms: int
    fusion_score: float
    content_score: float
    speaker_score: float


@dataclass
class KPIResult:
    """Output of the full KPI evaluation suite."""

    ta_clean: float
    ta_noisy: dict[int, float]  # SNR (dB) → TA
    fa_per_hour: float
    q2_rejection_rate: float
    q3_rejection_rate: float
    param_count: int
    xrt_fp32: float
    xrt_int8: float
    seed: int
    num_eval_samples: int
