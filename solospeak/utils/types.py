"""Type definitions used across SoloSpeak modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray

AudioWaveform: TypeAlias = torch.Tensor
MelSpectrogram: TypeAlias = torch.Tensor
Embedding128: TypeAlias = torch.Tensor
FloatArray: TypeAlias = NDArray[np.float32]

DatasetValue: TypeAlias = torch.Tensor | NDArray[Any] | str | int | float | bool | None
DatasetItem: TypeAlias = dict[str, DatasetValue]
BatchValue: TypeAlias = torch.Tensor | list[str] | list[int] | list[float] | list[bool]
BatchDict: TypeAlias = dict[str, BatchValue]
StageBatch: TypeAlias = dict[str, BatchDict]
LossDict: TypeAlias = dict[str, torch.Tensor]
MetricsDict: TypeAlias = dict[str, float]
SubgroupDict: TypeAlias = dict[str, dict[str, float] | None]
AugRanges: TypeAlias = dict[str, tuple[int, int] | tuple[float, float]]
ProbeBaseline: TypeAlias = dict[str, float]
ProbeResult: TypeAlias = dict[str, float | None]
XRTResult: TypeAlias = dict[str, float | str]
AblationResult: TypeAlias = dict[str, dict[str, float | int | str]]

BackboneVariant = Literal["bcresnet1", "bcresnet5", "bcresnet8", "bcresnet10", "bcresnet16"]
QuadrantLabel = Literal["Q1_accept", "Q2_imposter", "Q3_wrong_word", "Q4_background"]

WORD_IGNORE_INDEX: int = -100


@dataclass(frozen=True)
class UserProfile:
    """On-device user profile. Serialized to about 1.1 KB per user."""

    user_id: str
    keyword_text: str
    content_template: FloatArray
    speaker_template: FloatArray
    tau: float
    model_version: str


@dataclass
class WakeEvent:
    user_id: str
    keyword_text: str
    timestamp_ms: int
    fusion_score: float
    content_score: float
    speaker_score: float


@dataclass
class KPIResult:
    ta_clean: float
    ta_noisy: dict[int, float]
    ta_noisy_macro: float
    distance_ta: dict[float, float]
    fa_per_hour_per_user: float
    fa_per_hour_device: dict[int, float]
    q2_rejection: float
    q3_rejection: float
    q3_rejection_real: float
    q3_rejection_synth: float
    q4_rejection: float
    param_count: int
    xrt_fp32: float
    xrt_int8: float
    seed: int
    num_eval_samples: int
    per_demographic: SubgroupDict
