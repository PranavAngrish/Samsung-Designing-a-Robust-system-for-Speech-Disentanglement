"""Configuration schema. All training/eval/deploy flows read from here."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160       # 10 ms at 16 kHz
    win_length: int = 400       # 25 ms at 16 kHz
    n_mels: int = 80
    fmin: float = 20.0
    fmax: float = 7600.0
    window_duration_s: float = 1.5  # ring buffer length


class BackboneConfig(BaseModel):
    variant: Literal[
        "bcresnet1", "bcresnet5", "bcresnet8",
        "bcresnet10", "bcresnet16", "matchboxnet"
    ] = "bcresnet8"
    scale: float = 1.0          # BC-ResNet channel width multiplier


class HeadConfig(BaseModel):
    content_dim: int = 128
    speaker_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.1


class FusionConfig(BaseModel):
    feature_names: list[str] = Field(
        default=["s_c", "s_s", "s_c_times_s_s", "abs_diff", "s_c_sq", "s_s_sq"]
    )
    hidden_dims: list[int] = Field(default=[20, 10])
    tau_on: float = 0.65
    tau_off: float = 0.45
    mode: Literal["learned", "min", "product"] = "learned"


class LossWeights(BaseModel):
    supcon_content: float = 1.0
    supcon_speaker: float = 1.0
    orthogonality: float = 0.1
    adversarial: float = 0.1
    ce_aux: float = 0.5
    adversarial_ramp_steps: int = 5000  # linear ramp-up duration


class TrainingConfig(BaseModel):
    stage: Literal[1, 2, 3, 4, 5, 6]
    batch_size: int = 128
    num_epochs: int = 30
    optimizer: Literal["adamw", "sgd"] = "adamw"
    lr: float = 3e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_grad_norm: float = 5.0
    precision: Literal["fp32", "amp_bf16", "amp_fp16"] = "amp_bf16"
    seed: int = 42
    num_workers: int = 4
    resume_from: Path | None = None
    checkpoint_dir: Path = Path("checkpoints")


class DataConfig(BaseModel):
    root: Path = Path("data/processed")
    manifests_dir: Path = Path("data/manifests")
    snr_range_db: tuple[int, int] = (-5, 30)
    distance_range_m: tuple[float, float] = (0.5, 5.0)
    augmentation_prob: float = 0.8
    hard_neg_fraction: float = 0.3
    curriculum: bool = True


class EnrollmentConfig(BaseModel):
    tts_n_variants: int = 10    # number of Parler-TTS augmentations per keyword
    min_recordings: int = 3
    max_recordings: int = 5


class SoloSpeakConfig(BaseModel):
    """Top-level config. Every script takes one of these."""

    run_name: str
    audio: AudioConfig = AudioConfig()
    backbone: BackboneConfig = BackboneConfig()
    heads: HeadConfig = HeadConfig()
    fusion: FusionConfig = FusionConfig()
    losses: LossWeights = LossWeights()
    training: TrainingConfig
    data: DataConfig = DataConfig()
    enrollment: EnrollmentConfig = EnrollmentConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> SoloSpeakConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)
