"""Configuration schema. All training/eval/deploy flows read from here."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

AblationName = Literal[
    "full",
    "no_ortho",
    "no_adv",
    "no_disent",
    "no_tts_enroll",
    "no_gated_fusion",
    "no_curriculum",
]
_DEFAULT_ABLATIONS: tuple[AblationName, ...] = (
    "full",
    "no_ortho",
    "no_adv",
    "no_disent",
    "no_tts_enroll",
    "no_gated_fusion",
    "no_curriculum",
)


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 80
    fmin: float = 20.0
    fmax: float = 7600.0
    window_duration_s: float = 1.6
    window_samples: int = 25600
    window_frames: int = 160


class BackboneConfig(BaseModel):
    variant: Literal["bcresnet1", "bcresnet5", "bcresnet8", "bcresnet10", "bcresnet16"] = (
        "bcresnet8"
    )


class HeadConfig(BaseModel):
    content_dim: int = 128
    speaker_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.1


class FusionConfig(BaseModel):
    feature_names: list[str] = Field(
        default_factory=lambda: [
            "s_c",
            "s_s",
            "s_c_times_s_s",
            "abs_diff",
            "s_c_sq",
            "s_s_sq",
        ]
    )
    hidden_dims: list[int] = Field(default_factory=lambda: [20, 10])
    tau_on: float = 0.65
    tau_off: float = 0.45


class LossWeights(BaseModel):
    supcon_content: float = 1.0
    supcon_speaker: float = 1.0
    orthogonality: float = 0.1
    adversarial: float = 0.1
    ce_aux: float = 0.5
    adversarial_ramp_steps: int = 5000


class TrainingConfig(BaseModel):
    stage: Literal[1, 2, 3, 4, 5, 6, 7]
    batch_size: int = 128
    num_epochs: int = 30
    optimizer: Literal["adamw", "sgd"] = "adamw"
    lr: float = 3e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_grad_norm: float = 5.0
    precision: Literal["fp32", "amp_bf16", "amp_fp16"] = "amp_fp16"
    seed: int = 42
    num_workers: int = 4
    resume_from: Path | None = None
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_every_steps: int = 500
    upload_every_steps: int = 2000
    n_gsc_classes: int | None = None
    n_aux_word_classes: int | None = None
    n_aux_speaker_classes: int | None = None


class DataConfig(BaseModel):
    root: Path = Path("data/processed")
    manifests_dir: Path = Path("data/manifests")
    snr_range_db: tuple[int, int] = (-5, 30)
    distance_range_m: tuple[float, float] = (0.5, 5.0)
    augmentation_prob: float = 0.8
    hard_neg_fraction: float = 0.3
    external_common_voice_roots: list[Path] = Field(default_factory=list)
    external_librispeech_roots: list[Path] = Field(default_factory=list)
    external_background_noise_roots: list[Path] = Field(default_factory=list)
    external_urbansound_roots: list[Path] = Field(default_factory=list)
    external_fa_max_total_trials: int = 40000


class Stage4DConfig(BaseModel):
    """Notebook-faithful Stage 4D hard-Q2 mining settings."""

    wake_word: str = "zero"
    n_enroll: int = 2
    max_eval_profiles: int = 70
    output_checkpoint: Path = Path("checkpoints/stage4d_hardq2_mining_balanced.pt")
    last_checkpoint: Path = Path("checkpoints/stage4d_hardq2_last.pt")
    production_checkpoint: Path = Path("checkpoints/stage4d_hardq2_production.pt")
    results_json: Path = Path("reports/stage4d_hardq2_results.json")
    replace_stage4_robust_on_accept: bool = False
    hard_q2_mining: bool = True
    mining_rounds: int = 2
    q1_per_profile: int = 2
    q2_per_profile: int = 6
    q3_per_profile: int = 2
    q4_per_profile: int = 2


class Stage5Config(BaseModel):
    """Notebook-winning Stage 5 dual-candidate fusion search settings."""

    reproduce_winner_only: bool = False
    stage4_candidate_name: str = "stage4d_hardq2_mining_balanced"
    stage4_candidate_checkpoint: Path = Path("checkpoints/stage4d_hardq2_mining_balanced.pt")
    data_variant_name: str = "zero_e3_product"
    wake_word: str = "zero"
    n_enroll: int = 3
    q1_per_profile: int = 2
    q2_per_profile: int = 4
    q3_per_profile: int = 2
    q4_per_profile: int = 2
    fusion_variant_name: str = "q2_very_strong"
    lr: float = 0.0006
    epochs: int = 700
    patience: int = 100
    batch_size: int = 256
    weight_decay: float = 0.0001
    q1_weight: float = 1.2
    q2_weight: float = 4.0
    q3_weight: float = 0.2
    q4_weight: float = 0.2
    pos_weight_scale: float = 0.8


class Stage6Config(BaseModel):
    """Final production evaluation/export settings. Stage 6 is not QAT."""

    test_kpi_manifest: Path = Path("data/manifests/test_kpi.csv")
    test_fa_manifest: Path = Path("data/manifests/test_fa.csv")
    output_checkpoint: Path = Path("checkpoints/stage6_final.pt")
    compatibility_checkpoint: Path = Path("checkpoints/stage6_final.pt")
    deployable_path: Path = Path("exports/solospeak_stage6_deployable.pt")
    metrics_json: Path = Path("reports/stage6_final_metrics.json")
    summary_json: Path = Path("reports/stage6_summary.json")
    test_scores_csv: Path = Path("reports/stage6_test_scores.csv")
    fa_scores_csv: Path = Path("reports/stage6_fa_scores.csv")
    threshold_diagnostic_csv: Path = Path("reports/stage6_threshold_diagnostic.csv")
    evaluate_noisy_q1: bool = True
    noisy_snrs: list[int] = Field(default_factory=lambda: [-5, 0, 5, 10, 15, 20, 25, 30])
    recovery_fallback_to_last: bool = False


class Stage7Config(BaseModel):
    """External false-accept hard-negative tuning and corrected export settings."""

    wake_word: str = "zero"
    n_enroll: int = 3
    max_profiles: int = 300
    max_external_fa_trials: int = 40000
    max_easy_external_negatives: int = 12000
    stage7_checkpoint: Path = Path("checkpoints/stage7_fusion.pt")
    final_checkpoint: Path = Path("checkpoints/stage7_final_corrected.pt")
    deployable_path: Path = Path("exports/solospeak_stage7_deployable_corrected.pt")
    external_manifest: Path = Path("reports/stage7_external_fa_manifest.csv")
    stage6_external_scores: Path = Path("reports/stage7_external_fa_stage6_scores.csv")
    stage7_external_scores: Path = Path("reports/stage7_external_fa_scores.csv")
    external_summary: Path = Path("reports/stage7_external_fa_summary.json")
    internal_scores: Path = Path("reports/stage7_final_internal_scores.csv")
    joint_calibration_summary: Path = Path(
        "reports/stage7_joint_threshold_calibration_summary.json"
    )
    final_kpi_verification: Path = Path("reports/stage7_final_kpi_verification.json")
    external_fa_rate_max: float = 0.005
    common_voice_fa_rate_max: float = 0.005
    q2_rejection_min: float = 0.95
    ta_clean_min: float = 0.92


class SoloSpeakConfig(BaseModel):
    """Top-level config. Every training script takes one of these."""

    run_name: str
    audio: AudioConfig = Field(default_factory=AudioConfig)
    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    heads: HeadConfig = Field(default_factory=HeadConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    losses: LossWeights = Field(default_factory=LossWeights)
    training: TrainingConfig
    data: DataConfig = Field(default_factory=DataConfig)
    stage4d: Stage4DConfig = Field(default_factory=Stage4DConfig)
    stage5: Stage5Config = Field(default_factory=Stage5Config)
    stage6: Stage6Config = Field(default_factory=Stage6Config)
    stage7: Stage7Config = Field(default_factory=Stage7Config)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "SoloSpeakConfig":
        data = _load_yaml_with_base(Path(path))
        return cls.model_validate(data)

    def to_yaml(self, path: Path | str) -> None:
        with open(Path(path), "w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override onto base; override wins on scalar/list values."""
    out = dict(base)
    for key, value in override.items():
        if key == "__base__":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml_with_base(path: Path) -> dict[str, Any]:
    """Load YAML with optional '__base__: relative/path.yaml' inheritance."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    base_ref = data.get("__base__")
    if base_ref is None:
        return data
    base_path = (path.parent / base_ref).resolve()
    return _deep_merge(_load_yaml_with_base(base_path), data)


ConfigT = TypeVar("ConfigT", bound="_YamlConfig")


class _YamlConfig(BaseModel):
    """Shared YAML loader for non-training config models."""

    @classmethod
    def from_yaml(cls: type[ConfigT], path: Path | str) -> ConfigT:
        data = _load_yaml_with_base(Path(path))
        return cls.model_validate(data)

    def to_yaml(self, path: Path | str) -> None:
        with open(Path(path), "w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)


@dataclass(frozen=True)
class EvalSet:
    """Resolved eval manifests and any loaded calibration metadata."""

    test_kpi_manifest: Path
    test_fa_manifest: Path
    keyword_vocab: Path = field(default_factory=lambda: Path("data/manifests/keyword_vocab.json"))
    speaker_vocab: Path = field(default_factory=lambda: Path("data/manifests/speaker_vocab.json"))


@dataclass(frozen=True)
class ValidationResult:
    name: str
    passed: bool
    value: float | str
    threshold: float | str
    message: str = ""


@dataclass
class ValidationReport:
    results: list[ValidationResult]

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)


class ArtifactValidationError(RuntimeError):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__("artifact validation failed")


class EvalConfig(_YamlConfig):
    test_manifests: list[Path] = Field(
        default_factory=lambda: [
            Path("data/manifests/test_kpi.csv"),
            Path("data/manifests/test_fa.csv"),
        ]
    )
    noise_snrs_db: list[int] = Field(default_factory=lambda: [-5, 0, 5, 10, 15, 20, 25, 30])
    distance_buckets_m: list[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0, 3.5, 5.0])
    fa_audio_hours: float = 10.0
    fa_n_user_buckets: list[int] = Field(default_factory=lambda: [1, 4, 8])
    n_seeds: int = 3
    subgroups_optional: list[str] = Field(default_factory=lambda: ["gender", "age", "accent"])
    subgroups_mandatory: list[str] = Field(default_factory=lambda: ["keyword_syllable_count"])
    q3_neighbors_per_keyword: int = 5
    q3_max_edit_distance: int = 2
    q3_synthesis_fallback: bool = True


class AblationConfig(_YamlConfig):
    ablations: list[AblationName] = Field(default_factory=lambda: list(_DEFAULT_ABLATIONS))
    seeds: list[int] = Field(default_factory=lambda: [42])
    extra_seeds_for: list[str] = Field(default_factory=lambda: ["full", "no_disent"])
    extra_seeds: list[int] = Field(default_factory=lambda: [137, 2718])
    base_checkpoint: Path = Path("checkpoints/stage2_dualhead.pt")
    max_parallel: int = 1


class DeploymentConfig(_YamlConfig):
    target_platform: Literal["arm64_android", "arm64_linux", "x86_64_linux"] = "arm64_android"
    xrt_target: float = 0.08
    onnx_opset: int = 17
    fixed_time_dim: int = 160
    streaming_hop_ms: int = 160
    calibration_num_samples: int = 500
    quant_path: Literal["qat_then_ort_ptq", "qat_only", "ptq_only"] = "qat_then_ort_ptq"
    fuse_bn_into_conv: bool = True


class ValidationGates(_YamlConfig):
    """Hard ship-gates for the deployed artifact."""

    max_filesize_mb: float = 5.0
    min_opset: int = 17
    max_xrt: float = 0.20
    min_ta_clean: float = 0.92
    min_ta_noisy_macro: float = 0.80
    max_fa_per_hr_per_user: float = 2.0
    max_param_count: int = 3_000_000
    max_int8_vs_fp32_degradation_pp: float = 1.0


def load_eval_config_bundle(path: Path | str) -> tuple[EvalConfig, ValidationGates]:
    """Load eval protocol settings plus hard validation gates from one YAML file."""
    data = dict(_load_yaml_with_base(Path(path)))
    gates_data = data.pop("validation_gates", {}) or {}
    return EvalConfig.model_validate(data), ValidationGates.model_validate(gates_data)
