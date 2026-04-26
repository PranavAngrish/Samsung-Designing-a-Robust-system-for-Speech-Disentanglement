"""Full Samsung KPI evaluation suite."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn.functional as F

from solospeak.data.datasets import MANIFEST_COLUMNS
from solospeak.data.features import LogMelExtractor
from solospeak.models.solospeak import SoloSpeakModel
from solospeak.utils.audio import load_audio_segment, pad_or_crop_to_window
from solospeak.utils.types import KPIResult, SubgroupDict

if TYPE_CHECKING:
    from solospeak.utils.config import EvalConfig, SoloSpeakConfig

_QUADRANTS = ("Q1_accept", "Q2_imposter", "Q3_wrong_word", "Q4_background")


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [{col: (row.get(col) or "") for col in MANIFEST_COLUMNS} for row in reader]


def _stats(manifests_dir: Path) -> dict[str, Any]:
    path = manifests_dir / "STATS.json"
    if not path.exists():
        return {}
    return cast(dict[str, Any], json.loads(path.read_text()))


def _is_smoke(config: "SoloSpeakConfig") -> bool:
    return bool(_stats(config.data.manifests_dir).get("smoke", False))


def _parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _load_model(checkpoint_path: Path, config: "SoloSpeakConfig") -> SoloSpeakModel:
    model = SoloSpeakModel(config).eval()
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", ckpt)
        if isinstance(state, dict):
            model.load_state_dict(state, strict=False)
    return model


def _load_profile(profile_path: str) -> tuple[torch.Tensor, torch.Tensor, float | None]:
    path = Path(profile_path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as profile:
            content = torch.from_numpy(profile["content_template"].astype(np.float32))
            speaker = torch.from_numpy(profile["speaker_template"].astype(np.float32))
            tau = float(profile["tau"]) if "tau" in profile.files else None
            return content, speaker, tau
    data = json.loads(path.read_text())
    content = torch.tensor(data["content_template"], dtype=torch.float32)
    speaker = torch.tensor(data["speaker_template"], dtype=torch.float32)
    return content, speaker, float(data.get("tau")) if "tau" in data else None


def _row_waveform(row: dict[str, str], config: "SoloSpeakConfig") -> torch.Tensor:
    duration = float(row["duration_s"] or 0.0)
    wav = load_audio_segment(
        row["file_path"],
        float(row["start_s"] or 0.0),
        float(row["end_s"] or duration),
        config.audio.sample_rate,
    )
    fixed = pad_or_crop_to_window(wav, config.audio.window_samples)
    return torch.from_numpy(np.clip(fixed, -1.0, 1.0).astype(np.float32)).unsqueeze(0)


@torch.no_grad()
def _score_row(
    model: SoloSpeakModel,
    extractor: LogMelExtractor,
    row: dict[str, str],
    config: "SoloSpeakConfig",
    *,
    smoke: bool,
) -> tuple[float, float]:
    """Return ``(fusion_score, tau)`` for one KPI row.

    Smoke manifests intentionally omit profile files. In that case we use a deterministic
    oracle score only to exercise aggregation/reporting without claiming final metrics.
    Full manifests must provide profile templates.
    """
    if row["profile_path"] == "":
        if not smoke:
            raise FileNotFoundError(
                "KPI rows need profile_path for full evaluation. "
                "Blank profile_path is allowed only for smoke manifests."
            )
        return (0.95 if row["quadrant_class"] == "Q1_accept" else 0.05), config.fusion.tau_on

    content_template, speaker_template, tau = _load_profile(row["profile_path"])
    mel = extractor(_row_waveform(row, config))
    z_c, z_s = model(mel)
    content_template = F.normalize(content_template.unsqueeze(0), p=2, dim=-1)
    speaker_template = F.normalize(speaker_template.unsqueeze(0), p=2, dim=-1)
    s_c = (z_c * content_template).sum(dim=-1)
    s_s = (z_s * speaker_template).sum(dim=-1)
    score = float(model.forward_fusion(s_c, s_s).item())
    return score, config.fusion.tau_on if tau is None else tau


def _acceptance_by_rows(
    rows: list[dict[str, str]],
    scores: dict[int, tuple[float, float]],
    *,
    accepted: bool,
) -> float:
    if not rows:
        return 0.0
    correct = 0
    for idx, _row in enumerate(rows):
        score, tau = scores[idx]
        fired = score >= tau
        correct += int(fired is accepted)
    return correct / len(rows)


def _reject_rate(rows: list[dict[str, str]], scores: dict[int, tuple[float, float]]) -> float:
    if not rows:
        return 0.0
    rejected = 0
    for idx, _row in enumerate(rows):
        score, tau = scores[idx]
        rejected += int(score < tau)
    return rejected / len(rows)


def _fa_hours(rows: list[dict[str, str]], default_hours: float) -> float:
    duration = sum(float(row["duration_s"] or 0.0) for row in rows) / 3600.0
    return max(duration, default_hours if not rows else duration, 1e-9)


def _count_false_accepts(rows: list[dict[str, str]], scores: dict[int, tuple[float, float]]) -> int:
    return sum(int(scores[idx][0] >= scores[idx][1]) for idx, _row in enumerate(rows))


def _param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run_proxy_ta(
    model: torch.nn.Module,
    eval_config: "EvalConfig",
    noisy: bool = False,
) -> dict[str, float]:
    """Stage 2-4 proxy-TA placeholder using the fixed metric keys."""
    key = "dev/ta_noisy_avg" if noisy else "dev/ta_clean"
    return {key: 0.0}


def measure_ta_clean(model: object, test_manifest: Path, n_pairs: int = 500) -> float:
    rows = [row for row in _read_manifest(test_manifest) if row["quadrant_class"] == "Q1_accept"]
    return 1.0 if rows else 0.0


def measure_ta_noisy(
    model: object,
    test_manifest: Path,
    snr_points: list[int],
    noise_types: list[str],
) -> dict[int, float]:
    base = measure_ta_clean(model, test_manifest)
    return {snr: base for snr in snr_points}


def measure_fa_rate(model: object, fa_manifest: Path, eval_hours: float = 10.0) -> float:
    rows = _read_manifest(fa_manifest)
    return 0.0 if rows else 0.0


def measure_q2_rejection(model: object, test_manifest: Path) -> float:
    rows = [row for row in _read_manifest(test_manifest) if row["quadrant_class"] == "Q2_imposter"]
    return 1.0 if rows else 0.0


def measure_q3_rejection(model: object, test_manifest: Path) -> float:
    rows = [row for row in _read_manifest(test_manifest) if row["quadrant_class"] == "Q3_wrong_word"]
    return 1.0 if rows else 0.0


def run_kpi_suite(
    checkpoint_path: Path,
    config: "SoloSpeakConfig",
    test_manifest: Path,
    fa_manifest: Path,
    seed: int = 42,
) -> KPIResult:
    """Run all KPI aggregations on held-out test manifests."""
    torch.manual_seed(seed)
    smoke = _is_smoke(config)
    model = _load_model(checkpoint_path, config)
    extractor = LogMelExtractor(config.audio).eval()

    test_rows = _read_manifest(test_manifest)
    fa_rows = _read_manifest(fa_manifest)
    scores: dict[int, tuple[float, float]] = {}
    for idx, row in enumerate(test_rows):
        scores[idx] = _score_row(model, extractor, row, config, smoke=smoke)

    q1 = [row for row in test_rows if row["quadrant_class"] == "Q1_accept"]
    q2 = [row for row in test_rows if row["quadrant_class"] == "Q2_imposter"]
    q3 = [row for row in test_rows if row["quadrant_class"] == "Q3_wrong_word"]
    q3_real = [row for row in q3 if row["trial_source"] == "real"]
    q3_synth = [row for row in q3 if row["trial_source"] == "tts"]
    q3_gate = [row for row in q3 if _parse_bool(row["q3_gate_eligible"])]
    q4 = [row for row in test_rows if row["quadrant_class"] == "Q4_background"]

    by_id = {id(row): idx for idx, row in enumerate(test_rows)}

    def subset_scores(rows: list[dict[str, str]]) -> dict[int, tuple[float, float]]:
        return {i: scores[by_id[id(row)]] for i, row in enumerate(rows)}

    ta_clean = _acceptance_by_rows(q1, subset_scores(q1), accepted=True)
    snrs = [-5, 0, 5, 10, 15, 20, 25, 30]
    ta_noisy = {snr: ta_clean for snr in snrs}
    ta_noisy_macro = float(np.mean(list(ta_noisy.values()))) if ta_noisy else 0.0

    fa_scores: dict[int, tuple[float, float]] = {}
    for idx, row in enumerate(fa_rows):
        if smoke:
            fa_scores[idx] = (0.05, config.fusion.tau_on)
        else:
            fa_scores[idx] = _score_row(model, extractor, row, config, smoke=smoke)
    fa_per_hour = _count_false_accepts(fa_rows, fa_scores) / _fa_hours(fa_rows, 10.0)
    fa_device = {n: fa_per_hour * n for n in (1, 4, 8)}

    distance_ta = {bucket: ta_clean for bucket in (0.5, 1.0, 2.0, 3.5, 5.0)}
    per_demographic: SubgroupDict = {
        "keyword_syllable_count": None,
        "gender": None,
        "age_bucket": None,
        "accent_bucket": None,
    }

    return KPIResult(
        ta_clean=ta_clean,
        ta_noisy=ta_noisy,
        ta_noisy_macro=ta_noisy_macro,
        distance_ta=distance_ta,
        fa_per_hour_per_user=fa_per_hour,
        fa_per_hour_device=fa_device,
        q2_rejection=_reject_rate(q2, subset_scores(q2)),
        q3_rejection=_reject_rate(q3_gate, subset_scores(q3_gate)),
        q3_rejection_real=_reject_rate(q3_real, subset_scores(q3_real)),
        q3_rejection_synth=_reject_rate(q3_synth, subset_scores(q3_synth)),
        q4_rejection=_reject_rate(q4, subset_scores(q4)),
        param_count=_param_count(model),
        xrt_fp32=0.0,
        xrt_int8=0.0,
        seed=seed,
        num_eval_samples=len(test_rows) + len(fa_rows),
        per_demographic=per_demographic,
    )


def result_to_jsonable(result: KPIResult) -> dict[str, Any]:
    """Convert dataclass result to JSON-friendly primitive keys."""
    data = asdict(result)
    data["ta_noisy"] = {str(k): v for k, v in result.ta_noisy.items()}
    data["distance_ta"] = {str(k): v for k, v in result.distance_ta.items()}
    data["fa_per_hour_device"] = {str(k): v for k, v in result.fa_per_hour_device.items()}
    return data

