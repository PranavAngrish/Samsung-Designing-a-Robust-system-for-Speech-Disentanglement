"""Smoke tests for evaluation infrastructure (no real checkpoints needed)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import torch

from solospeak.eval.ablations import ABLATION_CONFIGS, format_ablation_table, run_all_ablations
from solospeak.eval.kpi_suite import result_to_jsonable, run_kpi_suite
from solospeak.models.solospeak import SoloSpeakModel
from solospeak.utils.config import SoloSpeakConfig


def test_ablation_configs_count() -> None:
    assert len(ABLATION_CONFIGS) == 7


def test_ablation_configs_have_required_names() -> None:
    required = {
        "full", "no_ortho", "no_adv", "no_disent",
        "no_tts_enroll", "no_gated_fusion", "no_curriculum",
    }
    assert set(ABLATION_CONFIGS.keys()) == required


def test_full_config_is_empty_override() -> None:
    assert ABLATION_CONFIGS["full"] == {}


def _smoke_ready() -> bool:
    return Path("data/manifests/STATS.json").exists() and Path("data/raw/smoke").exists()


def _write_smoke_checkpoint(tmp_path: Path) -> Path:
    cfg = SoloSpeakConfig.from_yaml("configs/defaults.yaml")
    model = SoloSpeakModel(cfg)
    path = tmp_path / "smoke_model.pt"
    torch.save({"model_state": model.state_dict(), "stage_origin": 6}, path)
    return path


def test_kpi_suite_smoke_fields(tmp_path: Path) -> None:
    if not _smoke_ready():
        pytest.skip("Run make download-data-smoke && make prepare-manifests-smoke first")
    cfg = SoloSpeakConfig.from_yaml("configs/defaults.yaml")
    result = run_kpi_suite(
        _write_smoke_checkpoint(tmp_path),
        cfg,
        Path("data/manifests/test_kpi.csv"),
        Path("data/manifests/test_fa.csv"),
        seed=42,
    )
    data = result_to_jsonable(result)
    required = {
        "ta_clean",
        "ta_noisy",
        "ta_noisy_macro",
        "fa_per_hour_per_user",
        "fa_per_hour_device",
        "q2_rejection",
        "q3_rejection",
        "q3_rejection_real",
        "q3_rejection_synth",
        "q4_rejection",
        "distance_ta",
        "param_count",
        "xrt_fp32",
        "xrt_int8",
        "per_demographic",
    }
    assert required.issubset(data)


def test_run_eval_writes_phase4_reports(tmp_path: Path) -> None:
    if not _smoke_ready():
        pytest.skip("Run make download-data-smoke && make prepare-manifests-smoke first")
    ckpt = _write_smoke_checkpoint(tmp_path)
    kpi = tmp_path / "kpi_final.json"
    probes = tmp_path / "probes.json"
    subgroup = tmp_path / "subgroup_report.md"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_eval",
            "--checkpoint",
            str(ckpt),
            "--seeds",
            "42",
            "--output",
            str(kpi),
            "--probes-output",
            str(probes),
            "--subgroup-output",
            str(subgroup),
        ],
        check=True,
    )
    assert "ta_clean" in json.loads(kpi.read_text())
    assert "reduction_c" in json.loads(probes.read_text())
    assert "keyword_syllable_count" in subgroup.read_text()


def test_ablation_table_has_seven_rows(tmp_path: Path) -> None:
    results = run_all_ablations(tmp_path / "stage2.pt", tmp_path, seeds=[42])
    table = format_ablation_table(results)
    rows = [line for line in table.splitlines() if line.startswith("| ") and not line.startswith("| Ablation")]
    assert len(rows) == 7
    assert "no_gated_fusion" in table
