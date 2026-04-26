"""Smoke tests for evaluation infrastructure (no real checkpoints needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from solospeak.eval.ablations import ABLATION_CONFIGS


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


@pytest.mark.slow
def test_kpi_suite_runs(tmp_path: Path) -> None:
    pytest.skip("Requires trained checkpoint — implement in Phase 4")


@pytest.mark.slow
def test_probe_eval_runs(tmp_path: Path) -> None:
    pytest.skip("Requires trained checkpoint — implement in Phase 4")
