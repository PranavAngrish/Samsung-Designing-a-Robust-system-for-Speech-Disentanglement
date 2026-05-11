from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pytest

from solospeak.eval.threshold_calibration import calibrate_joint_threshold
from solospeak.training.stages.common import save_checkpoint
from solospeak.utils.config import SoloSpeakConfig


FIXTURE_DIR = Path("tests/fixtures/calibration")


def test_checkpoint_extra_schema(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    config = SoloSpeakConfig(run_name="checkpoint-schema-test", training={"stage": 1})
    path = tmp_path / "schema.pt"

    save_checkpoint(
        path,
        stage_origin=1,
        config=config,
        step=7,
        metrics={"ok": 1.0},
        model_state={},
        extra={"abc": 123},
    )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["metrics"]["ok"] == 1.0
    assert ckpt["model_state"] == {}
    assert ckpt["stage_origin"] == 1
    assert ckpt["step"] == 7
    assert ckpt["extra"]["abc"] == 123
    assert ckpt["abc"] == 123


def test_corrected_tau_not_bad_tau(tmp_path: Path) -> None:
    summary = calibrate_joint_threshold(
        FIXTURE_DIR / "internal_scores.csv",
        FIXTURE_DIR / "external_fa_scores.csv",
        calibration_csv=tmp_path / "stage7_joint_calibration_fixture.csv",
        summary_json=tmp_path / "stage7_joint_calibration_fixture.json",
    )

    assert abs(summary["export_tau"] - 0.27) < 1e-9
    assert summary["export_tau"] != 0.935
    assert summary["best"]["ta_clean"] < 1.0


def test_joint_calibration_preserves_internal_kpi(tmp_path: Path) -> None:
    summary = calibrate_joint_threshold(
        FIXTURE_DIR / "internal_scores.csv",
        FIXTURE_DIR / "external_fa_scores.csv",
        calibration_csv=tmp_path / "stage7_joint_calibration_kpi.csv",
        summary_json=tmp_path / "stage7_joint_calibration_kpi.json",
    )
    best = summary["best"]

    assert best["ta_clean"] >= 0.92
    assert best["q2_rejection"] >= 0.95
    assert best["q3_rejection"] >= 0.90
    assert best["q4_rejection"] >= 0.99
    assert best["overall_fa_rate"] <= 0.005
    assert summary["export_mode"] == "production_candidate"


def test_audio_loader_mp3_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from solospeak.utils import audio as audio_mod

    def fail_soundfile(*_args: object, **_kwargs: object) -> tuple[np.ndarray, int]:
        raise RuntimeError("mocked mp3 decode failure")

    fallback_calls: list[tuple[str, int, bool]] = []

    def fake_librosa_load(path: str, sr: int, mono: bool) -> tuple[np.ndarray, int]:
        fallback_calls.append((path, sr, mono))
        return np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32), sr

    monkeypatch.setattr(audio_mod.sf, "read", fail_soundfile)
    monkeypatch.setitem(sys.modules, "librosa", types.SimpleNamespace(load=fake_librosa_load))

    wav = audio_mod.load_audio(tmp_path / "common_voice_clip.mp3", target_sr=16000)

    assert fallback_calls == [(str(tmp_path / "common_voice_clip.mp3"), 16000, True)]
    assert wav.dtype == np.float32
    assert np.max(wav) <= 1.0
    assert np.min(wav) >= -1.0
