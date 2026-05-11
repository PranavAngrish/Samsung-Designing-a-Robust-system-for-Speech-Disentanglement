from __future__ import annotations

from pathlib import Path

from solospeak.eval.stage7_joint_calibration import calibrate_joint_threshold
from tests.regression.test_stage7_joint_tau import _write_joint_tau_fixture


def test_joint_calibration_rejects_external_only_high_tau(tmp_path: Path) -> None:
    internal_path, external_path = _write_joint_tau_fixture(tmp_path)

    summary = calibrate_joint_threshold(
        internal_path,
        external_path,
        calibration_csv=tmp_path / "calibration.csv",
        summary_json=tmp_path / "summary.json",
    )

    assert summary["export_tau"] < 0.5
    assert summary["export_tau"] != 0.935
    assert summary["best"]["ta_clean"] >= 0.92
