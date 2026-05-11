from __future__ import annotations

from pathlib import Path

import pytest

from solospeak.deployment.export_stage7 import export_corrected_stage7


def test_corrected_deployable_writes_joint_tau(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    old_deployable = tmp_path / "solospeak_stage7_deployable.pt"
    corrected_checkpoint = tmp_path / "stage7_final_corrected.pt"
    corrected_deployable = tmp_path / "solospeak_stage7_deployable_corrected.pt"
    torch.save(
        {
            "model_state": {},
            "tau_on": 0.935,
            "config": {"fusion": {"tau_on": 0.935}},
        },
        old_deployable,
    )

    export_corrected_stage7(
        old_deployable=old_deployable,
        corrected_checkpoint=corrected_checkpoint,
        corrected_deployable=corrected_deployable,
        calibration_summary={
            "export_tau": 0.27,
            "export_mode": "production_candidate",
            "best": {
                "ta_clean": 0.939655,
                "q2_rejection": 0.951691,
                "q3_rejection": 0.978261,
                "q4_rejection": 1.0,
                "quadrant_min": 0.939655,
                "overall_fa_rate": 0.003,
                "overall_false_accepts": 120,
            },
        },
        config_path=Path("configs/training/stage7_external_fa.yaml"),
    )

    corrected = torch.load(corrected_deployable, map_location="cpu", weights_only=False)
    assert corrected["tau_on"] < 0.5
    assert abs(corrected["tau_on"] - 0.27) <= 0.03
    assert corrected["tau_on"] != 0.935
    assert corrected["config"]["fusion"]["tau_on"] == pytest.approx(0.27)
    assert corrected["stage7_extra"]["export_tau"] == pytest.approx(0.27)
    assert corrected["stage7_extra"]["export_mode"] == "production_candidate"
