from __future__ import annotations

import csv
from pathlib import Path

from solospeak.eval.stage7_joint_calibration import calibrate_joint_threshold


def _write_joint_tau_fixture(tmp_path: Path) -> tuple[Path, Path]:
    internal_path = tmp_path / "internal.csv"
    external_path = tmp_path / "external.csv"

    rows: list[dict[str, object]] = []
    rows += [{"quadrant": "Q1_accept", "prob": 0.276} for _ in range(94)]
    rows += [{"quadrant": "Q1_accept", "prob": 0.20} for _ in range(6)]
    rows += [{"quadrant": "Q2_imposter", "prob": 0.30} for _ in range(4)]
    rows += [{"quadrant": "Q2_imposter", "prob": 0.267} for _ in range(5)]
    rows += [{"quadrant": "Q2_imposter", "prob": 0.10} for _ in range(91)]
    rows += [{"quadrant": "Q3_wrong_word", "prob": 0.30} for _ in range(2)]
    rows += [{"quadrant": "Q3_wrong_word", "prob": 0.10} for _ in range(98)]
    rows += [{"quadrant": "Q4_background", "prob": 0.10} for _ in range(100)]

    with open(internal_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["quadrant", "prob"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    external_rows: list[dict[str, object]] = []
    external_rows += [{"source_dataset": "common_voice", "prob": 0.30} for _ in range(2)]
    external_rows += [{"source_dataset": "common_voice", "prob": 0.10} for _ in range(698)]
    external_rows += [{"source_dataset": "librispeech", "prob": 0.30}]
    external_rows += [{"source_dataset": "librispeech", "prob": 0.10} for _ in range(199)]
    external_rows += [{"source_dataset": "urbansound8k", "prob": 0.10} for _ in range(90)]
    external_rows += [{"source_dataset": "background_noise", "prob": 0.10} for _ in range(10)]

    with open(external_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_dataset", "prob"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(external_rows)

    return internal_path, external_path


def test_stage7_joint_calibration_selects_corrected_tau(tmp_path: Path) -> None:
    internal_path, external_path = _write_joint_tau_fixture(tmp_path)

    summary = calibrate_joint_threshold(
        internal_path,
        external_path,
        calibration_csv=tmp_path / "calibration.csv",
        summary_json=tmp_path / "summary.json",
    )

    assert abs(summary["export_tau"] - 0.27) < 1e-9
    assert summary["export_tau"] != 0.935
    assert summary["best"]["ta_clean"] >= 0.92
    assert summary["best"]["q2_rejection"] >= 0.95
