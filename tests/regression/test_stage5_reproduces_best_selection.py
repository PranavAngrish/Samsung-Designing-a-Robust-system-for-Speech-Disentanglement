from __future__ import annotations

from solospeak.training.stages.stage5_notebook_search import (
    EXPECTED_STAGE5_BEST_SIGNATURE,
    EXPECTED_STAGE5_WINNER_CONFIG,
    stage5_result_signature,
)
from solospeak.utils.config import SoloSpeakConfig


def test_stage5_best_signature_matches_final_notebook_path() -> None:
    result = {
        "stage4_candidate": "stage4d_hardq2_mining_balanced",
        "data_variant": {"name": "zero_e3_product"},
        "fusion_variant": {"name": "q2_very_strong"},
    }

    assert stage5_result_signature(result) == EXPECTED_STAGE5_BEST_SIGNATURE


def test_stage5_winner_signature() -> None:
    config = SoloSpeakConfig.from_yaml("configs/training/production.yaml")

    assert config.stage5.stage4_candidate_name == "stage4d_hardq2_mining_balanced"
    assert config.stage5.data_variant_name == "zero_e3_product"
    assert config.stage5.fusion_variant_name == "q2_very_strong"
    assert config.stage5.n_enroll == 3
    assert config.stage5.wake_word == "zero"
    assert EXPECTED_STAGE5_WINNER_CONFIG["data_variant"]["n_enroll"] == 3
    assert EXPECTED_STAGE5_WINNER_CONFIG["fusion_variant"]["q2_weight"] == 4.0
