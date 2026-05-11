"""Rebuild Stage-7 verification scores from source code.

This replaces the final verification notebook cells:
  - rebuild GSC Q1/Q2/Q3/Q4 internal scores
  - score Stage7 external false-accept rows
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage7 internal/external verification")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/stage7_fusion.pt"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/stage7_external_fa.yaml"),
    )
    parser.add_argument(
        "--external-manifest",
        type=Path,
        default=Path("reports/stage7_external_fa_manifest.csv"),
    )
    parser.add_argument(
        "--internal-scores",
        type=Path,
        default=Path("reports/stage7_final_internal_scores.csv"),
    )
    parser.add_argument(
        "--external-scores",
        type=Path,
        default=Path("reports/stage7_external_fa_scores.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/stage7_final_kpi_verification.json"),
    )
    args = parser.parse_args()

    import torch

    from solospeak.data.features import LogMelExtractor
    from solospeak.eval.external_fa import read_external_fa_manifest, score_external_fa_rows
    from solospeak.eval.internal_quadrants import (
        InternalQuadrantSpec,
        build_gsc_internal_examples,
        score_internal_examples_to_csv,
    )
    from solospeak.models.solospeak import SoloSpeakModel
    from solospeak.training.stages.common import default_device
    from solospeak.utils.config import SoloSpeakConfig
    from solospeak.utils.seeding import seed_everything

    config = SoloSpeakConfig.from_yaml(args.config)
    seed_everything(config.training.seed)
    device = default_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metrics = checkpoint.get("metrics", {}) or {}
    extra = checkpoint.get("extra", {}) or {}
    tau = float(extra.get("stage7_tau", metrics.get("dev/tau_stage7", config.fusion.tau_on)))
    config.fusion.tau_on = tau

    model = SoloSpeakModel(config).to(device).eval()
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Bad Stage7 state: missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    extractor = LogMelExtractor(config.audio).to(device).eval()

    profiles, internal_examples = build_gsc_internal_examples(
        model=model,
        extractor=extractor,
        config=config,
        manifests_dir=config.data.manifests_dir,
        profile_dir=Path("data/stage7_verify_profiles"),
        spec=InternalQuadrantSpec(
            wake_word=config.stage7.wake_word,
            n_enroll=config.stage7.n_enroll,
            max_profiles=config.stage7.max_profiles,
            profile_seed=777,
            q1_per_profile=2,
            q2_per_profile=6,
            q3_per_profile=2,
            q4_per_profile=2,
            require_wrong_word_for_profile=True,
            profile_prefix="stage7_verify_profile",
        ),
        device=device,
    )
    internal_metrics = score_internal_examples_to_csv(
        model=model,
        examples=internal_examples,
        tau=tau,
        output_path=args.internal_scores,
        device=device,
    )

    external_summary = None
    if args.external_manifest.exists():
        external_rows = read_external_fa_manifest(args.external_manifest)
        external_summary = score_external_fa_rows(
            model=model,
            extractor=extractor,
            config=config,
            rows=external_rows,
            tau=tau,
            output_path=args.external_scores,
            device=device,
        )

    pass_checks = {
        "ta_clean_passed": internal_metrics["ta_clean"] >= config.stage7.ta_clean_min,
        "q2_rejection_passed": internal_metrics["q2_rejection"]
        >= config.stage7.q2_rejection_min,
        "q3_rejection_passed": internal_metrics["q3_rejection"] >= 0.90,
        "q4_rejection_passed": internal_metrics["q4_rejection"] >= 0.99,
        "quadrant_min_passed": internal_metrics["quadrant_accuracy_min"] >= 0.90,
    }
    if external_summary is not None:
        by_source = external_summary["by_source"]
        overall = external_summary["overall"]
        pass_checks.update(
            {
                "overall_external_fa_passed": overall["fa_rate"]
                <= config.stage7.external_fa_rate_max,
                "common_voice_fa_passed": by_source.get("common_voice", {"fa_rate": 0.0})[
                    "fa_rate"
                ]
                <= config.stage7.common_voice_fa_rate_max,
                "librispeech_fa_passed": by_source.get("librispeech", {"fa_rate": 0.0})[
                    "fa_rate"
                ]
                <= 0.005,
                "urbansound_fa_passed": by_source.get("urbansound8k", {"fa_rate": 0.0})[
                    "fa_rate"
                ]
                <= 0.005,
                "background_noise_fa_passed": by_source.get(
                    "background_noise", {"fa_rate": 0.0}
                )["fa_rate"]
                <= 0.005,
            }
        )

    payload = {
        "stage7_checkpoint": str(args.checkpoint),
        "export_tau": float(tau),
        "wake_word": config.stage7.wake_word,
        "n_enroll": config.stage7.n_enroll,
        "profiles": len(profiles),
        "internal_metrics": internal_metrics,
        "external_summary": external_summary,
        "pass_checks": pass_checks,
        "final_production_candidate_passed": bool(all(pass_checks.values())),
        "files": {
            "internal_scores": str(args.internal_scores),
            "external_scores": str(args.external_scores) if external_summary is not None else None,
            "verification_summary": str(args.summary),
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Internal scores: {args.internal_scores}")
    if external_summary is not None:
        print(f"External scores: {args.external_scores}")
    print(f"Verification summary: {args.summary}")


if __name__ == "__main__":
    main()
