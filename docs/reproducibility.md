# Reproducibility

## Environment

Use Python 3.10 or 3.11, then install the development dependencies:

```bash
pip install -e ".[dev]"
```

Generated checkpoints, profiles, reports, processed data, and exports are intentionally
gitignored. Put large model files in a GitHub Release or Kaggle Dataset, not normal git.

## Final Production Path

The source-controlled production path is:

```text
Stage 1 -> Stage 2 -> Stage 3 -> Stage 4C -> Stage 4D hard-Q2 mining
-> Stage 5 dual fusion search -> Stage 6 final evaluation/export
-> Stage 7 external FA tuning -> Stage 7 joint tau correction/export
```

Run it from the repository root:

```bash
python -m scripts.run_production_pipeline \
  --config configs/training/production.yaml \
  --common-voice-root /kaggle/input/datasets/organizations/mozillaorg/common-voice \
  --librispeech-root /kaggle/input/datasets/a24998667/librispeech \
  --background-noise-root /kaggle/input/datasets/axondata/background-noise-detection-dataset \
  --urbansound-root /kaggle/input/datasets/chrisfilo/urbansound8k
```

Expected final outputs:

```text
checkpoints/stage1_backbone.pt
checkpoints/stage2_dualhead.pt
checkpoints/stage3_disentangle.pt
checkpoints/stage4_robust.pt
checkpoints/stage4d_hardq2_mining_balanced.pt
checkpoints/stage5_fusion.pt
checkpoints/stage6_final.pt
checkpoints/stage7_fusion.pt
checkpoints/stage7_final_corrected.pt
exports/solospeak_stage7_deployable_corrected.pt
reports/production_pipeline_summary.json
reports/stage7_joint_threshold_calibration_summary.json
reports/stage7_final_kpi_verification.json
```

The corrected final deployable must report `tau_on` close to `0.27`. Do not treat
`exports/solospeak_stage7_deployable.pt` or any `tau=0.935` export as final.

## Stage Notes

Stage 4D is implemented in
`solospeak/training/stages/stage4d_hardq2_mining.py`. It loads
`checkpoints/stage4_robust.pt`, mines hard Q2 same-word imposters, and writes
`checkpoints/stage4d_hardq2_mining_balanced.pt`.

Stage 5 searches both Stage4C and Stage4D candidates. The final notebook-winning
signature is:

```text
stage4_candidate = stage4d_hardq2_mining_balanced
data_variant = zero_e3_product
fusion_variant = q2_very_strong
```

Stage 6 is the final Stage-5 handoff evaluation/export, not QAT and not the final
production-candidate threshold. It writes `checkpoints/stage6_final.pt` and
`exports/solospeak_stage6_deployable.pt`; Stage 7 is required for external FA tuning
and corrected joint threshold export.

Stage 7 builds GSC enrollment profiles, scores about 40,000 external false-accept trials,
fine-tunes only the fusion MLP, then performs joint internal + external threshold
calibration.

## Stage 7 Only

From an existing Stage 6 checkpoint:

```bash
python -m scripts.run_stage --stage 7 \
  --resume-from checkpoints/stage6_final.pt \
  --external-common-voice-root /kaggle/input/datasets/organizations/mozillaorg/common-voice \
  --external-librispeech-root /kaggle/input/datasets/a24998667/librispeech \
  --external-background-noise-root /kaggle/input/datasets/axondata/background-noise-detection-dataset \
  --external-urbansound-root /kaggle/input/datasets/chrisfilo/urbansound8k

python -m scripts.run_stage7_verify --checkpoint checkpoints/stage7_fusion.pt
python -m scripts.run_stage7_finalization
```

## Smoke Pipeline

The smoke path validates code wiring and artifact formats, not final accuracy:

```bash
make install-dev
make download-data-smoke
make prepare-manifests-smoke
python -m scripts.run_production_pipeline --config configs/training/production.yaml --smoke
make test
```

## Production Summary Schema

`reports/production_pipeline_summary.json` contains:

```text
final_artifact
final_tau
stage5_best_candidate
stage6_demo_passed
stage7_default_tau
stage7_corrected_tau
final_internal_metrics
final_external_fa_metrics
production_candidate
dataset_roots_used
git_commit
timestamp
```

## Determinism

Training and evaluation code uses deterministic seeds where practical. Exact bit-for-bit
reproducibility can still vary across CUDA, PyTorch, and audio decoding libraries, so
final reports should include config path, checkpoint path, seed, and git commit.
