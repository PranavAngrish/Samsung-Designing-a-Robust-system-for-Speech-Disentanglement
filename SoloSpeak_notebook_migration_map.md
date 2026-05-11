# SoloSpeak Kaggle Notebook → Repository Migration Map

This map identifies the notebook cells that produced the final successful SoloSpeak
pipeline and where each has been migrated in the source repository.

## Final successful artifacts from Kaggle

- Final deployable artifact: `exports/solospeak_stage7_deployable_corrected.pt`
- Corrected final tau: `0.27`
- Final joint KPI:
  - TA clean: `0.9397`
  - Q2 rejection: `0.9517`
  - Q3 rejection: `0.9783`
  - Q4 rejection: `1.0000`
  - Quadrant min: `0.9397`
  - External FA rate: `0.003000`
  - External FA count: `120 / 40000`

## Notebook Cell Mapping

| Pipeline part | Notebook | Cell(s) | Extracted line range | Repo target |
|---|---:|---:|---:|---|
| Dataset linking + manifest creation | `create-manifest.ipynb` | 2–7 | `create-manifest.py:L22-L812` | `scripts/prepare_manifests.py`, `scripts/download_datasets.py`, `data/README.md` |
| Stage 1 backbone pretraining | `samsung-stage1.ipynb` | 1–3 | `samsung-stage1.py:L22-L627` | `solospeak/training/stages/stage1_backbone.py` |
| Stage 2 dual-head training | `samsung-stage2.ipynb` | 1–4 | `samsung-stage2.py:L22-L843` | `solospeak/training/stages/stage2_dual_head.py` |
| Stage 3 initial/old continuation | `samsung-stage2.ipynb` | 6–10 | `samsung-stage2.py:L844-L1510` | historical only; superseded by clean Stage 3 |
| Final clean Stage 3 search | `samsung-stage3.ipynb` | 0–1 | `samsung-stage3.py:L3-L860` | `solospeak/training/stages/stage3_disentangle.py`, `solospeak/training/stages/stage3_notebook_search.py` |
| Stage 4D Hard-Q2 mining | `stage4-final-improvement.ipynb` | 1–3 | `stage4-final-improvement.py:L22-L2117` | `solospeak/training/stages/stage4d_hardq2_mining.py` |
| Stage 5 dual Stage4 candidate fusion search | `clean-the-mess.ipynb` | 1–3 | `clean-the-mess.py:L22-L2211` | `solospeak/training/stages/stage5_fusion.py`, `solospeak/training/stages/stage5_notebook_search.py` |
| Stage 6 final production eval/export | `clean-the-mess.ipynb` | 5 | `clean-the-mess.py:L2212-L3228` | `solospeak/training/stages/stage6_final_eval.py` |
| External FA test before Stage 7 | `ta-test.ipynb` | 1–8 | `ta-test.py:L22-L1317` | `solospeak/eval/external_fa.py` |
| Stage 7 setup | `ta-test.ipynb` | 10 | `ta-test.py:L1318-L1454` | `solospeak/training/stages/stage7_fa_tuning.py` setup helpers |
| Stage 7 GSC profile/internal examples | `ta-test.ipynb` | 11 | `ta-test.py:L1455-L1736` | `solospeak/eval/internal_quadrants.py`, `solospeak/training/stages/stage7_fa_tuning.py` |
| Stage 7 external FA discovery | `ta-test.ipynb` | 12 | `ta-test.py:L1737-L1840` | `solospeak/eval/external_fa.py` |
| Stage 7 external FA manifest | `ta-test.ipynb` | 13 | `ta-test.py:L1841-L2057` | `solospeak/eval/external_fa.py` |
| Stage 7 hard-negative dataset assembly | `ta-test.ipynb` | 14 | `ta-test.py:L2058-L2150` | `solospeak/training/stages/stage7_fa_tuning.py` |
| Stage 7 fusion training | `ta-test.ipynb` | 15 | `ta-test.py:L2151-L2493` | `solospeak/training/stages/stage7_fa_tuning.py` |
| Stage 7 external FA evaluation | `ta-test.ipynb` | 16/17 | `ta-test.py:L2494-L2945` | `solospeak/eval/external_fa.py`, `scripts/run_stage7_verify.py` |
| Incorrect external-only calibration/export | `ta-test.ipynb` | 18–19 | `ta-test.py:L2946-L3260` | do not use as final export logic; keep as historical caution |
| Final KPI verification rebuilding internal set | `ta-test.ipynb` | 20 | `ta-test.py:L3261-L3971` | `scripts/run_stage7_verify.py`, `solospeak/eval/internal_quadrants.py` |
| Correct joint tau calibration + corrected export | `ta-test.ipynb` | 21 | `ta-test.py:L3972-L4325` | `solospeak/eval/threshold_calibration.py`, `solospeak/deployment/export_stage7.py`, `scripts/run_stage7_finalization.py` |

## Cells To Treat As Final Authority

The final authority for the deployable product is:

1. `clean-the-mess.ipynb`, cell 3 — Stage 5 dual Stage4 candidate search.
2. `clean-the-mess.ipynb`, cell 5 — Stage 6 final evaluation/export.
3. `stage4-final-improvement.ipynb`, cell 3 — Stage 4D online hard-Q2 mining.
4. `ta-test.ipynb`, cells 10–17 — Stage 7 hard-negative fusion tuning and FA evaluation.
5. `ta-test.ipynb`, cell 20 — final KPI verification at exported tau, which revealed tau `0.935` was wrong.
6. `ta-test.ipynb`, cell 21 — corrected joint internal + external calibration and final corrected export using tau `0.27`.

## Important Migration Rule

Do not export based on external FA alone. The previous `tau=0.935` export had excellent
FA but killed true accepts. The repo implements **joint calibration** using both:

- internal Q1/Q2/Q3/Q4 scores, and
- external FA scores.

The final corrected threshold is `tau=0.27`.

## Migration Status In This Repo

The Stage 7 notebook path is now implemented as source code:

- Stage 1 now uses the notebook recoverability behavior by writing `checkpoints/stage1_last.pt` before the gate and only writing `checkpoints/stage1_backbone.pt` after the gate.
- Stage 2 now writes `checkpoints/stage2_last.pt` before the gate can fail, matching the notebook continuation path used by Stage 3.
- `python -m scripts.run_stage --stage 7` runs fusion-only external FA hard-negative tuning and writes `checkpoints/stage7_fusion.pt`.
- `python -m scripts.run_stage7_verify` rebuilds final GSC internal Q1/Q2/Q3/Q4 scores and rescored external FA CSVs.
- `python -m scripts.run_stage7_finalization` exports `exports/solospeak_stage7_deployable.pt`, runs joint internal + external tau calibration, and writes `exports/solospeak_stage7_deployable_corrected.pt`.
- Non-smoke `python -m scripts.run_stage --stage 5` now uses the notebook Stage 5 fusion search instead of the earlier compact smoke skeleton. It writes `reports/stage5_dual_search_results.json`, `checkpoints/stage5_last.pt`, and `checkpoints/stage5_fusion.pt` only if the notebook gate passes.
- Non-smoke `python -m scripts.run_stage --stage 3` now uses the notebook Stage 3 variant search and probe gate, writing `reports/stage3_search_results.json`, `checkpoints/stage3_last.pt`, and `checkpoints/stage3_disentangle.pt` only when the strict probe gate passes.
- `python -m scripts.run_stage --stage 4d` now runs the notebook hard-Q2 mining search from Stage4C and writes `checkpoints/stage4d_hardq2_mining_balanced.pt`.
- Non-smoke `python -m scripts.run_stage --stage 6` now performs final Stage 5 handoff evaluation/export and writes `checkpoints/stage6_final.pt`, `exports/solospeak_stage6_deployable.pt`, and Stage 6 reports.
- `python -m scripts.run_production_pipeline --config configs/training/production.yaml ...` now orchestrates the full Stage 1→7 corrected production path.

Stage 4C remains `solospeak/training/stages/stage4_robustness.py`; Stage 4D is a separate challenger checkpoint so Stage 5 can compare both Stage4C and Stage4D exactly as the winning notebook did.
