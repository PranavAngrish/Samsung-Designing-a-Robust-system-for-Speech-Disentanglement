# Release Manifest

Large checkpoints and deployable model files are not committed to git. Attach them to a
GitHub Release or Kaggle Dataset and keep this manifest updated with SHA-256 values from
the released files.

## Final Production Artifacts

| Artifact | Role | SHA-256 |
|---|---|---|
| `exports/solospeak_stage7_deployable_corrected.pt` | Final recommended deployable, `tau_on ~= 0.27` | `TBD_AFTER_RELEASE` |
| `checkpoints/stage7_final_corrected.pt` | Full corrected Stage 7 checkpoint | `TBD_AFTER_RELEASE` |
| `checkpoints/stage7_fusion.pt` | Stage 7 fusion-head tuned checkpoint before joint tau correction | `TBD_AFTER_RELEASE` |
| `checkpoints/stage6_final.pt` | Stage 6 final evaluation/export checkpoint | `TBD_AFTER_RELEASE` |
| `checkpoints/stage5_fusion.pt` | Stage 5 best fusion checkpoint | `TBD_AFTER_RELEASE` |
| `checkpoints/stage4d_hardq2_mining_balanced.pt` | Stage 4D hard-Q2 mining checkpoint used by Stage 5 | `TBD_AFTER_RELEASE` |

## Validation Files

| Report | Expected Location |
|---|---|
| Production pipeline summary | `reports/production_pipeline_summary.json` |
| Joint threshold calibration | `reports/stage7_joint_threshold_calibration_summary.json` |
| Final KPI verification | `reports/stage7_final_kpi_verification.json` |
| Stage 7 external FA summary | `reports/stage7_external_fa_summary.json` |

## Expected Final Metrics

```text
tau_on: 0.27
TA clean: 93.97%
Q2 rejection: 95.17%
Q3 rejection: 97.83%
Q4 rejection: 100.00%
External FA rate: 0.300%
External FA count: 120 / 40000
Parameters: 1,100,897
```

## Validate A Release Artifact

```bash
python - <<'PY'
import torch
obj = torch.load("exports/solospeak_stage7_deployable_corrected.pt", map_location="cpu")
assert abs(float(obj["tau_on"]) - 0.27) <= 0.03
print("tau_on =", obj["tau_on"])
PY
```

The uncorrected `exports/solospeak_stage7_deployable.pt` is an intermediate artifact.
It is not the final recommended model.
