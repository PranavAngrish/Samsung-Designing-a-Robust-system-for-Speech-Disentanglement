# Release Manifest

Large checkpoints and deployable model files are not committed to git. Attach them to a
GitHub Release or Kaggle Dataset and verify the released files against this manifest.

Recommended release tag: `v1.0.0-stage7-corrected`

## Final Production Artifacts

| Artifact | Role | SHA-256 |
|---|---|---|
| `exports/solospeak_stage7_deployable_corrected.pt` | Final recommended deployable, `tau_on ~= 0.27` | `e1121ba113a8d8a523841538b14b33d58bfcb5265e219ba3afaf759ea1cc91fc` |
| `checkpoints/stage7_final_corrected.pt` | Full corrected Stage 7 checkpoint | `1e9f30faebf871cc9416aa8eb1d70e8b182b936e8822166f5132469638aad4f6` |
| `checkpoints/stage7_fusion.pt` | Stage 7 fusion-head tuned checkpoint before joint tau correction | `2a11f1b122ca6f42704dfc28c3b7d55f4995cfa1a61093c103ef59cb401a633b` |
| `checkpoints/stage6_final.pt` | Stage 6 final evaluation/export checkpoint | `732ede7454748301b1bbf8c03fd69857ada1ca6d5f7483c2935419f6a32d8d65` |
| `checkpoints/stage5_fusion.pt` | Stage 5 best fusion checkpoint | `b32d7d1e338ecdd2024f6c5f48413112ac4138686fcb8f005edf39f47083f4ec` |
| `checkpoints/stage4d_hardq2_mining_balanced.pt` | Stage 4D hard-Q2 mining checkpoint used by Stage 5 | `d36f5b0ad0ac4659b97ffdb6c6a595376fccd0729b88cd6f7fd3939912efac46` |

## Validation Files

| Report | Role | SHA-256 |
|---|---|---|
| `reports/stage7_joint_threshold_calibration_summary.json` | Joint internal plus external calibration summary | `ee481f27f455d6760f8d81c2719e8206390f623218ed46d0fdeba2f9fe0f591b` |
| `reports/stage7_external_fa_summary.json` | Stage 7 external FA score summary at the pre-correction Stage 7 tau | `f935f8abbab75ba86b5a9a0da36f4801e394a88d275ef529f2032253823d3ea0` |

`stage7_external_fa_summary.json` is useful provenance for the Stage 7 tuning run. The
judge-facing final threshold and final metrics come from
`stage7_joint_threshold_calibration_summary.json`.

## Expected Final Metrics

```text
tau_on: 0.27
TA clean: 93.97%
Q2 rejection: 95.17%
Q3 rejection: 97.83%
Q4 rejection: 100.00%
Quadrant minimum: 93.97%
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
