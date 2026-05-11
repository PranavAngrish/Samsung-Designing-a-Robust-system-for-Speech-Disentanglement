# Deployment Guide

## Final Submitted Artifact

The hackathon submission artifact is:

```text
exports/solospeak_stage7_deployable_corrected.pt
```

It is a lightweight PyTorch deployable package containing the Stage 7 model weights,
deployment config, final `tau_on = 0.27`, and Stage 7 calibration metadata. This is the
artifact to attach to the release and use for final metric claims.

## ONNX And INT8 Export

For a mobile-style deployment package, export from the corrected Stage 7 checkpoint:

```bash
python -m scripts.export_and_validate \
  --checkpoint checkpoints/stage7_final_corrected.pt \
  --output-dir artifacts
```

The legacy shortcut below is still available, but it expects `checkpoints/latest.pt` to
point at the intended checkpoint:

```bash
make export
```

`scripts/export_and_validate.py` performs FP32 ONNX export, Conv-only INT8 quantization,
validation gates, and OTA package assembly.

## Artifact Layout

Generated files:

```text
artifacts/solospeak_fp32.onnx
artifacts/solospeak_int8.onnx
artifacts/solospeak_int8_previous.onnx
artifacts/silero_vad_v4.onnx
artifacts/ValidationReport.json
artifacts/solospeak_ota_v1.0.0.zip
```

The `previous` slot is created as a local rollback simulation if no earlier artifact is
present. In a real release, it should contain the last known-good signed model.

## Validation Gates

The ONNX/INT8 deployment package must pass:

1. File size <= 5 MB.
2. ONNX opset >= 17.
3. No unsupported mobile-control-flow ops.
4. xRT p95 <= 0.20.
5. TA clean >= 0.92.
6. TA noisy macro >= 0.80.
7. FA per hour per user <= 2.0.
8. Parameter count <= 3,000,000.
9. Required output names and shapes.
10. INT8 vs FP32 degradation <= 1.0 percentage point.

Gate 10 runs only when an FP32 path is supplied. `scripts.export_and_validate` supplies
it during the full export command above.

## OTA Package

The OTA zip contains:

```text
MANIFEST.json
solospeak_int8.onnx
silero_vad_v4.onnx
README.txt
```

Signing is outside this repository. The hackathon package is unsigned. The smoke
workflow can generate a placeholder `silero_vad_v4.onnx`; replace it with the real
pinned Silero VAD artifact before presenting a real microphone demo.

## Rollback Demo

Run the demo with the current slot:

```bash
python demo/cli/live_demo.py --model-slot current --eval
```

Run the demo with the previous slot:

```bash
python demo/cli/live_demo.py --model-slot previous --eval
```

If the previous slot is absent, the CLI prints a warning and falls back to the current
slot for boot testing. The real two-slot Android rollout would switch between two signed
ONNX artifacts managed by platform deployment infrastructure.
