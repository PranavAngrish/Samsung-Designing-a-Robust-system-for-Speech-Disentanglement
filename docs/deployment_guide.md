# Deployment Guide

## Export

Run:

```bash
make export
```

This calls `scripts/export_and_validate.py`, which performs FP32 ONNX export, Conv-only
INT8 quantization, validation gates, and OTA package assembly.

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
present. In a real release, it should contain the last known-good model.

## Validation Gates

The final artifact must pass:

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

Gate 10 runs only when an FP32 path is supplied. `make export` supplies it.

## OTA Package

The OTA zip contains:

```text
MANIFEST.json
solospeak_int8.onnx
silero_vad_v4.onnx
README.txt
```

Signing is outside this repository. The hackathon package is unsigned.

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
