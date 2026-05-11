# SoloSpeak Architecture

## Goal

SoloSpeak is a speaker-personalized custom wake-word detector. A wake event should fire
only when the enrolled user says the enrolled phrase. The system separates "what was
said" from "who said it" and fuses both scores on device.

## Data Flow

1. Raw 16 kHz mono audio is trimmed or padded to a 1.6 s window.
2. `LogMelExtractor` produces an 80 by 160 log-mel tensor.
3. `SoloSpeakModel` encodes the tensor with a compact residual CNN.
4. The content head emits `z_c`, a 128-dim normalized phrase embedding.
5. The speaker head emits `z_s`, a 128-dim normalized speaker embedding.
6. Enrollment stores one content template and one speaker template per user.
7. Inference computes cosine scores against both templates.
8. `GatedFusionMLP` maps the two scores to a wake probability.
9. Streaming inference applies hysteresis and a short refractory period.

## Model Contract

Training forward:

```python
z_c, z_s = model(mel)
```

Deployment ONNX forward:

```text
Inputs:
  mel: (B, 1, 80, 160)
  content_template: (B, 128)
  speaker_template: (B, 128)

Outputs:
  z_c: (B, 128)
  z_s: (B, 128)
  content_score: (B,)
  speaker_score: (B,)
  fusion_score: (B,)
```

The ONNX graph uses a fixed time dimension and dynamic batch only. Template cosine
normalization clamps the denominator at `1e-8`, so zero templates are valid for pure
embedding extraction.

## Training Stages

Stage 1 trains the backbone on speech-command classification. Stage 2 trains the
dual-head embedding model. Stage 3 turns on disentanglement/probe search. Stage 4 adds
robustness. Stage 4D mines hard Q2 same-word imposters. Stage 5 trains and selects the
fusion MLP. Stage 6 performs the final Stage 5 handoff evaluation/export. Stage 7 tunes
the fusion head against external false accepts, rebuilds internal verification scores,
and applies the corrected joint threshold.

The final submitted artifact is `exports/solospeak_stage7_deployable_corrected.pt` with
`tau_on = 0.27`.

## Calibration

The final notebook found that calibrating on external false accepts alone selected
`tau=0.935`, which rejected too many true accepts. The corrected Stage 7 export sweeps
thresholds jointly over internal Q1/Q2/Q3/Q4 scores and external false-accept scores.

## Deployment

`scripts.export_and_validate` writes `artifacts/solospeak_fp32.onnx`, quantizes Conv
nodes into `artifacts/solospeak_int8.onnx`, runs validation gates, and assembles
`artifacts/solospeak_ota_v1.0.0.zip`. The fusion MLP remains FP32 inside the ONNX graph.

## Privacy Boundary

Enrollment profiles contain only normalized 128-dim templates and a scalar threshold. No
raw audio is stored by default. Local observability counters stay on device unless a
future opt-in telemetry path applies differential privacy noise.

## Current Caveats

The Stage 7 artifact is a production-candidate hackathon result, not a field-certified
commercial wake-word model. It still needs real Samsung-device latency, microphone,
streaming, replay, cloned-voice, and UX validation before product release.
