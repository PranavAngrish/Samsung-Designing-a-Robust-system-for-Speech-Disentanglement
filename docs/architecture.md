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

Stage 1 trains the backbone on speech-command classification. Stages 2-4 train the dual
content/speaker embedding heads with contrastive, orthogonality, adversarial, and
robustness objectives. Stage 5 trains the fusion MLP on quadrant trials. Stage 6 prepares
the checkpoint for quantization-aware deployment.

## Deployment

`make export` writes `artifacts/solospeak_fp32.onnx`, quantizes Conv nodes only into
`artifacts/solospeak_int8.onnx`, runs validation gates, and assembles
`artifacts/solospeak_ota_v1.0.0.zip`. The fusion MLP remains FP32 inside the ONNX graph.

## Privacy Boundary

Enrollment profiles contain only normalized 128-dim templates and a scalar threshold. No
raw audio is stored by default. Local observability counters stay on device unless a
future opt-in telemetry path applies differential privacy noise.

## Current Caveats

The current repository has smoke data and smoke checkpoints. Final KPI claims require
real data download, full training, real profile templates, and the real pinned Silero VAD
artifact.
