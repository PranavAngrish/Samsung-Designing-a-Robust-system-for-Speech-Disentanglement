# SoloSpeak

Speaker-personalized custom wake-word detection for Samsung ennovateX AX Hackathon
2026, Problem #04.

SoloSpeak is an on-device keyword spotter that wakes only when the enrolled user says
the enrolled phrase. It separates **what was said** from **who said it**, then fuses both
scores in a compact deployment graph.

## Final Result

Final artifact: `exports/solospeak_stage7_deployable_corrected.pt`

Final threshold: `tau = 0.27`

Validation protocol: internal GSC Q1/Q2/Q3/Q4 verification plus 40,000 external
false-accept trials from Common Voice, LibriSpeech, background noise, and UrbanSound8K.

| Metric | Final Stage 7 Value |
|---|---:|
| TA clean | 93.97% |
| Q2 imposter rejection | 95.17% |
| Q3 wrong-word rejection | 97.83% |
| Q4 background rejection | 100.00% |
| Quadrant minimum | 93.97% |
| External FA rate | 0.300% |
| External FA count | 120 / 40,000 |
| Parameters | 1.10M |

The corrected Stage 7 export uses joint internal + external threshold calibration. An
earlier external-only calibration selected `tau = 0.935`, which suppressed true accepts;
that path is intentionally not used for the final artifact.

This is a production-candidate hackathon artifact, not a field-certified commercial
wake-word model. It passed large external false-accept validation, but still needs
real-device latency, microphone, streaming, and UX validation before product use.

## Links

- Repository: https://github.com/pranavangrish/solospeak
- Demo video: `PASTE_UNLISTED_YOUTUBE_URL_HERE`
- Release artifacts: https://github.com/pranavangrish/solospeak/releases
- Architecture: [docs/architecture.md](docs/architecture.md)
- Reproducibility: [docs/reproducibility.md](docs/reproducibility.md)
- Deployment guide: [docs/deployment_guide.md](docs/deployment_guide.md)
- Final report outline: [docs/final_report_outline.md](docs/final_report_outline.md)
- Submission email: [docs/submission_email.md](docs/submission_email.md)

## Architecture

SoloSpeak uses a shared residual CNN backbone with two orthogonal 128-dim heads:

- Content head: phrase identity, or "what was said".
- Speaker head: enrolled user identity, or "who said it".

At inference, the deployable ONNX graph receives `mel`, `content_template`, and
`speaker_template`, returns both embeddings, computes safe cosine scores, and emits a
fusion probability. The streaming demo applies hysteresis so a wake event fires once per
decisive window.

## Quick Start

```bash
make install-dev
make download-data-smoke
make prepare-manifests-smoke
python -m scripts.run_stage --all
make eval
make ablation
make export
make test
```

## Reproduce Final Result

Large generated files are not committed to git. Full reproduction regenerates them from
the source pipeline and attached datasets.

```bash
make download-data
make prepare-manifests

python -m scripts.run_production_pipeline \
  --config configs/training/production.yaml \
  --common-voice-root /kaggle/input/datasets/organizations/mozillaorg/common-voice \
  --librispeech-root /kaggle/input/datasets/a24998667/librispeech \
  --background-noise-root /kaggle/input/datasets/axondata/background-noise-detection-dataset \
  --urbansound-root /kaggle/input/datasets/chrisfilo/urbansound8k
```

This produces the Stage 1-7 checkpoints, `reports/production_pipeline_summary.json`,
`reports/stage7_joint_threshold_calibration_summary.json`, and the corrected deployable
`exports/solospeak_stage7_deployable_corrected.pt`.

Fast wiring smoke test:

```bash
python -m scripts.run_production_pipeline --config configs/training/production.yaml --smoke
```

Stage-specific reproduction:

```bash
python -m scripts.run_stage --stage 5 --resume-from checkpoints/stage4d_hardq2_mining_balanced.pt
python -m scripts.run_stage --stage 6 --resume-from checkpoints/stage5_fusion.pt
python -m scripts.run_stage --stage 7 --resume-from checkpoints/stage6_final.pt \
  --external-common-voice-root /data/common_voice \
  --external-librispeech-root /data/librispeech \
  --external-background-noise-root /data/background_noise \
  --external-urbansound-root /data/urbansound8k
python -m scripts.run_stage7_verify --checkpoint checkpoints/stage7_fusion.pt
python -m scripts.run_stage7_finalization
```

Required external FA datasets for the final Stage 7 run are Common Voice English,
LibriSpeech, Background Noise Detection Dataset, and UrbanSound8K. The Kaggle paths
above are examples only; package code reads roots from config or CLI arguments.

Validate the corrected threshold:

```bash
python - <<'PY'
import torch
obj = torch.load("exports/solospeak_stage7_deployable_corrected.pt", map_location="cpu")
assert abs(float(obj["tau_on"]) - 0.27) <= 0.03
print("tau_on =", obj["tau_on"])
PY
```

To run only Stage 7 from an existing Stage 6 checkpoint with Kaggle-style CLI roots:

```bash
python -m scripts.run_stage --stage 7 \
  --resume-from checkpoints/stage6_final.pt \
  --external-common-voice-root /data/common_voice \
  --external-librispeech-root /data/librispeech \
  --external-background-noise-root /data/background_noise \
  --external-urbansound-root /data/urbansound8k
python -m scripts.run_stage7_verify --checkpoint checkpoints/stage7_fusion.pt
python -m scripts.run_stage7_finalization
```

Generated raw data, checkpoints, profiles, reports, and ONNX artifacts are gitignored and
should be attached to a Release or Kaggle Dataset rather than committed.

## Test Status

Install the dev environment before claiming a tested run:

```bash
pip install -e ".[dev]"
make test
```

In a bare container without project dependencies, tests can fail at import time, for
example on missing `lmdb`, `torch`, or `pytest`. That is an environment issue, not a
passing test result.

## Deployment

```bash
make export
```

This produces:

- `artifacts/solospeak_fp32.onnx`
- `artifacts/solospeak_int8.onnx`
- `artifacts/solospeak_int8_previous.onnx`
- `artifacts/ValidationReport.json`
- `artifacts/solospeak_ota_v1.0.0.zip`

The smoke workflow creates a valid placeholder `silero_vad_v4.onnx`. Replace it with the
real pinned Silero VAD artifact before recording the live demo.

## Demo

Smoke enrollment:

```bash
python demo/cli/live_demo.py --enroll --user pranav --keyword "hey prism"
```

Real microphone enrollment:

```bash
python demo/cli/live_demo.py --enroll --user pranav --keyword "hey prism" --mic
```

Rollback slot boot check:

```bash
python demo/cli/live_demo.py --model-slot previous --eval
```

## Submission Helpers

```bash
make submission-package
```

This regenerates:

- [docs/demo_video_script.md](docs/demo_video_script.md)
- [docs/final_report_outline.md](docs/final_report_outline.md)
- [docs/release_manifest.md](docs/release_manifest.md)
- [docs/submission_email.md](docs/submission_email.md)

## License

Apache-2.0. See [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff).
