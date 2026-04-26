# SoloSpeak

Speaker-personalized custom wake-word detection for Samsung ennovateX AX Hackathon
2026, Problem #04.

SoloSpeak is an on-device keyword spotter that wakes only when the enrolled user says
the enrolled phrase. It separates **what was said** from **who said it**, then fuses both
scores in a compact deployment graph.

## Current Status

The repository is complete through Phase 7 at the smoke-validation level: data smoke
pipeline, six training stages, KPI evaluation, ablation table generation, ONNX INT8
export, validation gates, OTA packaging, production-hardening docs, and submission
helpers are wired and tested.

Important caveat: the numbers below are from the current smoke run. Replace them with a
full-data training/evaluation run before making final leaderboard or submission claims.

## KPI Snapshot

| Metric | Current Smoke Value | Hard Gate |
|---|---:|---:|
| TA clean | 100.0% | >= 92.0% |
| TA noisy macro | 100.0% | >= 80.0% |
| FA per hour per user | 0.00 | <= 2.00 |
| Q2 imposter rejection | 100.0% | reported |
| Q3 wrong-word rejection | 100.0% | reported |
| Q4 background rejection | 100.0% | reported |
| Parameters | 1.10M | <= 3.00M |
| INT8 artifact size | 1.47 MB | <= 5.00 MB |
| xRT local p95 | 0.0032 | <= 0.20 |
| INT8 vs FP32 degradation | 0.0 pp | <= 1.0 pp |

Generated reports live in `reports/` after running `make eval`, `make ablation`, and
`make export`.

## Links

- Repository: https://github.com/pranavangrish/solospeak
- Demo video: `PASTE_UNLISTED_YOUTUBE_URL_HERE`
- Release artifacts: https://github.com/pranavangrish/solospeak/releases/tag/v1.0.0-phase2
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

## Full Data Run

```bash
make download-data
make prepare-manifests
python -m scripts.run_stage --all
make eval
make ablation
make export
```

Full data is required for final accuracy claims. Generated raw data, checkpoints, reports,
and ONNX artifacts are intentionally gitignored and should be attached to the GitHub
Release rather than committed.

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
