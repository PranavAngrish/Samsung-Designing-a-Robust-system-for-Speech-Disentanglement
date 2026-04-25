# SoloSpeak

Speaker-personalized custom wake-word detection. Samsung ennovateX AX Hackathon 2026, Problem #04.

SoloSpeak is a sub-3M-parameter on-device keyword spotter that only wakes for *your* voice saying *your* custom word — robust from −5 dB to 30 dB SNR, from 0.5 m to 5 m, trained end-to-end on open data, released under Apache-2.0.

## Status

**Phase 0 — Foundation.** Repo scaffold, CI, and skeleton stubs in place. Training not yet started.

| KPI | Target | Status |
|-----|--------|--------|
| True Acceptance (Clean) | ≥ 99% | — |
| True Acceptance (Noisy) | ≥ 90% | — |
| False Acceptance | < 1/hr | — |
| Parameters | < 3 M | — |
| xRT (INT8, ARM) | < 0.2 | — |

## Architecture

A single shared BC-ResNet-8 encoder with two orthogonal output heads — a **content head** (what was said) and a **speaker head** (who said it) — trained jointly with a disentanglement objective (adversarial gradient reversal + orthogonality penalty). At inference, a learned gating fusion combines the two cosine similarities into a single accept/reject decision.

See [docs/architecture.md](docs/architecture.md) for the full design.

## Quick Start

```bash
git clone https://github.com/pranavangrish/solospeak.git
cd solospeak
make install-dev
make test
```

## Training

```bash
# Download datasets (requires VoxCeleb academic access — see data/README.md)
make download-data
make prepare-manifests

# Run training stages sequentially
make stage-1   # backbone pretrain on GSC-v2, ~3 days
make stage-2   # dual-head joint training, ~5 days
make stage-3   # disentanglement, ~2 days
make stage-4   # noise robustness, ~3 days
make stage-5   # fusion MLP, ~1 day
make stage-6   # quantization-aware fine-tune, ~1 day
```

## Evaluation

```bash
make eval          # full KPI suite → reports/kpi_final.json
make ablation      # 7-config ablation study → reports/ablation_table.md
```

## Deployment

```bash
make export        # PyTorch → ONNX FP32 → ONNX INT8, runs 9 validation gates
```

## License

Apache-2.0. See [LICENSE](LICENSE).

All training datasets are CC-BY 4.0 or CC0 compatible with Apache-2.0 distribution. See [data/README.md](data/README.md) for per-dataset license details.

## Citation

```bibtex
@software{solospeak2026,
  author = {Pranav Angrish},
  title  = {SoloSpeak: Speaker-Personalized Custom Wake-Word Detection},
  year   = {2026},
  url    = {https://github.com/pranavangrish/solospeak},
}
```
