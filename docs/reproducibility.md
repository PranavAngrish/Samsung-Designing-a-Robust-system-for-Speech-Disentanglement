# Reproducibility

## Environment

Use Python 3.10 or 3.11 for the pinned dependencies in `pyproject.toml`. The current
workspace has also been smoke-tested on the local Python environment used during
implementation.

## Smoke Pipeline

From a fresh checkout:

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

The smoke path uses generated audio in `data/raw/smoke/` and small manifests in
`data/manifests/`. It validates code wiring, shapes, reports, export, quantization, and
artifact gates. It does not prove final model accuracy.

## Full Data Pipeline

For final numbers, run the non-smoke data and manifest targets:

```bash
make download-data
make prepare-manifests
python -m scripts.run_stage --all
make eval
make ablation
make export
```

Full training must produce `checkpoints/stage6_qat.pt` and `checkpoints/latest.pt`.
Generated checkpoints, raw data, reports, and ONNX artifacts are intentionally gitignored.

## Determinism

Training and evaluation scripts call seeded paths where practical. Exact bit-for-bit
reproducibility is not guaranteed across PyTorch, CPU, and BLAS versions. Reported
metrics should include seed, config path, checkpoint path, and git commit when used in
the final submission.

## Current Smoke Outputs

The current smoke export produces:

```text
artifacts/solospeak_fp32.onnx
artifacts/solospeak_int8.onnx
artifacts/ValidationReport.json
artifacts/solospeak_ota_v1.0.0.zip
reports/quantization_report.json
```

`artifacts/silero_vad_v4.onnx` is a valid placeholder in the smoke workflow. Replace it
with the real pinned Silero VAD artifact before recording the final demo.
