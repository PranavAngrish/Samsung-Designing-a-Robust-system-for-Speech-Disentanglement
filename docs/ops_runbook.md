# Operations Runbook

## Before A Demo

1. Run `make test`.
2. Run `make export`.
3. Confirm `artifacts/ValidationReport.json` has `"passed": true`.
4. Replace the placeholder `artifacts/silero_vad_v4.onnx` with the real pinned VAD
   artifact if running a real microphone demo.
5. Enroll the demo user with `python demo/cli/live_demo.py --enroll --user pranav
   --keyword "hey prism" --mic`.

## If Export Fails

Check that `checkpoints/latest.pt` points to a Stage 6 checkpoint. If ONNX export reports
an adaptive-pooling error, verify that `solospeak/deployment/export_onnx.py` still swaps
the dynamic frequency pool for the fixed deployment pool before export.

## If Validation Fails

Gate 1 means the artifact is too large. Gate 2 or 3 means an unsupported ONNX contract
changed. Gate 4 means xRT is too slow. Gates 5-7 mean evaluation quality is below the
hard ship threshold. Gate 8 means the model exceeded the parameter budget. Gate 9 means
the ONNX I/O contract changed. Gate 10 means INT8 degraded too much versus FP32.

Do not mark a failed gate as passed. Fix the model, quantization settings, or checkpoint
and rerun `make export`.

## If The Demo False Accepts

1. Re-enroll with cleaner recordings.
2. Raise the per-user threshold in the saved profile temporarily.
3. Use `--model-slot previous` to demonstrate rollback if a previous artifact exists.
4. Record the scenario as a Q2, Q3, Q4, or replay failure for later training data.

## If The Demo Misses The Wake Phrase

1. Check microphone input and sample rate.
2. Re-enroll with three to five clear utterances.
3. Lower the profile threshold only for debugging.
4. Run the smoke listen path without `--mic` to confirm the ONNX session boots.

## Privacy Rule

Never upload raw enrollment or listening audio from a user device. Only local aggregate
metrics may be exported, and future telemetry must use the DP mechanism in
`solospeak/observability/dp_noise.py`.
