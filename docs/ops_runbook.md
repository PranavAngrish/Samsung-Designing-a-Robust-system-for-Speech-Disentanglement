# Operations Runbook

## Before A Submission Demo

1. Confirm the final release artifact exists:
   `exports/solospeak_stage7_deployable_corrected.pt`.
2. Confirm the corrected threshold:
   `tau_on = 0.27`.
3. Run regression tests in Python 3.10 or 3.11:
   `pytest tests/regression -q`.
4. Export the optional ONNX/INT8 package:
   `python -m scripts.export_and_validate --checkpoint checkpoints/stage7_final_corrected.pt --output-dir artifacts`.
5. Confirm `artifacts/ValidationReport.json` has `"passed": true` if you are claiming
   ONNX/INT8 deployment gates.
6. Replace the placeholder `artifacts/silero_vad_v4.onnx` with the real pinned VAD
   artifact if running a live microphone demo.
7. Enroll the demo user with:
   `python demo/cli/live_demo.py --enroll --user pranav --keyword "hey prism" --mic`.

## If Export Fails

Check that the checkpoint passed to `scripts.export_and_validate` is
`checkpoints/stage7_final_corrected.pt` or another intentionally selected Stage 7
checkpoint. If ONNX export reports an adaptive-pooling error, verify that
`solospeak/deployment/export_onnx.py` still swaps the dynamic frequency pool for the
fixed deployment pool before export.

## If Validation Fails

Gate 1 means the artifact is too large. Gate 2 or 3 means an unsupported ONNX contract
changed. Gate 4 means xRT is too slow. Gates 5-7 mean evaluation quality is below the
hard ship threshold. Gate 8 means the model exceeded the parameter budget. Gate 9 means
the ONNX I/O contract changed. Gate 10 means INT8 degraded too much versus FP32.

Do not mark a failed gate as passed. Fix the model, quantization settings, checkpoint,
or validation inputs and rerun the export.

## If The Demo False Accepts

1. Re-enroll with cleaner recordings.
2. Raise the per-user profile threshold temporarily for the demo.
3. Use `--model-slot previous` to demonstrate rollback if a previous artifact exists.
4. Record the scenario as a Q2, Q3, Q4, replay, or cloned-voice failure for later data.

## If The Demo Misses The Wake Phrase

1. Check microphone input and sample rate.
2. Re-enroll with three to five clear utterances.
3. Confirm the demo is loading the corrected Stage 7 artifact, not the bad
   external-only `tau=0.935` export.
4. Lower the profile threshold only for debugging.
5. Run the smoke listen path without `--mic` to confirm the runtime session boots.

## Privacy Rule

Never upload raw enrollment or listening audio from a user device. Only local aggregate
metrics may be exported, and future telemetry must use the DP mechanism in
`solospeak/observability/dp_noise.py`.
