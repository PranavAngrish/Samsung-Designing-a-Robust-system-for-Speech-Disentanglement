# API Reference

This hand-maintained reference lists the public functions used by the hackathon scripts.
It can be replaced by generated `pdoc` output later, but the entries below are the
interfaces that matter for the Stage 7 submission.

## Training And Reproduction

`scripts.run_stage.main()`

Runs an individual stage. Supported stages are `1`, `2`, `3`, `4`, `4d`, `5`, `6`, and
`7`. Stage `4d` is the hard-Q2 mining stage; Stage `6` is final evaluation/export;
Stage `7` is external false-accept tuning.

`scripts.run_production_pipeline.run_production_pipeline(...)`

Runs the source-controlled production path from Stage 1 through Stage 7 corrected
export. The external false-accept dataset roots can be supplied by CLI or config.

`scripts.run_stage7_verify.main()`

Rebuilds internal Q1/Q2/Q3/Q4 verification scores and rescored external false-accept
scores for a Stage 7 checkpoint.

`scripts.run_stage7_finalization.main()`

Exports the default Stage 7 deployable, runs joint internal plus external threshold
calibration, and writes `exports/solospeak_stage7_deployable_corrected.pt`.

## Stage 7 Evaluation

`solospeak.eval.external_fa.discover_external_fa_audio(spec)`

Discovers Common Voice, LibriSpeech, background-noise, and UrbanSound8K audio files for
external false-accept trials.

`solospeak.eval.external_fa.score_external_fa_rows(...)`

Scores external false-accept rows and writes a CSV with `prob`, `s_c`, `s_s`,
`accepted`, and source metadata.

`solospeak.eval.internal_quadrants.build_gsc_internal_examples(...)`

Builds speaker-disjoint GSC enrollment profiles and Q1/Q2/Q3/Q4 internal examples.

`solospeak.eval.threshold_calibration.calibrate_joint_threshold(...)`

Selects the final Stage 7 threshold using both internal Q1/Q2/Q3/Q4 scores and external
false-accept scores. This is the corrected replacement for the bad external-only
`tau=0.935` path.

## Deployment

`solospeak.deployment.export_stage7.export_stage7_deployable(...)`

Writes the intermediate Stage 7 deployable artifact before joint threshold correction.

`solospeak.deployment.export_stage7.export_corrected_stage7(...)`

Applies the joint-calibrated threshold and writes the final corrected deployable and
checkpoint.

`solospeak.deployment.export_onnx.ExportWrapper`

Wraps `SoloSpeakModel` for ONNX export with inputs `mel`, `content_template`, and
`speaker_template`.

`solospeak.deployment.export_onnx.export_to_onnx(checkpoint_path, output_path,
opset_version=17, input_shape=(1, 1, 80, 160))`

Exports the FP32 ONNX graph.

`solospeak.deployment.quantize.quantize_to_int8(fp32_onnx_path, output_path,
calibration_manifest, per_channel=True)`

Runs Conv-only static ONNX Runtime quantization.

`solospeak.deployment.validate_artifact.validate(onnx_path, eval_set, config=None,
gates=None, fp32_onnx_path=None)`

Runs artifact validation gates.

`solospeak.deployment.ota_package.build_ota_package(int8_onnx, vad_onnx, output_dir,
version="1.0.0")`

Builds the unsigned OTA zip.

## Enrollment And Inference

`solospeak.enrollment.service.EnrollmentService.enroll(user_id, keyword_text, recordings)`

Builds a `UserProfile` from enrollment recordings.

`solospeak.inference.streaming.StreamingInference`

Path-based streaming inference with `enroll_user`, `remove_user`, and `step`.

`solospeak.inference.multi_user.MultiUserInference`

Multi-profile streaming inference that emits at most one wake event per frame.

## Security

`solospeak.security.threat_model.documented_threats()`

Returns the documented threat identifiers.

`solospeak.security.replay_eval.run_replay_eval(checkpoint_path, replay_audio_dir,
enrolled_profiles_dir)`

Returns replay baseline metrics.

`solospeak.security.adversarial_eval.run_fgsm_eval(checkpoint_path, test_manifest,
epsilon_values=None)`

Returns a median FGSM flip epsilon.

## Observability

`solospeak.observability.metrics.SoloSpeakLocalMetrics.to_dict()`

Returns local metrics as JSON-safe primitives.

`solospeak.observability.dp_noise.add_dp_noise(value, sensitivity, epsilon)`

Applies Laplace noise to a scalar metric for a future opt-in telemetry path.
