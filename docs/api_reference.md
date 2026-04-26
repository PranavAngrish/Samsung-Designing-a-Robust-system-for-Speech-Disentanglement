# API Reference

This hand-maintained reference lists the public functions used by the hackathon scripts.
It can be replaced by generated `pdoc` output before final submission.

## Deployment

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
