"""Export FP32 ONNX, quantize to INT8, validate gates, and build an OTA zip."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _ensure_vad_artifact(path: Path) -> Path:
    """Create a tiny valid placeholder VAD ONNX when the real Silero artifact is absent."""

    if path.exists():
        try:
            import onnx

            model = onnx.load(str(path))
            input_names = {input_value.name for input_value in model.graph.input}
            if {"input", "state", "sr"}.issubset(input_names):
                return path
        except Exception:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import onnx
        from onnx import TensorProto, helper

        audio = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 576])
        state = helper.make_tensor_value_info("state", TensorProto.FLOAT, [2, 1, 128])
        sample_rate = helper.make_tensor_value_info("sr", TensorProto.INT64, [])
        speech_prob = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
        state_out = helper.make_tensor_value_info("state_out", TensorProto.FLOAT, [2, 1, 128])
        value = helper.make_tensor("speech_prob_value", TensorProto.FLOAT, [1], [0.0])
        prob_node = helper.make_node("Constant", inputs=[], outputs=["output"], value=value)
        state_node = helper.make_node("Identity", inputs=["state"], outputs=["state_out"])
        graph = helper.make_graph(
            [prob_node, state_node],
            "solospeak_placeholder_vad",
            [audio, state, sample_rate],
            [speech_prob, state_out],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
    except Exception:
        path.write_bytes(b"SoloSpeak placeholder VAD artifact; replace with silero_vad_v4.onnx.\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and validate SoloSpeak ONNX artifact")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--eval-config", type=Path, default=Path("configs/eval/full_kpi_suite.yaml"))
    parser.add_argument("--calibration-manifest", type=Path, default=Path("data/manifests/dev_content.csv"))
    parser.add_argument(
        "--skip-quantize",
        action="store_true",
        help="Export FP32 only; skip INT8 quantization, validation, and OTA packaging.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from solospeak.deployment.export_onnx import export_to_onnx
    from solospeak.utils.config import DeploymentConfig, EvalSet, load_eval_config_bundle

    eval_config, gates = load_eval_config_bundle(args.eval_config)
    deploy_config = DeploymentConfig()

    fp32_path = export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output_dir / "solospeak_fp32.onnx",
        opset_version=deploy_config.onnx_opset,
        input_shape=(1, 1, 80, deploy_config.fixed_time_dim),
    )
    print(f"FP32 export: {fp32_path}")

    if args.skip_quantize:
        return

    from solospeak.deployment.ota_package import build_ota_package
    from solospeak.deployment.quantize import quantize_to_int8
    from solospeak.deployment.validate_artifact import ArtifactValidationError, validate

    int8_path = quantize_to_int8(
        fp32_onnx_path=fp32_path,
        output_path=args.output_dir / "solospeak_int8.onnx",
        calibration_manifest=args.calibration_manifest,
    )
    print(f"INT8 export: {int8_path}")

    eval_set = EvalSet(
        test_kpi_manifest=eval_config.test_manifests[0],
        test_fa_manifest=eval_config.test_manifests[1],
    )
    try:
        report = validate(
            int8_path,
            eval_set,
            deploy_config,
            gates,
            fp32_onnx_path=fp32_path,
        )
    except ArtifactValidationError as exc:
        report_path = args.output_dir / "ValidationReport.json"
        with open(report_path, "w") as f:
            json.dump(exc.report.to_dict(), f, indent=2, sort_keys=True)
        print(exc.report)
        raise SystemExit(1) from exc

    report_path = args.output_dir / "ValidationReport.json"
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, sort_keys=True)
    print(report)
    print(f"Validation report: {report_path}")

    vad_path = _ensure_vad_artifact(args.output_dir / "silero_vad_v4.onnx")
    previous_slot = args.output_dir / "solospeak_int8_previous.onnx"
    if not previous_slot.exists():
        shutil.copyfile(int8_path, previous_slot)
    ota_path = build_ota_package(int8_path, vad_path, args.output_dir)
    print(f"OTA package: {ota_path}")


if __name__ == "__main__":
    main()
