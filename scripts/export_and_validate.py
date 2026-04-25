"""CLI entry point for ONNX export, INT8 quantization, and 9-gate validation.

Usage:
    python -m scripts.export_and_validate --checkpoint checkpoints/stage6_qat.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and validate SoloSpeak ONNX artifact")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--skip-quantize", action="store_true",
                        help="Export FP32 only, skip INT8 quantization")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from solospeak.deployment.export_onnx import export_to_onnx
    fp32_path = export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output_dir / "solospeak_fp32.onnx",
    )
    print(f"FP32 export: {fp32_path}")

    if not args.skip_quantize:
        from solospeak.deployment.quantize import quantize_to_int8
        int8_path = quantize_to_int8(
            fp32_onnx_path=fp32_path,
            output_path=args.output_dir / "solospeak_int8.onnx",
            calibration_manifest=Path("data/manifests/dev_content.csv"),
        )
        print(f"INT8 export: {int8_path}")

        from solospeak.deployment.validate_artifact import validate
        report = validate(int8_path, Path("data/manifests/test_kpi.csv"))
        print(report)
        if not report.passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
