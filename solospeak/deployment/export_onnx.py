"""PyTorch → ONNX export.

Strips training-only components before export:
    - Auxiliary classification heads
    - Adversarial probe heads
    - Gradient reversal layers
    - BatchNorm statistics (fused into conv weights in eval mode)

Input:  checkpoints/stage6_qat.pt
Output: artifacts/solospeak_fp32.onnx  (~18 MB before quantization)
"""

from __future__ import annotations

from pathlib import Path


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    opset_version: int = 17,
    input_shape: tuple[int, int, int, int] = (1, 1, 80, 160),
) -> Path:
    """Export Stage 6 checkpoint to ONNX FP32.

    Verifies output shapes: z_c (1, 128) and z_s (1, 128).
    Dynamic axes: batch dimension only.
    """
    raise NotImplementedError("Implement in Phase 5")
