"""PyTorch to FP32 ONNX export for the deployable SoloSpeak graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from solospeak.models.solospeak import SoloSpeakModel
from solospeak.utils.config import SoloSpeakConfig


_OUTPUT_NAMES = ["z_c", "z_s", "content_score", "speaker_score", "fusion_score"]


class ExportWrapper(nn.Module):
    """Wrap ``SoloSpeakModel`` with template scoring for ONNX export."""

    def __init__(self, model: SoloSpeakModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        mel: torch.Tensor,
        content_template: torch.Tensor,
        speaker_template: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_c, z_s = self.model(mel)
        ct = content_template / content_template.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        st = speaker_template / speaker_template.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        content_score = (z_c * ct).sum(dim=-1)
        speaker_score = (z_s * st).sum(dim=-1)
        fusion_score = self.model.forward_fusion(content_score, speaker_score)
        return z_c, z_s, content_score, speaker_score, fusion_score


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected checkpoint dict in {path}, got {type(loaded)!r}")
    return cast(dict[str, Any], loaded)


def _config_from_checkpoint(checkpoint: dict[str, Any]) -> SoloSpeakConfig:
    snapshot = checkpoint.get("config_snapshot")
    if isinstance(snapshot, dict):
        return SoloSpeakConfig.model_validate(snapshot)
    return SoloSpeakConfig.from_yaml("configs/defaults.yaml")


def load_deployable_model(
    checkpoint_path: Path,
    *,
    fuse_bn: bool = True,
) -> SoloSpeakModel:
    """Load a Stage-6 checkpoint into a clean deployable ``SoloSpeakModel``."""

    checkpoint = _load_checkpoint(checkpoint_path)
    config = _config_from_checkpoint(checkpoint)
    model = SoloSpeakModel(config).eval()

    state = checkpoint.get("model_state", checkpoint)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint {checkpoint_path} does not contain a state dict.")
    model.load_state_dict(cast(dict[str, torch.Tensor], state), strict=False)
    model.eval()

    if fuse_bn and hasattr(model.backbone, "fuse_model"):
        model.backbone.fuse_model()
        model.eval()
    return model


def _verify_wrapper_outputs(
    wrapper: ExportWrapper,
    input_shape: tuple[int, int, int, int],
    config: SoloSpeakConfig,
) -> None:
    mel = torch.zeros(input_shape, dtype=torch.float32)
    content_template = torch.zeros((input_shape[0], config.heads.content_dim), dtype=torch.float32)
    speaker_template = torch.zeros((input_shape[0], config.heads.speaker_dim), dtype=torch.float32)
    with torch.no_grad():
        outputs = wrapper(mel, content_template, speaker_template)
    expected_shapes = (
        (input_shape[0], config.heads.content_dim),
        (input_shape[0], config.heads.speaker_dim),
        (input_shape[0],),
        (input_shape[0],),
        (input_shape[0],),
    )
    for name, output, expected in zip(_OUTPUT_NAMES, outputs, expected_shapes):
        if tuple(output.shape) != expected:
            raise RuntimeError(f"{name} has shape {tuple(output.shape)}, expected {expected}.")
        if torch.isnan(output).any():
            raise RuntimeError(f"{name} produced NaN for zero-template export check.")


def _replace_dynamic_freq_pool(model: SoloSpeakModel, input_shape: tuple[int, int, int, int]) -> None:
    # The training backbone uses AdaptiveAvgPool2d((1, None)); ONNX requires a fixed
    # output size. For the deployment window, the backbone frequency dimension is 5.
    freq_after_strides = max(1, input_shape[2] // 16)
    setattr(
        model.backbone,
        "freq_pool",
        nn.AvgPool2d(kernel_size=(freq_after_strides, 1), stride=(1, 1)),
    )


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    opset_version: int = 17,
    input_shape: tuple[int, int, int, int] = (1, 1, 80, 160),
) -> Path:
    """Export a Stage-6 checkpoint as FP32 ONNX with fixed time dim and dynamic batch."""

    checkpoint = _load_checkpoint(checkpoint_path)
    config = _config_from_checkpoint(checkpoint)
    model = SoloSpeakModel(config).eval()
    state = checkpoint.get("model_state", checkpoint)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint {checkpoint_path} does not contain a state dict.")
    model.load_state_dict(cast(dict[str, torch.Tensor], state), strict=False)
    if hasattr(model.backbone, "fuse_model"):
        model.backbone.fuse_model()
    _replace_dynamic_freq_pool(model, input_shape)
    model.eval()

    wrapper = ExportWrapper(model).eval()
    _verify_wrapper_outputs(wrapper, input_shape, config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mel = torch.zeros(input_shape, dtype=torch.float32)
    content_template = torch.zeros((input_shape[0], config.heads.content_dim), dtype=torch.float32)
    speaker_template = torch.zeros((input_shape[0], config.heads.speaker_dim), dtype=torch.float32)
    dynamic_axes = {
        "mel": {0: "batch"},
        "content_template": {0: "batch"},
        "speaker_template": {0: "batch"},
        "z_c": {0: "batch"},
        "z_s": {0: "batch"},
        "content_score": {0: "batch"},
        "speaker_score": {0: "batch"},
        "fusion_score": {0: "batch"},
    }

    torch.onnx.export(
        wrapper,
        (mel, content_template, speaker_template),
        output_path,
        opset_version=opset_version,
        input_names=["mel", "content_template", "speaker_template"],
        output_names=_OUTPUT_NAMES,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    if not output_path.exists():
        raise RuntimeError(f"ONNX export did not create {output_path}.")
    return output_path


def export_onnx(
    checkpoint_path: Path,
    output_path: Path,
    opset: int = 17,
    fixed_time_dim: int = 160,
    include_fusion: bool = True,
) -> Path:
    """Compatibility wrapper matching the public deployment API."""

    if not include_fusion:
        raise ValueError("SoloSpeak deployment export always includes the fusion score path.")
    return export_to_onnx(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        opset_version=opset,
        input_shape=(1, 1, 80, fixed_time_dim),
    )
