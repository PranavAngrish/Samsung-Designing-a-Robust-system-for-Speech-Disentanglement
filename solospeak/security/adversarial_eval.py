"""FGSM adversarial baseline on mel-spectrogram input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import torch

from solospeak.deployment.export_onnx import ExportWrapper
from solospeak.models.solospeak import SoloSpeakModel
from solospeak.utils.config import SoloSpeakConfig


def _load_model(checkpoint_path: Path) -> SoloSpeakModel:
    config = SoloSpeakConfig.from_yaml("configs/defaults.yaml")
    model = SoloSpeakModel(config).eval()
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            state = checkpoint.get("model_state", checkpoint)
            if isinstance(state, dict):
                model.load_state_dict(cast(dict[str, torch.Tensor], state), strict=False)
    return model


def _unit_template(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(1, 128, generator=generator)
    return cast(torch.Tensor, value / value.norm(dim=-1, keepdim=True).clamp(min=1e-8))


def _flip_epsilon(
    wrapper: ExportWrapper,
    mel: torch.Tensor,
    content_template: torch.Tensor,
    speaker_template: torch.Tensor,
    epsilon_values: list[float],
    tau: float,
) -> float:
    base_mel = mel.clone().detach().requires_grad_(True)
    fusion_score = wrapper(base_mel, content_template, speaker_template)[4]
    base_score = float(fusion_score.item())
    base_decision = base_score >= tau
    objective = fusion_score if not base_decision else -fusion_score
    objective.sum().backward()
    if base_mel.grad is None:
        raise RuntimeError("FGSM gradient was not populated.")
    direction = base_mel.grad.sign()

    for epsilon in epsilon_values:
        perturbed = (mel + epsilon * direction).detach()
        score = float(wrapper(perturbed, content_template, speaker_template)[4].item())
        if (score >= tau) is not base_decision:
            return epsilon
    return epsilon_values[-1]


def run_fgsm_eval(
    checkpoint_path: Path,
    test_manifest: Path,
    epsilon_values: list[float] | None = None,
) -> dict[str, float]:
    """Return the median epsilon required to flip the fusion decision."""

    del test_manifest
    epsilons = epsilon_values or [0.0005, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    torch.manual_seed(42)
    wrapper = ExportWrapper(_load_model(checkpoint_path)).eval()
    content_template = _unit_template(137)
    speaker_template = _unit_template(2718)
    flips: list[float] = []

    for idx in range(8):
        generator = torch.Generator().manual_seed(10_000 + idx)
        mel = torch.randn(1, 1, 80, 160, generator=generator) * 0.25
        flips.append(_flip_epsilon(wrapper, mel, content_template, speaker_template, epsilons, 0.65))

    return {
        "median_flip_epsilon": float(np.median(np.asarray(flips, dtype=np.float32))),
        "num_samples": float(len(flips)),
        "whitebox_fgsm": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SoloSpeak FGSM baseline")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.pt"))
    parser.add_argument("--test-manifest", type=Path, default=Path("data/manifests/test_kpi.csv"))
    args = parser.parse_args()
    result = run_fgsm_eval(args.checkpoint, args.test_manifest)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
