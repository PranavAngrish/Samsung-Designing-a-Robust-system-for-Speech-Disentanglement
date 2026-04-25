"""DANN-style gradient reversal for disentanglement.

Forward: identity (pass-through).
Backward: multiply gradient by -lambda.

Purpose: train a probe head to predict speaker from z_c, then reverse gradient
so the content head is penalized for making z_c speaker-predictable, and vice versa.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx,
                x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_  # type: ignore[attr-defined]
        return x.view_as(x)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx,  # type: ignore[override]
                 grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None  # type: ignore[attr-defined]


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal with given lambda."""
    return cast(torch.Tensor, GradientReversalFunction.apply(x, lambda_))  # type: ignore[no-untyped-call]


class AdversarialProbeHead(nn.Module):
    """Probe head trained to predict the 'wrong' attribute.

    Content probe: predicts speaker-ID from z_c (should fail after disentanglement).
    Speaker probe: predicts word-class from z_s (should fail after disentanglement).

    Used in Stage 3 only — discarded at end of stage.
    """

    def __init__(self, embed_dim: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, z: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        """Apply gradient reversal then linear classifier."""
        return cast(torch.Tensor, self.fc(grad_reverse(z, lambda_)))


def adversarial_lambda(step: int, ramp_steps: int, max_lambda: float = 0.1) -> float:
    """Linear ramp-up of adversarial weight from 0 to max_lambda over ramp_steps."""
    if ramp_steps <= 0:
        return max_lambda
    return max_lambda * min(1.0, step / ramp_steps)
