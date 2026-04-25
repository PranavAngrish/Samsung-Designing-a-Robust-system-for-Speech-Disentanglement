"""Learning rate schedulers and loss-weight ramp-up helpers."""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LRScheduler:
    """Linear warmup then cosine decay to 0."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def linear_ramp(step: int, ramp_steps: int, final_value: float) -> float:
    """Linear ramp from 0 to final_value over ramp_steps."""
    if ramp_steps <= 0:
        return final_value
    return final_value * min(1.0, step / ramp_steps)
