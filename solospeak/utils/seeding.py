"""Deterministic seeding. Call seed_everything() at the start of every script."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set all RNG seeds for reproducibility.

    Note: exact bit-level determinism across CUDA versions is not guaranteed.
    We target ±0.2 pp absolute accuracy tolerance on regression tests.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
