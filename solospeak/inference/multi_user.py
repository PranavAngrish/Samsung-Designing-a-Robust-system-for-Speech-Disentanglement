"""Shared-device multi-profile inference.

The expensive BC-ResNet-8 forward pass runs ONCE regardless of enrolled user count.
All per-user comparisons are cosine dot products (128-dim — essentially free).
"""

from __future__ import annotations

import numpy as np

from solospeak.utils.types import UserProfile, WakeEvent


class MultiUserDetector:
    """Detect wake events for any of N enrolled users with a single forward pass.

    📋 CONTRACT
        step(mel: np.ndarray) → WakeEvent | None
        Returns the first user whose fusion score exceeds their threshold,
        or None if no user triggered.
    """

    def __init__(
        self,
        onnx_session: object,
        user_profiles: dict[str, UserProfile],
    ) -> None:
        raise NotImplementedError("Implement in Phase 5")

    def step(self, mel: np.ndarray) -> WakeEvent | None:
        raise NotImplementedError("Implement in Phase 5")

    def add_user(self, profile: UserProfile) -> None:
        raise NotImplementedError("Implement in Phase 5")

    def remove_user(self, user_id: str) -> None:
        raise NotImplementedError("Implement in Phase 5")
