"""Hysteresis detector with refractory period.

Prevents:
    - Flapping: rises on tau_on, falls only when below tau_off (tau_off < tau_on)
    - Double-fire: 1.5 s refractory period after any positive detection
"""

from __future__ import annotations


class HysteresisDetector:
    """State machine for streaming wake-word detection.

    📋 CONTRACT
        step(score: float) → bool
        Returns True exactly once per wake event.
        Refractory period prevents double-fires.
    """

    def __init__(
        self,
        tau_on: float = 0.75,
        tau_off: float = 0.45,
        refractory_s: float = 1.5,
        stride_s: float = 0.1,
        smoothing_frames: int = 3,
    ) -> None:
        self.tau_on = tau_on
        self.tau_off = tau_off
        self.refractory_frames = int(refractory_s / stride_s)
        self.smoothing_frames = smoothing_frames

        self._active = False
        self._refractory_countdown = 0
        self._score_buffer: list[float] = []

    def step(self, score: float) -> bool:
        """Process one inference frame. Returns True on a new wake event."""
        self._score_buffer.append(score)
        if len(self._score_buffer) > self.smoothing_frames:
            self._score_buffer.pop(0)

        smoothed = max(self._score_buffer)

        # Refractory: block re-firing; reset active + clear history on expiry
        if self._refractory_countdown > 0:
            self._refractory_countdown -= 1
            if self._refractory_countdown == 0:
                self._active = False
                self._score_buffer.clear()
            return False

        # Fall edge: use raw score so stale smoothing history doesn't block deactivation
        if self._active and score < self.tau_off:
            self._active = False
            self._score_buffer.clear()
            return False

        # Rise edge: use smoothed score to tolerate single noisy low frame
        if not self._active and smoothed >= self.tau_on:
            self._active = True
            self._refractory_countdown = self.refractory_frames
            return True

        return False

    def reset(self) -> None:
        self._active = False
        self._refractory_countdown = 0
        self._score_buffer.clear()
