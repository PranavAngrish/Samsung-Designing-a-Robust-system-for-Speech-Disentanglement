"""Disentanglement verification via linear probe classifiers.

For disentanglement to be claimed:
    - Speaker probe on z_c: accuracy <= 20% (speaker info removed from content space)
    - Word probe on z_s:    accuracy <= 20% (content info removed from speaker space)

Procedure:
    1. Freeze the trained encoder
    2. Train a small 2-layer MLP probe on (z_c → speaker_id)
    3. Report dev accuracy
    4. Same for (z_s → word_class)

If probe succeeds (high accuracy) → disentanglement failed.
If probe fails (low accuracy)     → disentanglement succeeded.
"""

from __future__ import annotations

from pathlib import Path


def run_probe_eval(
    checkpoint_path: Path,
    dev_manifest: Path,
    probe_epochs: int = 10,
) -> dict[str, float]:
    """Train and evaluate both probes. Returns dict with:
        speaker_probe_on_content: float  (should be <= 0.20)
        word_probe_on_speaker:    float  (should be <= 0.20)
    """
    raise NotImplementedError("Implement in Phase 4")
