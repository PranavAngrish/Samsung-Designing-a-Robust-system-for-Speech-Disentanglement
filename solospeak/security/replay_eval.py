"""Replay attack test harness.

Simulates the most practical real-world attack: a recording of the legitimate
user's wake word played back through a speaker in the same room.
Pass criterion: < 20% success rate across 10 diverse playback setups.
"""

from __future__ import annotations

from pathlib import Path


def run_replay_eval(
    checkpoint_path: Path,
    replay_audio_dir: Path,
    enrolled_profiles_dir: Path,
) -> dict[str, float]:
    """Run replay attack evaluation. Returns {setup_id: success_rate}."""
    raise NotImplementedError("Implement in Phase 6")
