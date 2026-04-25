"""Template storage, serialisation, and encryption.

Per-user profile is ~1.1 KB:
    content_template: 128 × float32 = 512 bytes
    speaker_template: 128 × float32 = 512 bytes
    tau:              float32        =   4 bytes
    metadata:         ~80 bytes JSON
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from solospeak.utils.types import UserProfile


def save_profile(profile: UserProfile, path: Path) -> None:
    """Serialise UserProfile to a JSON file (plaintext — use OS-level encryption)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "user_id": profile.user_id,
        "keyword_text": profile.keyword_text,
        "content_template": profile.content_template.tolist(),
        "speaker_template": profile.speaker_template.tolist(),
        "tau": profile.tau,
        "model_version": profile.model_version,
    }
    with open(path, "w") as f:
        json.dump(data, f)


def load_profile(path: Path) -> UserProfile:
    """Deserialise UserProfile from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return UserProfile(
        user_id=data["user_id"],
        keyword_text=data["keyword_text"],
        content_template=np.array(data["content_template"], dtype=np.float32),
        speaker_template=np.array(data["speaker_template"], dtype=np.float32),
        tau=float(data["tau"]),
        model_version=data["model_version"],
    )
