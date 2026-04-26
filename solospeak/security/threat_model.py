"""Runtime-visible SoloSpeak threat model.

Documented threats:
1. Replay attacks: recorded enrolled-user audio played through a speaker.
2. Imposter attacks: mimicry, voice cloning, and deepfake speech.
3. Adversarial audio: gradient-based perturbations on mel features.
4. Profile theft: extraction of content/speaker templates from device storage.
5. Multi-user confusion: one enrolled user matching another user's keyword/profile.

The detailed attack vector, mitigation, and residual-risk table lives in
``docs/threat_model.md``. This module exists so deployed/demo code can state the
security scope of the current build without parsing markdown.
"""

from __future__ import annotations


THREAT_MODEL_VERSION = "1.0"

_THREATS = [
    "replay_attacks",
    "imposter_voice_clone_mimicry",
    "adversarial_audio",
    "profile_theft",
    "multi_user_confusion",
]


def documented_threats() -> list[str]:
    """Return the threats explicitly documented for this build."""

    return list(_THREATS)

