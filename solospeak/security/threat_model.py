"""SoloSpeak threat model documentation (as code-comment).

IN-SCOPE THREATS:
    1. Replay attack — attacker records and plays back the user's wake word
    2. Voice cloning — attacker synthesises the user's voice (ElevenLabs, Vall-E)
    3. Adversarial audio — imperceptible perturbations causing false accept
    4. Phonetic impersonation — human mimics the user's voice and accent
    5. Template theft — attacker extracts template files from a compromised device
    6. Model extraction — attacker reconstructs model weights (IP theft)

OUT-OF-SCOPE THREATS (explicit non-goals):
    1. Physical-layer microphone access (device owner compromise)
    2. Side-channel attacks on the DSP co-processor
    3. DolphinAttack-style ultrasonic injection (handled by Samsung mic DSP)

MITIGATIONS (see architecture doc §7.7 for details):
    Replay:      liveness via room acoustic variation; replay cooldown; contextual rejection
    Cloning:     speaker manifold is tight (L2-normalized 128-dim); anti-spoofing head (Phase 2)
    Adversarial: architectural mass; INT8 input quantization as natural defense
    Template:    AES-256 at rest (Android Keystore); 128-dim vector can't reconstruct audio
    Model:       treat as public (Apache-2.0); security comes from template secrecy

SECURITY TEST PASS CRITERIA (Phase 2):
    Replay:  < 20% success rate across 10 diverse playback setups
    Cloning: < 50% success rate against ElevenLabs Professional clone
    FGSM:    quantify epsilon for 50% FAR (whitebox)
    Storage: profile file cannot be decrypted without Android Keystore access
"""

THREAT_MODEL_VERSION = "1.0"
