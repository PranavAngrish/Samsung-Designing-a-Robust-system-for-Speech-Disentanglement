# Threat Model

## Scope

This document names the security risks SoloSpeak v1.0 explicitly considers. It is not a
claim that the current smoke artifact is secure against all attacks. The goal is to make
the risks measurable and honest for the hackathon submission.

## Replay Attacks

Attack vector: an attacker records the enrolled user saying the wake phrase and plays it
through a nearby speaker.

Current mitigation in v1.0: none beyond normal acoustic mismatch and the existing
content/speaker fusion threshold. `solospeak.security.replay_eval` provides a runnable
baseline metric.

Planned mitigation: add an anti-spoofing or replay-liveness head, train with room replay
captures, and require replay stress tests before release.

Residual risk: high until real replay data is collected and evaluated.

## Imposter Attacks

Attack vector: another person says the same phrase, mimics the user, or uses a synthetic
voice clone.

Current mitigation in v1.0: the speaker embedding head and fusion MLP require both phrase
and speaker similarity. Phase 4 reports Q2 imposter rejection.

Planned mitigation: add voice-clone stress data and report rejection on cloned and mimicry
trials separately.

Residual risk: medium. Voice cloning can improve quickly and should be tracked as a
moving benchmark.

## Adversarial Audio

Attack vector: a white-box attacker perturbs mel features to force accept or reject.

Current mitigation in v1.0: no certified defense. INT8 quantization may reduce some small
perturbations, but it is not a guarantee. `solospeak.security.adversarial_eval` emits a
white-box FGSM baseline.

Planned mitigation: adversarial training on mel perturbations and post-window score
consistency checks.

Residual risk: medium in white-box settings, lower for casual physical-world attackers.

## Profile Theft

Attack vector: an attacker extracts stored content and speaker templates from the device.

Current mitigation in v1.0: profiles store normalized embeddings, not raw audio. The demo
stores plaintext local profiles for development only.

Planned mitigation: Android Keystore-backed encryption, biometric-gated profile export,
and automatic profile invalidation on device compromise signals.

Residual risk: medium until encrypted profile storage is implemented.

## Multi-User Confusion

Attack vector: one enrolled user's phrase or voice accidentally triggers another user's
profile on a shared device.

Current mitigation in v1.0: per-user templates and thresholds; multi-user inference emits
at most one highest-scoring wake event per frame.

Planned mitigation: cross-profile calibration, conflict warnings during enrollment, and
per-user false-accept reporting.

Residual risk: low to medium, depending on how similar enrolled users and phrases are.
