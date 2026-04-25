"""On-device quality metric collection (privacy-preserving).

No raw audio leaves the device. Only derived, anonymised statistics.
Uploaded opt-in only, with differential privacy noise applied before upload.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SoloSpeakLocalMetrics:
    """Local telemetry collected over a 24-hour window."""

    wake_events_count: int = 0
    wake_events_followed_by_command: int = 0  # TP proxy
    wake_events_cancelled: int = 0            # user said "never mind" → likely FP
    enrollment_retries: int = 0
    average_inference_latency_ms: float = 0.0
    model_version: str = ""
    backbone_variant: str = ""
    device_tier: str = ""                     # "flagship", "midrange", "wearable"


def record_wake_event(metrics: SoloSpeakLocalMetrics, followed_by_command: bool) -> None:
    metrics.wake_events_count += 1
    if followed_by_command:
        metrics.wake_events_followed_by_command += 1


def record_cancellation(metrics: SoloSpeakLocalMetrics) -> None:
    metrics.wake_events_cancelled += 1
