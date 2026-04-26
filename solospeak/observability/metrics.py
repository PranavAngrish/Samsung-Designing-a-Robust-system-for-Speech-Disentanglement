"""On-device metric collection. No raw audio leaves the device by default."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median


MetricValue = int | float | str | dict[str, int]


@dataclass
class SoloSpeakLocalMetrics:
    """Local privacy-preserving counters for demo and device health."""

    wake_events_total: int = 0
    wake_events_per_user: dict[str, int] = field(default_factory=dict)
    inference_latency_ms_p50: float = 0.0
    inference_latency_ms_p99: float = 0.0
    fa_events_estimated_per_hr: float = 0.0
    enrollment_attempts: int = 0
    enrollment_failures: int = 0
    model_version: str = "solospeak-v1.0.0"

    def to_dict(self) -> dict[str, MetricValue]:
        return {
            "wake_events_total": self.wake_events_total,
            "wake_events_per_user": dict(self.wake_events_per_user),
            "inference_latency_ms_p50": self.inference_latency_ms_p50,
            "inference_latency_ms_p99": self.inference_latency_ms_p99,
            "fa_events_estimated_per_hr": self.fa_events_estimated_per_hr,
            "enrollment_attempts": self.enrollment_attempts,
            "enrollment_failures": self.enrollment_failures,
            "model_version": self.model_version,
        }


def record_wake_event(metrics: SoloSpeakLocalMetrics, user_id: str) -> None:
    metrics.wake_events_total += 1
    metrics.wake_events_per_user[user_id] = metrics.wake_events_per_user.get(user_id, 0) + 1


def record_enrollment_attempt(metrics: SoloSpeakLocalMetrics, *, succeeded: bool) -> None:
    metrics.enrollment_attempts += 1
    if not succeeded:
        metrics.enrollment_failures += 1


def update_latency(metrics: SoloSpeakLocalMetrics, latency_ms: list[float]) -> None:
    if not latency_ms:
        return
    ordered = sorted(latency_ms)
    p99_index = min(len(ordered) - 1, int(round(0.99 * (len(ordered) - 1))))
    metrics.inference_latency_ms_p50 = float(median(ordered))
    metrics.inference_latency_ms_p99 = float(ordered[p99_index])

