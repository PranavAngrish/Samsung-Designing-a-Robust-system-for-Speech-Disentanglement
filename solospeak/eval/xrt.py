"""Real-time factor (xRT) benchmark.

Definition: xRT = model_inference_time / audio_duration
Target: < 0.2 (Samsung spec), < 0.08 (our goal with INT8 on ARM)

Measurement platform priority:
    1. M-series MacBook CPU (ARM) — primary
    2. Raspberry Pi 4 (Cortex-A72) — secondary
    3. Android device via Termux (optional)
"""

from __future__ import annotations

from pathlib import Path


def benchmark_xrt(
    onnx_path: Path,
    audio_duration_s: float = 1.0,
    n_runs: int = 100,
    n_warmup: int = 10,
) -> dict[str, float]:
    """Measure xRT statistics on CPU (single thread).

    Returns: {p50: float, p95: float, p99: float, mean: float}
    """
    raise NotImplementedError("Implement in Phase 4")
