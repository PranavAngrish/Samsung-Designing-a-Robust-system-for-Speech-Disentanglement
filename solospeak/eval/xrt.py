"""Real-time factor benchmark helpers."""

from __future__ import annotations

import platform
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from solospeak.utils.types import XRTResult


def measure_xrt(
    onnx_path: Path,
    num_clips: int = 100,
    clip_duration_s: float = 1.6,
    threads: int = 1,
) -> XRTResult:
    """Measure ONNX Runtime xRT p50/p95/p99 over fixed 1.6 s windows."""
    if not onnx_path.exists():
        return {
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "platform": f"{platform.machine()}:{platform.system()}:onnx_missing",
        }
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = threads
    opts.intra_op_num_threads = threads
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    inputs = {inp.name: inp for inp in session.get_inputs()}
    feed: dict[str, NDArray[np.float32]] = {}
    for name, inp in inputs.items():
        shape = [1 if not isinstance(dim, int) or dim <= 0 else dim for dim in inp.shape]
        if name == "mel":
            shape = [1, 1, 80, 160]
        elif name in {"content_template", "speaker_template"}:
            shape = [1, 128]
        feed[name] = np.zeros(shape, dtype=np.float32)
    timings: list[float] = []
    for _ in range(5):
        session.run(None, feed)
    for _ in range(num_clips):
        start = time.perf_counter()
        session.run(None, feed)
        timings.append((time.perf_counter() - start) / clip_duration_s)
    values = np.asarray(timings, dtype=np.float64)
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "platform": f"{platform.machine()}:{platform.system()}:onnxruntime",
    }


def benchmark_xrt(
    onnx_path: Path,
    audio_duration_s: float = 1.6,
    n_runs: int = 100,
    n_warmup: int = 10,
) -> dict[str, float]:
    """Backward-compatible wrapper returning numeric xRT stats."""
    del n_warmup
    result = measure_xrt(onnx_path, num_clips=n_runs, clip_duration_s=audio_duration_s)
    return {
        "p50": float(result["p50"]),
        "p95": float(result["p95"]),
        "p99": float(result["p99"]),
        "mean": float(result["p50"]),
    }
