"""Silero-VAD ONNX wrapper for streaming inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray


class SileroVAD:
    """Thin wrapper around the pinned Silero VAD v4 ONNX model."""

    FRAME_SAMPLES: ClassVar[int] = 512
    CONTEXT_SAMPLES: ClassVar[int] = 64
    SAMPLE_RATE: ClassVar[int] = 16000

    def __init__(self, onnx_path: Path, threshold: float = 0.5) -> None:
        self.threshold = threshold

        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session: Any = ort.InferenceSession(str(onnx_path), sess_options=opts)

        input_names = {inp.name for inp in self._session.get_inputs()}
        expected = {"input", "state", "sr"}
        if not expected.issubset(input_names):
            raise ValueError(
                f"Unexpected Silero ONNX inputs {sorted(input_names)}; "
                f"expected {sorted(expected)}."
            )

        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT_SAMPLES), dtype=np.float32)

    def is_speech(self, frame: NDArray[np.float32]) -> bool:
        """Return True if the 512-sample frame contains speech."""
        if frame.shape != (self.FRAME_SAMPLES,):
            raise ValueError(f"VAD expects exactly {self.FRAME_SAMPLES} samples, got {frame.shape}")

        x = frame[None, :].astype(np.float32)
        x = np.concatenate([self._context, x], axis=1)
        sr = np.array(self.SAMPLE_RATE, dtype=np.int64)
        outputs = self._session.run(None, {"input": x, "state": self._state, "sr": sr})
        speech_prob = np.asarray(outputs[0]).item()
        self._state = np.asarray(outputs[1], dtype=np.float32)
        self._context = x[:, -self.CONTEXT_SAMPLES :]
        return bool(speech_prob >= self.threshold)

    def reset_state(self) -> None:
        """Reset hidden state between independent audio streams."""
        self._state[:] = 0.0
        self._context[:] = 0.0

    def reset_states(self) -> None:
        """Backward-compatible alias for older call sites."""
        self.reset_state()
