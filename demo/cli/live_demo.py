"""Terminal-based live wake-word demo.

Usage:
    python demo/cli/live_demo.py --profile profiles/user.json --model artifacts/solospeak_int8.onnx

Requires: pyaudio (pip install pyaudio)
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="SoloSpeak live demo")
    parser.add_argument("--profile", type=Path, required=True, help="Path to UserProfile JSON")
    parser.add_argument("--model", type=Path, default=Path("artifacts/solospeak_int8.onnx"))
    parser.add_argument("--enroll", action="store_true", help="Run enrollment before detection")
    args = parser.parse_args()

    raise NotImplementedError("Implement in Phase 7")


if __name__ == "__main__":
    main()
