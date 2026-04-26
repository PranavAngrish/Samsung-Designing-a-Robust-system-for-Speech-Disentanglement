"""Terminal live demo for SoloSpeak enrollment and listening."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from solospeak.data.features_deploy import LogMelExtractorDeploy
from solospeak.enrollment.service import EnrollmentService
from solospeak.enrollment.templates import load_profile, save_profile
from solospeak.inference.streaming import StreamingInference
from solospeak.observability.metrics import (
    SoloSpeakLocalMetrics,
    record_enrollment_attempt,
    record_wake_event,
    update_latency,
)
from solospeak.utils.audio import load_audio, pad_or_crop_to_window
from solospeak.utils.config import AudioConfig
from solospeak.utils.types import FloatArray


def _slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_") or "keyword"


def _syllable_count(keyword: str) -> int:
    words = keyword.lower().split()
    count = 0
    for word in words:
        groups = 0
        in_group = False
        for ch in word:
            is_vowel = ch in "aeiouy"
            if is_vowel and not in_group:
                groups += 1
            in_group = is_vowel
        count += max(1, groups)
    return count


def _validate_keyword(keyword: str) -> None:
    syllables = _syllable_count(keyword)
    if syllables < 2:
        raise SystemExit("Please choose a 2-syllable or longer keyword for reliability.")
    if syllables > 6:
        raise SystemExit("Please choose a shorter keyword (<= 6 syllables) for low-latency wake.")


def _synthetic_recordings(keyword: str, count: int = 3) -> list[FloatArray]:
    seed = int(hashlib.sha256(keyword.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    sr = 16000
    t = np.arange(int(1.6 * sr), dtype=np.float32) / sr
    recordings: list[FloatArray] = []
    for idx in range(count):
        f0 = 220.0 + 25.0 * idx
        tone = 0.08 * np.sin(2.0 * np.pi * f0 * t)
        noise = 0.005 * rng.standard_normal(t.shape)
        recordings.append(np.asarray(tone + noise, dtype=np.float32))
    return recordings


def _record_mic(seconds: float) -> FloatArray:
    import sounddevice as sd

    sr = 16000
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return np.asarray(audio[:, 0], dtype=np.float32)


class OnnxEnrollmentEncoder:
    """Embed enrollment waveforms through the deployed ONNX graph."""

    model_version = "solospeak-v1.0.0"

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.audio_config = AudioConfig()
        self.extractor = LogMelExtractorDeploy(self.audio_config)
        self.session: Any | None = None
        if model_path.exists():
            import onnxruntime as ort

            self.session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )

    def _fallback_embedding(self, wav: FloatArray) -> tuple[FloatArray, FloatArray]:
        digest = hashlib.sha256(wav.tobytes()).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        content = rng.standard_normal(128).astype(np.float32)
        speaker = rng.standard_normal(128).astype(np.float32)
        content /= max(float(np.linalg.norm(content)), 1e-8)
        speaker /= max(float(np.linalg.norm(speaker)), 1e-8)
        return content, speaker

    def embed(self, wav: FloatArray) -> tuple[FloatArray, FloatArray]:
        if self.session is None:
            return self._fallback_embedding(wav)
        mel = self.extractor(pad_or_crop_to_window(wav, self.audio_config.window_samples))
        outputs = self.session.run(
            None,
            {
                "mel": mel[None, :, :, :].astype(np.float32),
                "content_template": np.zeros((1, 128), dtype=np.float32),
                "speaker_template": np.zeros((1, 128), dtype=np.float32),
            },
        )
        return (
            np.asarray(outputs[0], dtype=np.float32).reshape(-1),
            np.asarray(outputs[1], dtype=np.float32).reshape(-1),
        )


def _resolve_model(args: argparse.Namespace) -> Path:
    if args.model_slot == "previous":
        previous = Path("artifacts/solospeak_int8_previous.onnx")
        if previous.exists():
            return previous
        current = Path(args.model)
        if current.exists():
            print("Previous model slot missing; using current slot for this demo run.")
            return current
        return previous
    return Path(args.model)


def _load_recordings(args: argparse.Namespace, keyword: str) -> list[FloatArray]:
    if args.input_files:
        return [
            pad_or_crop_to_window(load_audio(path), AudioConfig().window_samples)
            for path in args.input_files
        ]
    if args.mic:
        return [_record_mic(args.record_seconds) for _ in range(args.num_recordings)]
    return _synthetic_recordings(keyword, args.num_recordings)


def _run_listen(profile_path: Path, model_path: Path, seconds: float, mic: bool) -> None:
    profile = load_profile(profile_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    metrics = SoloSpeakLocalMetrics(model_version=profile.model_version)
    latencies: list[float] = []
    detector = StreamingInference(model_path)
    detector.enroll_user(profile)
    hop = int(0.16 * 16000)

    if not mic:
        start = time.perf_counter()
        event = detector.step(np.zeros(hop, dtype=np.float32))
        latencies.append((time.perf_counter() - start) * 1000.0)
        update_latency(metrics, latencies)
        if event:
            record_wake_event(metrics, event[0].user_id)
        print("Listen smoke check complete." if not event else f"WakeEvent: {event[0]}")
        print(json.dumps(metrics.to_dict(), sort_keys=True))
        return

    import sounddevice as sd

    print("Listening. Press Ctrl+C to stop.")
    start = time.time()
    last_metrics_print = start
    try:
        while time.time() - start < seconds:
            audio = sd.rec(hop, samplerate=16000, channels=1, dtype="float32")
            sd.wait()
            infer_start = time.perf_counter()
            events = detector.step(np.asarray(audio[:, 0], dtype=np.float32))
            latencies.append((time.perf_counter() - infer_start) * 1000.0)
            update_latency(metrics, latencies[-200:])
            for event in events:
                record_wake_event(metrics, event.user_id)
                print(
                    f"WakeEvent user={event.user_id} fusion={event.fusion_score:.3f} "
                    f"content={event.content_score:.3f} speaker={event.speaker_score:.3f}"
                )
            now = time.time()
            if now - last_metrics_print >= 60.0:
                print(json.dumps(metrics.to_dict(), sort_keys=True))
                last_metrics_print = now
    except KeyboardInterrupt:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="SoloSpeak live demo")
    parser.add_argument("--profile", type=Path, help="Path to UserProfile JSON")
    parser.add_argument("--profile-dir", type=Path, default=Path("profiles"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/solospeak_int8.onnx"))
    parser.add_argument("--model-slot", choices=["current", "previous"], default="current")
    parser.add_argument("--user", default="")
    parser.add_argument("--keyword", default="")
    parser.add_argument("--enroll", action="store_true", help="Create an enrollment profile")
    parser.add_argument("--listen", action="store_true", help="Run streaming detection")
    parser.add_argument("--eval", action="store_true", help="Run a boot-time demo check")
    parser.add_argument("--mic", action="store_true", help="Use microphone input")
    parser.add_argument("--record-seconds", type=float, default=1.6)
    parser.add_argument("--num-recordings", type=int, default=3)
    parser.add_argument("--input-files", type=Path, nargs="*")
    parser.add_argument("--listen-seconds", type=float, default=30.0)
    args = parser.parse_args()

    model_path = _resolve_model(args)
    profile_path = args.profile

    if args.enroll:
        if not args.user or not args.keyword:
            raise SystemExit("--enroll requires --user and --keyword.")
        _validate_keyword(args.keyword)
        metrics = SoloSpeakLocalMetrics()
        encoder = OnnxEnrollmentEncoder(model_path)
        recordings = _load_recordings(args, args.keyword)
        try:
            profile = EnrollmentService(encoder).enroll(args.user, args.keyword, recordings)
            record_enrollment_attempt(metrics, succeeded=True)
        except Exception:
            record_enrollment_attempt(metrics, succeeded=False)
            raise
        profile_path = args.profile_dir / f"{_slug(args.user)}_{_slug(args.keyword)}.json"
        save_profile(profile, profile_path)
        print(f"Saved profile: {profile_path}")
        print(
            "SoloSpeak has learned how YOU say this phrase. "
            "It will not respond to other people saying it."
        )
        print(json.dumps(metrics.to_dict(), sort_keys=True))

    if args.listen:
        if profile_path is None:
            raise SystemExit("--listen requires --profile or a preceding --enroll.")
        _run_listen(profile_path, model_path, args.listen_seconds, args.mic)
    elif args.eval:
        print(f"Demo boot check: model={model_path} profile={profile_path}")


if __name__ == "__main__":
    main()
