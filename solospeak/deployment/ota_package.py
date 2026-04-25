"""OTA package assembly: zip + manifest + SHA-256 checksums."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


def build_ota_package(
    int8_onnx: Path,
    vad_onnx: Path,
    output_dir: Path,
    version: str = "1.0.0",
) -> Path:
    """Assemble the OTA zip package.

    Output structure:
        solospeak_ota_v{version}.zip
        ├── MANIFEST.json
        ├── solospeak_int8.onnx
        ├── silero_vad_v4.onnx
        └── (signature.p7s — added by Samsung infra, not here)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"solospeak_ota_v{version}.zip"

    manifest = {
        "version": version,
        "files": {
            "solospeak_int8.onnx": _sha256(int8_onnx),
            "silero_vad_v4.onnx": _sha256(vad_onnx),
        },
        "min_android_api": 31,
    }

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
        zf.write(int8_onnx, "solospeak_int8.onnx")
        zf.write(vad_onnx, "silero_vad_v4.onnx")

    return zip_path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
