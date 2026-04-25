# SoloSpeak — Implementation Reference v2.4

**Purpose.** This document is the single, authoritative source of truth for building SoloSpeak end-to-end. It encodes every architectural decision as a concrete, ordered build sequence: every file path, every function signature, every test, every acceptance criterion.

**How to use this document.**
- Read top-to-bottom on first pass to understand the full build sequence.
- Each phase has GO/NO-GO gates with two thresholds: a `MIN` gate that must pass to advance, and a `TARGET` aspiration that the final report aims for. Do not block advancement on the `TARGET` value.
- Every file listed has its purpose, dependencies, and contract documented.
- Copy-paste-ready templates are marked `[TEMPLATE]`.
- Decision points are marked `[DECISION]` with the chosen option and reasoning.
- Where v1 referenced an external "architecture document," v2 inlines the relevant content. The reference is now self-contained.

**Migration rule for existing old-code repos.**
This document supersedes all earlier code and docs. If an existing repository contains
older choices that conflict with this reference (for example `matchboxnet`, 1.5 s audio
windows, hardcoded backbone channel maps, variable-length inference, or TARGET-only
GO/NO-GO gates), migrate the code to this document rather than preserving backward
compatibility. Backward-compatible aliases are allowed only when explicitly listed here
(for example `BCResNet = SoloSpeakResNet`).

**Migration preflight for old-code repos.** Before implementing new behavior, remove or
rewrite stale files that directly contradict this reference:

- Delete `configs/backbone/matchboxnet.yaml` and
  `solospeak/models/backbones/matchboxnet.py` unless they are moved to a clearly marked
  research-only branch outside the shipping package.
- Rewrite `solospeak/utils/config.py` to the schema in 0.2.3: 1.6 s windows,
  `Field(default_factory=...)` nested defaults, injected class counts, and no
  `matchboxnet` variant.
- Rewrite `solospeak/training/stages/base.py` to the two-threshold MIN/TARGET API in
  3.1.1 and Appendix B. Old single-threshold `go_no_go_*` fields are not compatible.
- Rewrite `configs/eval/full_kpi_suite.yaml` so aspirational values such as
  `ta_clean_min: 0.99`, `fa_per_hour_max: 1.0`, and `xrt_int8_max: 0.08` are not hard
  gates. The hard gates are the `ValidationGates` values in Appendix C.
- Run `rg "matchboxnet|window_duration_s: 1.5|ta_clean_min: 0.99|xrt_int8_max: 0.08|go_no_go_threshold"`
  and resolve every hit before starting model/training work.

For existing old-code repositories, this migration preflight is a Phase-0 acceptance
gate. Do not begin feature implementation until every hit has either been removed or
rewritten to the v2.4 contract. For a greenfield repo the command should simply return no
matches.

**Document conventions.**
- `🔴 BLOCKING` — must work before moving forward.
- `🟡 STRETCH` — nice to have, can be deferred to post-hackathon.
- `🟢 VALIDATED` — known-working pattern, just implement it.
- `📋 CONTRACT` — function signature or API that other modules depend on.
- `🧪 TEST` — test that must pass before the step is considered done.
- `MIN: …` / `TARGET: …` — gate threshold (must-pass) vs aspiration (nice-to-have).

---

## Table of Contents

- [Phase 0: Foundation](#phase-0-foundation) — Days 0–3
- [Phase 1: Data Infrastructure](#phase-1-data-infrastructure) — Days 4–8
- [Phase 2: Core Model](#phase-2-core-model) — Days 9–14
- [Phase 3: Training Pipeline](#phase-3-training-pipeline) — Days 15–32
- [Phase 4: Evaluation Infrastructure](#phase-4-evaluation-infrastructure) — Days 33–37
- [Phase 5: Deployment Pipeline](#phase-5-deployment-pipeline) — Days 38–41
- [Phase 6: Production Hardening](#phase-6-production-hardening) — Days 42–44
- [Phase 7: Demo & Submission](#phase-7-demo--submission) — Days 42–46
- [Appendix A: Complete File Inventory](#appendix-a-complete-file-inventory)
- [Appendix B: Function Contracts Reference](#appendix-b-function-contracts-reference)
- [Appendix C: Configuration Schema](#appendix-c-configuration-schema)
- [Appendix D: Testing Matrix](#appendix-d-testing-matrix)
- [Appendix E: Failure Mode Playbook](#appendix-e-failure-mode-playbook)
- [Appendix F: Changelog from v1](#appendix-f-changelog-from-v1)

---

# Phase 0: Foundation

**Duration:** Days 0–3 (pre-kickoff prep).
**Goal:** Repo exists, environment reproducible, empty skeleton passes CI.
**Exit criterion:** `make test` returns exit code 0 on a fresh clone; CI green on `main`.

## 0.1 Repository Bootstrap

### 0.1.1 Create the repository

```bash
# Public repo from day 1 — signals commitment to judges.
gh repo create solospeak --public \
  --description "Speaker-personalized custom wake-word detection for Samsung ennovateX 2026"
git clone https://github.com/<user>/solospeak.git
cd solospeak
```

### 0.1.2 Directory scaffold

Create this exact structure on day 0. Empty directories get a `.gitkeep` file.

```
solospeak/
├── .github/
│   └── workflows/
│       ├── ci.yml                    # pytest + lint + type check on PR
│       └── eval-regression.yml       # nightly: run eval on last checkpoint
├── configs/
│   ├── backbone/
│   │   ├── bcresnet1.yaml
│   │   ├── bcresnet5.yaml
│   │   ├── bcresnet8.yaml
│   │   ├── bcresnet10.yaml
│   │   └── bcresnet16.yaml
│   ├── training/
│   │   ├── stage1_backbone_pretrain.yaml
│   │   ├── stage2_dual_head.yaml
│   │   ├── stage3_disentangle.yaml
│   │   ├── stage4_robustness.yaml
│   │   ├── stage5_fusion.yaml
│   │   └── stage6_qat.yaml
│   ├── eval/
│   │   ├── full_kpi_suite.yaml
│   │   └── ablations.yaml
│   └── defaults.yaml                 # base config all others inherit from
├── data/
│   ├── raw/                          # downloaded datasets (gitignored)
│   ├── processed/                    # prepared splits (gitignored)
│   ├── manifests/                    # CSV manifests (COMMITTED to git)
│   ├── licenses/                     # dataset license proofs (COMMITTED)
│   └── README.md                     # how to download + prepare data
├── solospeak/                        # main Python package
│   ├── __init__.py                   # exports __version__
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── augmentation.py
│   │   ├── hard_negatives.py
│   │   ├── splits.py
│   │   ├── features.py               # training-time log-mel (PyTorch)
│   │   ├── features_deploy.py        # deployment log-mel (NumPy, identical algo)
│   │   └── samplers.py               # class-aware batch sampler for SupCon
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbones/
│   │   │   ├── __init__.py
│   │   │   └── bcresnet.py           # SoloSpeakResNet; see Phase 2.1
│   │   ├── heads.py
│   │   ├── fusion.py
│   │   ├── solospeak.py              # full assembled model
│   │   └── vad.py                    # Silero-VAD wrapper
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── supcon.py
│   │   ├── orthogonality.py
│   │   ├── adversarial.py
│   │   └── combined.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── training_wrapper.py          # training-only aux/adversarial heads
│   │   ├── stages/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── stage1_backbone.py
│   │   │   ├── stage2_dual_head.py
│   │   │   ├── stage3_disentangle.py
│   │   │   ├── stage4_robustness.py
│   │   │   ├── stage5_fusion.py
│   │   │   └── stage6_qat.py
│   │   ├── trainer.py
│   │   ├── schedulers.py
│   │   └── callbacks.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── kpi_suite.py
│   │   ├── subgroup.py
│   │   ├── ablations.py
│   │   ├── probes.py
│   │   └── xrt.py
│   ├── enrollment/
│   │   ├── __init__.py
│   │   ├── service.py
│   │   ├── tts_augmentation.py
│   │   ├── calibration.py
│   │   └── templates.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── streaming.py
│   │   ├── hysteresis.py
│   │   └── multi_user.py
│   ├── deployment/
│   │   ├── __init__.py
│   │   ├── export_onnx.py
│   │   ├── quantize.py
│   │   ├── validate_artifact.py
│   │   └── ota_package.py
│   ├── security/
│   │   ├── __init__.py
│   │   ├── replay_eval.py
│   │   ├── adversarial_eval.py
│   │   └── threat_model.py
│   ├── observability/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── dp_noise.py
│   └── utils/
│       ├── __init__.py
│       ├── types.py
│       ├── config.py
│       ├── seeding.py
│       ├── audio.py
│       └── logging.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── fixtures/
│   │   └── audio/                    # tiny audio test files (committed)
│   ├── unit/
│   │   ├── test_backbones.py
│   │   ├── test_heads.py
│   │   ├── test_fusion.py
│   │   ├── test_losses.py
│   │   ├── test_features.py
│   │   ├── test_augmentation.py
│   │   ├── test_splits.py
│   │   ├── test_solospeak.py
│   │   ├── test_enrollment.py
│   │   ├── test_inference.py
│   │   └── test_deployment.py
│   └── integration/
│       ├── test_data_pipeline.py
│       ├── test_training_smoke.py
│       ├── test_eval_smoke.py
│       └── test_onnx_roundtrip.py
├── scripts/
│   ├── __init__.py                       # required: makes scripts/ a Python package for -m and entry points
│   ├── download_datasets.py
│   ├── prepare_libriphrase.py
│   ├── prepare_manifests.py
│   ├── pack_audio_lmdb.py
│   ├── precompute_speaker_embeddings.py
│   ├── run_stage.py
│   ├── run_eval.py
│   ├── run_ablation.py
│   └── export_and_validate.py
├── demo/
│   └── cli/
│       └── live_demo.py
├── docs/
│   ├── architecture.md
│   ├── reproducibility.md
│   ├── threat_model.md
│   ├── fairness_report.md
│   ├── ops_runbook.md
│   ├── deployment_guide.md
│   └── api_reference.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── Makefile
├── Dockerfile
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── README.md
└── CITATION.cff
```

### 0.1.3 Core configuration files

`🔴 BLOCKING` — everything depends on these.

#### `pyproject.toml` [TEMPLATE]

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "solospeak"
version = "0.1.0"
description = "Speaker-personalized custom wake-word detection"
readme = "README.md"
license = {file = "LICENSE"}
requires-python = ">=3.10,<3.12"
authors = [{name = "Pranav Angrish", email = "pangrish_be22@thapar.edu"}]

dependencies = [
    "torch==2.3.1",
    "torchaudio==2.3.1",    # must match torch minor version; SpeechBrain imports torchaudio
    "numpy==1.26.4",
    "scipy==1.13.1",
    "librosa==0.10.2",
    "soundfile==0.12.1",
    "lmdb==1.5.1",
    "onnx==1.16.0",
    "onnxruntime==1.18.0",
    "onnxsim==0.4.36",
    "pyyaml==6.0.1",
    "pydantic==2.7.1",
    "tqdm==4.66.4",
    "pandas==2.2.2",
    "scikit-learn==1.5.0",
    "wandb==0.17.0",
    "g2p-en==2.1.0",
    "speechbrain==1.0.3",   # for ECAPA-TDNN speaker embeddings
]

[project.optional-dependencies]
dev = [
    "pytest==8.2.1",
    "pytest-cov==5.0.0",
    "pytest-xdist==3.6.1",
    "ruff==0.4.4",
    "mypy==1.10.0",
    "pre-commit==3.7.1",
    "pdoc==14.5.1",
    "kaggle==1.6.14",
]
tts = [
    "transformers==4.41.0",
    "parler-tts==0.2.3",
]
demo = [
    "sounddevice==0.4.6",    # CLI demo microphone input
]

[project.scripts]
solospeak-train = "scripts.run_stage:main"
solospeak-eval = "scripts.run_eval:main"
solospeak-export = "scripts.export_and_validate:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["solospeak*", "scripts*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests requiring real data or long runtime",
    "gpu: marks tests requiring CUDA",
    "onnx: marks tests for ONNX export/runtime",
    "data: marks tests requiring downloaded or LMDB-packed datasets",
    "tts: marks optional tests requiring the .[tts] extra",
    "demo: marks optional tests requiring microphone/demo dependencies",
]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
strict = true
ignore_missing_imports = true
```

**Python version rule.** Use Python 3.10 or 3.11 for all reproducible runs. The pinned
dependency set is not guaranteed on Python 3.12+, so do not widen `requires-python` until
`make install-dev && make test` and the ONNX/export dependencies have been validated on
the newer interpreter.

**Optional dependency rule.** The `tts` and `demo` extras are intentionally not installed
in CI by default. Modules under `solospeak/enrollment/tts_augmentation.py` MUST import
Parler-TTS and Transformers lazily inside the function that uses them. The CLI demo MUST
import `sounddevice` lazily inside `main()`. Tests requiring optional extras are marked
`@pytest.mark.tts` or `@pytest.mark.demo` and run only in an optional workflow or by
`make test-optional`.

#### `Makefile` [TEMPLATE]

```makefile
.PHONY: install install-dev install-optional test test-unit test-integration test-slow test-all test-optional lint typecheck docs-check format download-data download-data-smoke prepare-manifests prepare-manifests-smoke pack-lmdb precompute-speaker-embeds eval ablation export clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

install-optional:
	pip install -e ".[tts,demo]"

# Phase-0 safe: no downloaded datasets, checkpoints, GPUs, or optional deps.
test: lint typecheck test-unit

test-unit:
	pytest tests/unit -v -n auto -m "not slow and not data and not tts and not demo"

test-integration:
	pytest tests/integration -v -m "not slow and not data and not tts and not demo"

test-slow:
	pytest tests/integration -v -m "slow or data"

test-optional:
	# Requires: make install-optional
	pytest tests -v -m "tts or demo"

test-all: test-unit test-integration test-slow

lint:
	ruff check solospeak tests scripts

typecheck:
	mypy solospeak

docs-check:
	find docs -name "*.md" -print0 | xargs -0 -n 1 npx -y markdown-link-check

format:
	ruff format solospeak tests scripts

download-data:
	python -m scripts.download_datasets

download-data-smoke:
	python -m scripts.download_datasets --minimal

prepare-manifests:
	python -m scripts.prepare_manifests

prepare-manifests-smoke:
	python -m scripts.prepare_manifests --smoke

pack-lmdb:
	python -m scripts.pack_audio_lmdb

precompute-speaker-embeds:
	python -m scripts.precompute_speaker_embeddings

stage-%:
	python -m scripts.run_stage --stage $*

eval:
	python -m scripts.run_eval --checkpoint checkpoints/latest.pt

ablation:
	python -m scripts.run_ablation --config configs/eval/ablations.yaml

export:
	python -m scripts.export_and_validate --checkpoint checkpoints/latest.pt

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info
```

#### `requirements.txt` and `requirements-dev.txt` [NOTE]

Both files are listed in the scaffold for tooling compatibility (e.g. `pip install -r requirements.txt` without setuptools). They are **thin redirects to `pyproject.toml`** — do NOT duplicate pinned versions here, as that creates a maintenance conflict.

`requirements.txt`:
```
# Install the package with all runtime dependencies (pins come from pyproject.toml).
-e .
```

`requirements-dev.txt`:
```
# Install the package with all dev dependencies (pins come from pyproject.toml).
-e .[dev]
```

#### `Dockerfile` [TEMPLATE]

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for soundfile / scipy and the Makefile entry point
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY README.md LICENSE Makefile ./
COPY solospeak/ ./solospeak/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

RUN pip install --no-cache-dir -e ".[dev]"

CMD ["make", "test"]
```

#### `.pre-commit-config.yaml` [TEMPLATE]

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
        additional_dependencies: [pydantic==2.7.1, torch==2.3.1]
```

#### `.github/workflows/ci.yml` [TEMPLATE]

```yaml
name: CI
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-22.04
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install
        run: |
          pip install --upgrade pip
          pip install -e ".[dev]"
      - name: Lint
        run: ruff check solospeak tests scripts
      - name: Type check
        run: mypy solospeak
      - name: Unit tests
        run: pytest tests/unit -v -m "not slow and not data and not tts and not demo" --cov=solospeak --cov-report=xml
      - name: Prepare smoke data
        run: |
          python -m scripts.download_datasets --minimal
          python -m scripts.prepare_manifests --smoke
      - name: Integration smoke tests
        run: pytest tests/integration -v -m "not slow and not data and not tts and not demo"
```

### 0.1.4 README

Required headers (judges see this first):

```markdown
# SoloSpeak

Speaker-personalized custom wake-word detection. Samsung ennovateX AX Hackathon 2026.

[One-paragraph pitch.]

## Status
[Current phase, latest KPI numbers.]

## Architecture
[Link to docs/architecture.md; embed key diagram.]

## Quick Start
[git clone, make install-dev, make test]

## Training
[How to reproduce our results.]

## Evaluation
[How to run the KPI suite.]

## Deployment
[How to export ONNX + validate.]

## License
Apache-2.0

## Citation
[CITATION.cff reference.]
```

### 0.1.5 Acceptance criteria — Phase 0.1

- 🧪 Clone to empty machine, run `make install-dev && make test`. Exit code 0. This is a
  Phase-0-safe check: unit/import/fixture tests only, no real datasets or checkpoints.
- 🧪 Existing old-code repos pass the migration preflight from the front matter: no
  product-path hits for `matchboxnet`, 1.5 s windows, TARGET-only eval hard gates, or old
  single-threshold `go_no_go_*` stage APIs.
- 🧪 `make test-all` is allowed to require downloaded data, checkpoints, or long-running
  integration resources.
- 🧪 CI green on `main`.

## 0.2 Skeleton Code Contracts

Before any real implementation, define every module's public contract as a stub. Stubs MUST import cleanly; constructor bodies MAY raise `NotImplementedError` when actually called, but module-level import and class definition must succeed.

**Wording rule.** "Module imports without error" means `import solospeak.X` succeeds. It does NOT mean class instantiation succeeds. Tests that exercise functionality come later; tests that exercise the import graph come now.

### 0.2.1 Top-level `__init__.py`

`solospeak/__init__.py`:

```python
"""SoloSpeak — speaker-personalized custom wake-word detection."""
__version__ = "0.1.0"

__all__ = ["__version__"]
```

### 0.2.2 Core data types

`solospeak/utils/types.py` — single source of truth for shared types.

```python
"""Type definitions used across SoloSpeak modules."""
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import torch

# Audio tensor shapes
AudioWaveform: TypeAlias = torch.Tensor      # (B, T) or (T,), float32, [-1, 1]
MelSpectrogram: TypeAlias = torch.Tensor     # (B, 1, F=80, T'), float32
Embedding128:  TypeAlias = torch.Tensor      # (B, 128), float32, L2-normalized

# Typed aliases — required by mypy strict mode; avoids bare `dict` annotations.
DatasetValue: TypeAlias = torch.Tensor | np.ndarray | str | int | float | bool | None
DatasetItem: TypeAlias = dict[str, DatasetValue]      # one row emitted by __getitem__()
BatchValue: TypeAlias = torch.Tensor | list[str] | list[int] | list[float] | list[bool]
BatchDict: TypeAlias = dict[str, BatchValue]          # collated batch keyed by field name
StageBatch: TypeAlias = dict[str, BatchDict]          # e.g. {"content": ..., "speaker": ...}
LossDict: TypeAlias = dict[str, torch.Tensor]         # keyed by loss component name
MetricsDict: TypeAlias = dict[str, float]             # keyed by metric name
SubgroupDict: TypeAlias = dict[str, dict[str, float] | None] # subgroup_name -> metric -> value, or None when metadata is unavailable
AugRanges: TypeAlias = dict[str, tuple[int, int] | tuple[float, float]]  # snr_range, distance_range
ProbeBaseline: TypeAlias = dict[str, float]
ProbeResult: TypeAlias = dict[str, float | None]
XRTResult: TypeAlias = dict[str, float | str]
AblationResult: TypeAlias = dict[str, dict[str, float | int | str]]

BackboneVariant = Literal[
    "bcresnet1", "bcresnet5", "bcresnet8", "bcresnet10", "bcresnet16",
]

QuadrantLabel = Literal["Q1_accept", "Q2_imposter", "Q3_wrong_word", "Q4_background"]

# Single source of truth for the keyword OOV sentinel.
# Import this constant in datasets.py, combined.py, and any other module that
# passes keyword labels to CrossEntropyLoss — never re-define -100 as a literal.
WORD_IGNORE_INDEX: int = -100


@dataclass(frozen=True)
class UserProfile:
    """On-device user profile. Serialized to ~1.1 KB per user."""
    user_id: str
    keyword_text: str
    content_template: np.ndarray   # (128,) float32, L2-normalized
    speaker_template: np.ndarray   # (128,) float32, L2-normalized
    tau: float                     # decision threshold (per-user)
    model_version: str             # e.g. "solospeak-v1.0.0"


@dataclass
class WakeEvent:
    user_id: str
    keyword_text: str
    timestamp_ms: int
    fusion_score: float
    content_score: float
    speaker_score: float


@dataclass
class KPIResult:
    ta_clean: float
    ta_noisy: dict[int, float]              # SNR (dB) -> TA
    ta_noisy_macro: float
    distance_ta: dict[float, float]         # distance bucket (m) -> TA
    fa_per_hour_per_user: float
    fa_per_hour_device: dict[int, float]    # N enrolled users -> fa/hr; N ∈ {1, 4, 8}
    q2_rejection: float
    q3_rejection: float             # real + speaker-verified synth weighted average
    q3_rejection_real: float        # Q3 rejection rate on real (non-synthesized) utterances only
    q3_rejection_synth: float       # all synthesized Q3 stress rows, reported separately
    q4_rejection: float
    param_count: int
    xrt_fp32: float
    xrt_int8: float
    seed: int
    num_eval_samples: int
    per_demographic: SubgroupDict           # optional subgroup KPI summaries
```

### 0.2.3 Configuration system — pydantic-backed YAML

`🔴 BLOCKING` — all training and eval reads config from here.

`solospeak/utils/config.py`:

```python
"""Configuration schema. All training/eval/deploy flows read from here."""
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160             # 10 ms
    win_length: int = 400             # 25 ms
    n_mels: int = 80
    fmin: float = 20.0
    fmax: float = 7600.0
    # Window length used by the streaming ring buffer and the fixed input shape
    # for the deployed ONNX model.
    #
    # Exact contract:
    # - ring buffer audio = 1.6 s = 25,600 samples at 16 kHz
    # - torch.stft(..., center=True) naturally emits 161 frames for 25,600 samples
    # - feature extractors MUST center-crop or right-pad the mel time axis to
    #   window_frames=160 before returning. The ONNX graph always sees exactly
    #   (B, 1, 80, 160).
    window_duration_s: float = 1.6
    window_samples: int = 25600
    window_frames: int = 160
    # CONSTRAINT: win_length MUST equal n_fft at all times.
    # LogMelExtractorDeploy (features_deploy.py) enforces this at call-time and raises
    # ValueError if violated. The training extractor (LogMelExtractor) naturally supports
    # win_length != n_fft via torch.stft, but doing so breaks train/deploy parity.
    # Do not change either value independently.


class BackboneConfig(BaseModel):
    variant: Literal["bcresnet1", "bcresnet5", "bcresnet8",
                     "bcresnet10", "bcresnet16"] = "bcresnet8"


class HeadConfig(BaseModel):
    content_dim: int = 128
    speaker_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.1


class FusionConfig(BaseModel):
    # Six engineered features: s_c, s_s, s_c*s_s, |s_c - s_s|, s_c^2, s_s^2
    feature_names: list[str] = Field(default_factory=lambda: [
        "s_c", "s_s", "s_c_times_s_s",
        "abs_diff", "s_c_sq", "s_s_sq",
    ])
    hidden_dims: list[int] = Field(default_factory=lambda: [20, 10])
    tau_on: float = 0.65
    tau_off: float = 0.45


class LossWeights(BaseModel):
    supcon_content: float = 1.0
    supcon_speaker: float = 1.0
    orthogonality: float = 0.1
    adversarial: float = 0.1
    ce_aux: float = 0.5
    # Stage-3 ramp-up over this many optimizer steps
    adversarial_ramp_steps: int = 5000


class TrainingConfig(BaseModel):
    stage: Literal[1, 2, 3, 4, 5, 6]
    batch_size: int = 128
    num_epochs: int = 30
    optimizer: Literal["adamw", "sgd"] = "adamw"
    lr: float = 3e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_grad_norm: float = 5.0
    precision: Literal["fp32", "amp_bf16", "amp_fp16"] = "amp_fp16"
    # Default is "amp_fp16": Kaggle T4 GPUs (compute capability 7.5) do NOT support
    # bfloat16. Override to "amp_bf16" only on Ampere (A100) or newer hardware.
    seed: int = 42
    num_workers: int = 4
    resume_from: Path | None = None
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_every_steps: int = 500       # write local checkpoint every N optimizer steps
    upload_every_steps: int = 2000          # upload to Kaggle dataset every N optimizer steps
    # Number of unique classes in the manifests for this stage. Loaded by
    # run_stage.py from data/manifests/STATS.json and injected into the config
    # at runtime — do NOT hardcode. Stage 1 uses n_gsc_classes; Stages 2-4 use
    # n_aux_word_classes / n_aux_speaker_classes.
    n_gsc_classes: int | None = None
    n_aux_word_classes: int | None = None
    n_aux_speaker_classes: int | None = None


class DataConfig(BaseModel):
    root: Path = Path("data/processed")
    manifests_dir: Path = Path("data/manifests")
    snr_range_db: tuple[int, int] = (-5, 30)
    distance_range_m: tuple[float, float] = (0.5, 5.0)
    augmentation_prob: float = 0.8
    hard_neg_fraction: float = 0.3


class SoloSpeakConfig(BaseModel):
    """Top-level config. Every script takes one of these."""
    run_name: str
    audio:    AudioConfig    = Field(default_factory=AudioConfig)
    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    heads:    HeadConfig     = Field(default_factory=HeadConfig)
    fusion:   FusionConfig   = Field(default_factory=FusionConfig)
    losses:   LossWeights    = Field(default_factory=LossWeights)
    training: TrainingConfig
    data:     DataConfig     = Field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "SoloSpeakConfig":
        data = _load_yaml_with_base(Path(path))
        return cls.model_validate(data)

    def to_yaml(self, path: Path | str) -> None:
        with open(Path(path), "w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override onto base; override wins on scalar/list values."""
    out = dict(base)
    for key, value in override.items():
        if key == "__base__":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml_with_base(path: Path) -> dict[str, Any]:
    """Load YAML with optional '__base__: relative/path.yaml' inheritance."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    base_ref = data.get("__base__")
    if base_ref is None:
        return data
    base_path = (path.parent / base_ref).resolve()
    return _deep_merge(_load_yaml_with_base(base_path), data)
```

**Note.** All nested config defaults use `Field(default_factory=...)` — never bare `= AudioConfig()` — to avoid pydantic v2's mutable-default warnings and to keep configs reproducible across imports.

**YAML inheritance contract.** Any config may include `__base__: ../defaults.yaml` (or
another relative file). The loader recursively deep-merges dicts; child values replace
parent scalar/list values. This is the only supported inheritance mechanism. If a file
does not include `__base__`, it must be fully self-contained.

#### Required YAML templates

`configs/defaults.yaml`:

```yaml
run_name: solospeak-default
training:
  stage: 1
  checkpoint_every_steps: 500
  upload_every_steps: 2000
audio:
  sample_rate: 16000
  n_fft: 400
  hop_length: 160
  win_length: 400
  n_mels: 80
  fmin: 20.0
  fmax: 7600.0
  window_duration_s: 1.6
  window_samples: 25600
  window_frames: 160
backbone:
  variant: bcresnet8
heads:
  content_dim: 128
  speaker_dim: 128
  hidden_dim: 256
  dropout: 0.1
fusion:
  hidden_dims: [20, 10]
  tau_on: 0.65
  tau_off: 0.45
losses:
  supcon_content: 1.0
  supcon_speaker: 1.0
  orthogonality: 0.1
  adversarial: 0.1
  ce_aux: 0.5
  adversarial_ramp_steps: 5000
data:
  root: data/processed
  manifests_dir: data/manifests
  snr_range_db: [-5, 30]
  distance_range_m: [0.5, 5.0]
  augmentation_prob: 0.8
  hard_neg_fraction: 0.3
```

`configs/training/stage<N>_*.yaml` files inherit from `../defaults.yaml` and override
only `run_name`, `training.stage`, and stage-specific hyperparameters. Example:

```yaml
__base__: ../defaults.yaml
run_name: stage1_backbone_pretrain
training:
  stage: 1
  batch_size: 256
  num_epochs: 30
  lr: 0.003
```

Backbone variant configs inherit from `../defaults.yaml` and override
`backbone.variant`. Eval/deployment config files use the additional schemas in
Appendix C.

**Backbone support rule.** `SoloSpeakResNet` is the only supported backbone family for
v2.4. Do not keep `matchboxnet` in `BackboneConfig`, config files, tests, or production
code. If the old repository contains a MatchboxNet stub/config, delete it or move it to
an explicitly marked research branch; it is not part of the shipping product path.

### 0.2.4 Seeding — determinism is a hard requirement

`solospeak/utils/seeding.py`:

```python
"""Deterministic seeding. Call seed_everything() at the start of every script.

Note: Exact bit-level determinism across CUDA versions is NOT guaranteed.
Regression tolerance: ±0.2 percentage points on accuracy metrics, ±3e-3
absolute on raw float tensors.
"""
import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 0.2.4b Utility modules — `audio.py` and `logging.py`

These two utility files must be created at Level 1. Their public contracts are minimal on purpose — they exist to centralise shared helpers so no other module re-implements them.

`solospeak/utils/audio.py`:

```python
"""Audio I/O and waveform helpers."""
from pathlib import Path

import numpy as np
import soundfile as sf


def load_wav(path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load a wav file and return a mono float32 waveform normalised to [-1, 1].

    Resamples to `target_sr` using linear interpolation if the file sample rate
    differs. For production training use the pre-processed LMDB store instead.

    Returns:
        waveform: (T,) float32
    """
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)          # stereo → mono
    if sr != target_sr:
        # Simple linear resample — acceptable for offline preprocessing;
        # production inference always receives 16 kHz input.
        ratio = target_sr / sr
        n_out = int(len(audio) * ratio)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    return audio


def pad_or_crop_to_window(wav: np.ndarray, window_samples: int = 25600) -> np.ndarray:
    """Pad (right, reflect) or centre-crop a waveform to exactly `window_samples`."""
    if len(wav) == window_samples:
        return wav
    if len(wav) < window_samples:
        return np.pad(wav, (0, window_samples - len(wav)), mode="reflect")
    start = (len(wav) - window_samples) // 2
    return wav[start:start + window_samples]
```

`solospeak/utils/logging.py`:

```python
"""Project-wide logging helpers.

Uses Python's stdlib `logging` module. W&B is a separate concern — use
`solospeak/training/callbacks.py` for metric logging.

All modules that need a logger should call `get_logger(__name__)`.
"""
import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return (or create) a module-level logger with a stderr StreamHandler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
```

### 0.2.5 Acceptance criteria — Phase 0.2

- 🧪 `python -c "import solospeak; print(solospeak.__version__)"` prints `0.1.0`.
- 🧪 `python -c "from solospeak.utils.config import SoloSpeakConfig; SoloSpeakConfig.from_yaml('configs/defaults.yaml')"` succeeds.
- 🧪 `mypy solospeak` returns 0 errors.
- 🧪 Every module in the directory tree imports without error (constructor calls may still raise `NotImplementedError`).

---

# Phase 1: Data Infrastructure

**Duration:** Days 4–8.
**Goal:** Reproducible pipeline from raw downloads to training-ready manifests, with on-the-fly augmentation.
**Exit criterion:** `make download-data && make prepare-manifests && make pack-lmdb`
produces 7 manifests, documented statistics, and a Kaggle-ready LMDB audio store; a
`DataLoader` over `DualHeadDataset` yields augmented batches at `MIN ≥ 80 samples/sec`
(TARGET ≥ 200 samples/sec) on a 4-worker CPU process.

**Smoke path.** CI and fresh-clone development MUST use the minimal path:
`make download-data-smoke && make prepare-manifests-smoke`. It generates tiny manifests
from committed/generated audio fixtures and never downloads multi-GB corpora. The full
download path is a data-curation task, not a CI prerequisite.

## 1.1 Dataset Acquisition

### 1.1.1 License audit (pre-code check)

`🔴 BLOCKING` — do this before touching any dataset.

| Dataset | License | Commercial-OK | SoloSpeak Usage |
|---|---|---|---|
| Google Speech Commands v2 | CC-BY 4.0 | Yes | Stage 1 pretraining |
| LibriSpeech (basis for LibriPhrase) | CC-BY 4.0 | Yes | Primary content training |
| VoxCeleb 1 & 2 | Metadata: CC BY-SA 4.0; audio access/YouTube-derived terms require approval | Unclear; treat as research-only unless approval says otherwise | Preferred speaker head source |
| MUSAN | CC-BY 4.0 | Yes | Noise augmentation plus FA/noisy evaluation buckets |
| OpenSLR-28 RIRs | Apache 2.0 | Yes | RIR augmentation |
| Common Voice | CC0-1.0 data, with Mozilla Data Collective use restrictions | Yes for permitted uses; do not re-host/share; do not attempt real-world identity discovery | Lower-confidence fallback (see 1.1.4) |

**Action.** Request VoxCeleb access on day 0 (can take up to 1 week). Save the approval
email/terms in `data/licenses/` for Phase 2 submission proof. If the terms do not permit
the planned use, do not use VoxCeleb audio.

**License proof files.** Save one markdown note per corpus under `data/licenses/` with:
source URL, license/terms snapshot date, allowed use, attribution text, and any use
restrictions. This is a project compliance check, not legal advice.

Primary URLs to snapshot:
- GSC v2: `https://www.tensorflow.org/datasets/catalog/speech_commands`
- LibriSpeech: `https://www.openslr.org/12`
- VoxCeleb1/2: `https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html`,
  `https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html`
- MUSAN: `https://www.openslr.org/17`
- OpenSLR-28: `https://www.openslr.org/28`
- Common Voice English: Mozilla Data Collective page for the pinned release.

### 1.1.2 LibriPhrase generation — pinned recipe

LibriPhrase is NOT a separately-distributed dataset. It is a re-organization of LibriSpeech into (keyword phrase, speaker) pairs. To make this reproducible:

- **Source corpus:** LibriSpeech `train-clean-100`, `train-clean-360`, and
  `train-other-500`.
- **Generator:** `scripts/prepare_libriphrase.py` (project-internal; spec below).
- **Default alignment path:** Montreal Forced Aligner (MFA) 3.3.9. Do not depend on
  unversioned third-party alignment dumps.
- **Pinned MFA setup:** `conda install -c conda-forge montreal-forced-aligner=3.3.9`;
  `mfa model download acoustic english_us_arpa`; `mfa model download dictionary english_us_arpa`.
- **Procedure:**
  1. Convert LibriSpeech transcripts to MFA corpus layout with one `.wav`/`.txt` pair per utterance.
  2. Run MFA with the pinned `english_us_arpa` acoustic model and dictionary. Cache the
     resulting TextGrid files under `data/processed/libriphrase_alignments/`.
  3. Extract every contiguous 2–4 word span that occurs across at least 5 distinct speakers.
  4. Rank phrase types by `(distinct_speaker_count, total_utterance_count)` descending,
     break ties lexicographically, and keep the top 1,500 phrase types.
  5. Cap each (phrase, speaker) pair at 10 utterances (random subsample if more available).
  6. Discard phrases shorter than 0.4 s or longer than 1.4 s.
  7. Output: rows containing `(file_path, start_s, end_s, duration_s, phrase, speaker_id)`.
     If `file_path` points at the original LibriSpeech utterance, `start_s` / `end_s`
     are mandatory and the dataset loader must crop the phrase before padding to 1.6 s.
     If the phrase has already been sliced into a standalone wav/LMDB item, use
     `start_s=0.0` and `end_s=duration_s`.
- **Expected output:** ~40,000–60,000 manifest rows / utterance segments after the
  per-(phrase, speaker) cap above, spanning up to 1,500 phrase types and ~2,300
  LibriSpeech training speakers. `STATS.md` / `STATS.json` must report both
  `row_count` and `unique_phrase_speaker_pairs` so agents do not confuse capped
  utterance rows with unique `(phrase, speaker)` identities. If `train-other-500` is
  intentionally excluded for a clean-only experiment, lower the expected speaker counts
  accordingly; do not keep the ~2,300-speaker expectation with clean-only input.
- **Reproducibility:** the script is committed to the repo; the MFA version/model names
  and alignment cache checksum are pinned in `data/README.md`.

### 1.1.3 Download script

`scripts/download_datasets.py`:

```python
"""Idempotent dataset downloader.

Usage:
    python -m scripts.download_datasets --dataset all
    python -m scripts.download_datasets --dataset gsc_v2
    python -m scripts.download_datasets --minimal

Design:
- Every download verified by SHA-256.
- Resumable (uses curl --continue or aria2c if available).
- Writes to data/raw/<dataset_name>/.
- Skips if the SHA matches.
- --minimal writes tiny generated/fixture audio under data/raw/smoke/ and performs no
  network download. It exists for CI and local smoke tests only.
"""
```

Required dataset entries:

1. **GSC v2** — `http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz`, ~2.4 GB, extract to `data/raw/gsc_v2/`. 105,829 wav files, 35 command classes.
2. **LibriSpeech `train-clean-100`** — ~6.3 GB.
3. **LibriSpeech `train-clean-360`** — ~23 GB.
4. **LibriSpeech `train-other-500`** — ~30 GB. Required for the default
   LibriPhrase recipe's speaker-count targets; optional only for a clearly marked
   clean-only ablation.
5. **VoxCeleb 1 + 2** — see 1.1.4 for fallback.
6. **MUSAN** — `https://www.openslr.org/resources/17/musan.tar.gz`, ~11 GB. Download
   all categories (`music`, `noise`, `speech`) because different project stages need
   different subsets:
   - training noise augmentation uses `noise` by default;
   - FA evaluation draws continuous non-target audio from `speech`, `music`, and `noise`;
   - noisy-TA buckets (`babble`, `traffic`, `music`) are resolved from tagged lists
     generated by `scripts/prepare_manifests.py`. `music` maps to MUSAN `music`,
     `babble` is mixed from MUSAN `speech`, and `traffic` must come from either a tagged
     subset of MUSAN `noise` or an explicitly documented CC-compatible supplemental noise
     source recorded in `data/licenses/`. Do not silently alias a missing bucket.
7. **OpenSLR-28 RIRs** — `https://www.openslr.org/resources/28/rirs_noises.zip`, ~1.3 GB.

### 1.1.4 VoxCeleb fallback plan

`🔴 BLOCKING` — VoxCeleb access can fail. The project must not depend on it.

If VoxCeleb access is denied, unavailable, or delayed past day 7, switch to a
**Common Voice + LibriSpeech speaker** pipeline:

- Common Voice English pinned release (default: Mozilla Data Collective Scripted Speech
  24.0 unless a newer release is explicitly snapshotted in `data/licenses/`): estimate
  speaker counts from the release metadata after download.
- LibriSpeech (all `train-clean-*` and `train-other-500`): ~2,500 speakers.
- Combined: use all release speaker IDs with ≥5 usable clips plus ~2,500 LibriSpeech
  speakers — enough to prototype a speaker head, though the speaker classes are noisier
  and lower-confidence than VoxCeleb/LibriSpeech.

The training code paths must work with either dataset; this is a first-class supported
path, not an emergency hack. The dataset choice is recorded in the manifest's
`source_dataset` column, and no code may assume VoxCeleb-specific directory layouts or
metadata fields. For Common Voice, treat labels as anonymous dataset speaker IDs only.
Do not attempt real-world identity discovery, and do not re-host/re-share the dataset.
If the downloaded Common Voice terms do not permit the planned speaker-ID use, use the
LibriSpeech-only fallback and mark speaker-head quality as lower confidence in the
report.

### 1.1.5 Storage budget — Kaggle / Colab strategy

**Estimated raw + processed disk usage:**
- Full path (with VoxCeleb2): ~290 GB raw, ~40 GB processed.
- Fallback path (pinned Common Voice English + LibriSpeech): release-dependent, roughly
  120–170 GB raw for recent English releases, ~25–40 GB processed.

**Kaggle constraints:**
- Persistent storage: 20 GB/notebook.
- Working storage: 107 GB/session, wiped on disconnect.
- Datasets attachment: a Kaggle dataset can be up to 100 GB and is mounted read-only.

**Strategy.**
1. Pre-process the data on a workstation (or Colab Pro) once: produce the manifests + an LMDB-packed audio store (`data/processed/audio.lmdb`) containing only the (keyword, speaker, clip) triplets we actually use, ~25 GB.
2. Upload the LMDB and manifests as a Kaggle Dataset.
3. Training notebooks attach the dataset read-only and stream from LMDB. No re-download is ever needed in a Kaggle session.
4. Checkpoints write to Kaggle's `output/` (per-session) and are uploaded to a dedicated `solospeak-checkpoints` Kaggle dataset between sessions.
5. W&B for metrics; checkpoints persist via Kaggle datasets, not via session storage.

This is documented as **the only supported training environment** for the hackathon. Any other environment is a deviation requiring the user to handle storage themselves.

## 1.2 Manifest Generation

### 1.2.1 Manifest schema

Every manifest is a CSV with columns:

```csv
file_path,start_s,end_s,duration_s,speaker_id,keyword_text,split,source_dataset,quadrant_class,profile_id,enrolled_user_id,enrolled_keyword_text,profile_path,trial_source,q3_gate_eligible,synthesis_backend,speaker_verification_score
```

`file_path` is either a real path under `data/raw/` or an LMDB URI of the form
`lmdb://data/processed/audio.lmdb/<key>`.

`speaker_id` and `keyword_text` always describe the trial utterance in `file_path`.
For ordinary training rows, the enrollment/profile columns are empty. For
`test_kpi.csv` and Stage-5 quadrant rows, `profile_id`, `enrolled_user_id`, and
`enrolled_keyword_text` identify the enrolled profile the trial is evaluated against.
`profile_path` points to a `.npz` profile file containing
`content_template`, `speaker_template`, `tau`, `user_id`, and `keyword_text`; it may be
blank before templates have been generated. `trial_source` is one of
`real`, `augmented`, `tts`, or `background` and is required for the Q3 real/synth split.
`q3_gate_eligible` is a boolean string (`true` / `false`) used by Stage-5 and KPI
evaluation. For non-Q3 rows it is normally `true`; for Q3 rows it is `true` only for real
right-voice trials or synthetic trials that passed the verification policy in 3.6.1.
Dataset code MUST parse this column explicitly; never use `bool(value)` on a CSV string,
because `bool("false")` is `True` in Python.
`synthesis_backend` is blank for real/non-synthetic rows and otherwise records the
backend name (for example `parler_tts` or `voice_clone_x`). `speaker_verification_score`
is blank unless a synthetic right-voice claim was checked against the enrolled speaker
template; when present it stores the cosine similarity used for the pass/fail decision.

`start_s` and `end_s` define the segment inside `file_path`. For pre-sliced wav files or
LMDB records they must be `0.0` and `duration_s`. For raw LibriSpeech/Common Voice files
they are mandatory crop offsets; dataset classes must crop this segment before
pad/crop-to-window.

`quadrant_class` is empty for normal training manifests and populated for `test_kpi.csv`
or generated Stage-5 quadrant manifests. Ordinary training rows leave the Q3-specific
columns blank except `q3_gate_eligible`, which may default to `true`.

Manifests are **committed to git** (small). Audio files are not.

### 1.2.2 Seven required manifests

| Manifest File | Purpose | Expected Rows |
|---|---|---|
| `train_content.csv` | Content head training (LibriPhrase) | 40,000–60,000 |
| `train_speaker.csv` | Speaker head training (VoxCeleb2 or fallback) | 200,000–500,000 |
| `train_gsc.csv` | Stage 1 backbone pretrain (GSC v2) | ~85,000 |
| `dev_content.csv` | Content validation, speaker-disjoint | ~5,000 |
| `dev_speaker.csv` | Speaker validation, speaker-disjoint | ~30,000 |
| `test_kpi.csv` | 4-quadrant KPI eval set | ~2,000 |
| `test_fa.csv` | 10 hours of continuous non-target audio | ~600 rows |

**Note on `train_speaker.csv` size.** v1 specified ~1,000,000 rows. That is unrealistic for hackathon throughput. v2 caps at ~500,000 with explicit per-speaker subsampling: max 50 utterances per speaker, ~10,000 speakers.

**`test_fa.csv` source contract.** This manifest contains continuous non-target audio
segments drawn from MUSAN speech/noise/music plus non-enrolled LibriSpeech/Common Voice
utterances that do not contain any enrolled keyword. Each row represents one continuous
segment with `duration_s`; the KPI suite sums durations until it reaches 10 hours. Rows
must be speaker-disjoint from `test_kpi.csv` enrolled users.

### 1.2.3 Label mapping files

Raw manifest values remain human-readable strings. Dataset classes convert them to integer
labels using committed mapping files generated by `scripts/prepare_manifests.py`:

```
data/manifests/keyword_vocab.json      # LibriPhrase wake phrases: {"hey prism": 0, ...}
data/manifests/gsc_vocab.json          # exactly 35 GSC v2 command labels for Stage 1
data/manifests/speaker_vocab.json      # {"speaker_raw_id": 0, ...}
data/manifests/source_vocab.json       # optional, for diagnostics only
```

Dataset output uses `keyword_label: int` and `speaker_label: int`. It may also include
`speaker_id_raw: str` and `keyword_text: str` for logging/debugging. Collation MUST use the
integer labels, never raw strings.

**Keyword OOV policy for speaker manifests.** `keyword_vocab.json` is deliberately a
wake-phrase vocabulary, not a general transcript vocabulary. Rows from
`train_speaker.csv` / `dev_speaker.csv` often have arbitrary VoxCeleb/Common Voice
transcripts that are not valid wake phrases. For any `DualHeadDataset` row whose
`keyword_text` is empty, missing, or absent from `keyword_vocab.json`, set:

```python
from solospeak.utils.types import WORD_IGNORE_INDEX  # defined once in types.py
keyword_label = WORD_IGNORE_INDEX
```

`-100` is chosen because it is PyTorch's default `CrossEntropyLoss(ignore_index=...)`.
Training code MUST mask out `keyword_label == -100` for every word-classification CE
term, including auxiliary word CE and word-adversary CE. Do not add an `UNK` wake-word
class and do not silently hash arbitrary transcripts into `keyword_vocab.json`.

### 1.2.4 Speaker-disjoint split algorithm

`solospeak/data/splits.py`:

```python
"""Speaker-disjoint train/dev/test split generation.

Invariant: no speaker appears in more than one split.
Enforced by deterministic hashing of speaker_id.
"""
import hashlib


def assign_split(speaker_id: str, seed: int = 42,
                 train_pct: float = 0.85,
                 dev_pct: float = 0.10) -> str:
    """Deterministic, reproducible, speaker-disjoint."""
    h = hashlib.sha256(f"{seed}:{speaker_id}".encode()).hexdigest()
    pct = int(h[:8], 16) / 0xFFFFFFFF
    if pct < train_pct:
        return "train"
    if pct < train_pct + dev_pct:
        return "dev"
    return "test"
```

🧪 **Test:** `tests/unit/test_splits.py` verifies:
- No speaker in multiple splits.
- Split proportions within ±2% of targets.
- Reproducible across runs with the same seed.

### 1.2.5 Statistics summary

`scripts/prepare_manifests.py` writes `data/manifests/STATS.md` and
`data/manifests/STATS.json`. The JSON is consumed at training time to inject
`n_gsc_classes`, `n_aux_word_classes`, and `n_aux_speaker_classes` into the config (see
`TrainingConfig`).

Usage:

```bash
python -m scripts.prepare_manifests          # full manifests from real corpora
python -m scripts.prepare_manifests --smoke  # tiny deterministic manifests for CI/dev
```

`--smoke` writes the same seven manifest filenames and vocab/STATS files, but with tiny
row counts over fixture/generated audio. It is for contract tests only and must set
`source_dataset=smoke`.

**Smoke `train_gsc.csv` contract.** The smoke path must still preserve the Stage-1 class
shape. It writes exactly 350 GSC rows: 35 canonical GSC v2 labels × 10 generated clips per
label. Generate deterministic 1.6 s, 16 kHz mono wavs under `data/raw/smoke/gsc/` using a
fixed NumPy seed. The waveform may be random noise plus a class-dependent sine tone; the
audio only needs to satisfy loader/shape contracts, not train a useful classifier.

`gsc_vocab.json` must contain exactly these labels in this order:

```text
backward, bed, bird, cat, dog, down, eight, five, follow, forward, four, go,
happy, house, learn, left, marvin, nine, no, off, on, one, right, seven,
sheila, six, stop, three, tree, two, up, visual, wow, yes, zero
```

`STATS.json` must report `n_gsc_classes = 35` separately from
`n_aux_word_classes = len(keyword_vocab.json)`. Stage 1 uses `n_gsc_classes` /
`gsc_vocab.json`; Stages 2-4 use `n_aux_word_classes` / `keyword_vocab.json`.
For LibriPhrase manifests, `STATS.json` must also distinguish `row_count` from
`unique_phrase_speaker_pairs`; `train_content.csv` row targets always refer to manifest
rows / utterance segments, not unique identities.

Smoke `STATS.md` / `STATS.json` must use tiny counts for all seven manifests. Example:

```
train_gsc.csv:       350 rows, 35 classes, 10 clips/class
train_content.csv:    80 rows,  8 speakers, 8 unique keywords
train_speaker.csv:    80 rows, 12 speakers
dev_content.csv:      24 rows,  4 speakers, 4 unique keywords
dev_speaker.csv:      24 rows,  4 speakers
test_kpi.csv:         40 rows,  4 profiles,  4 unique keywords
                     Q1: 10, Q2: 10, Q3: 10, Q4: 10
test_fa.csv:           6 rows, total_duration_s >= 60
n_gsc_classes:        35
n_aux_word_classes:    8
n_aux_speaker_classes: 12
```

Full-path stats use the expected row ranges from 1.2.2 instead; do not copy the smoke
counts into full data reports.

### 1.2.6 LMDB packing and profile files

`scripts/pack_audio_lmdb.py` is the only supported way to produce the Kaggle-ready audio
store referenced in 1.1.5.

Contract:
- Reads the seven manifests plus any generated Stage-5 quadrant manifests.
- For each row, loads `file_path`, crops `[start_s, end_s]`, resamples to 16 kHz mono if
  needed, and stores the resulting phrase/clip audio under a deterministic LMDB key.
- Writes `data/processed/audio.lmdb` and `data/processed/audio_lmdb_index.json`.
- Rewrites packed manifests so `file_path` is `lmdb://data/processed/audio.lmdb/<key>`,
  `start_s=0.0`, `end_s=duration_s`, and no downstream dataset loader needs the raw
  corpus path.
- Is idempotent: re-running with unchanged source rows produces the same keys and skips
  already-packed clips whose checksum matches.

**LMDB key format (canonical — both `pack_audio_lmdb.py` and all dataset loaders MUST use this exact formula):**

```python
import hashlib

def make_lmdb_key(file_path: str, start_s: float, end_s: float) -> str:
    """Deterministic LMDB key for a cropped audio clip.

    Key is the first 24 hex characters of the SHA-256 digest of
    '{file_path}:{start_s:.6f}:{end_s:.6f}'. This is long enough to be
    collision-free across all expected corpus sizes (< 10M clips).
    """
    raw = f"{file_path}:{start_s:.6f}:{end_s:.6f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]
```

Define `make_lmdb_key` in `solospeak/utils/audio.py` and import it in both
`scripts/pack_audio_lmdb.py` and `solospeak/data/datasets.py`. Never derive the key
differently in the two files — key format mismatches cause silent lookup failures at
training time.

Stage-5 and KPI evaluation profiles are not stored as giant vectors in CSV. They live
under `data/processed/enrollment_profiles/` as `.npz` files named by `profile_id`.
Each file contains `content_template`, `speaker_template`, `tau`, `user_id`,
`keyword_text`, and the manifest row IDs used for enrollment. `profile_path` in
`test_kpi.csv` / quadrant manifests points to this file. If templates have not yet been
computed, `profile_path` may be blank, but `profile_id`, `enrolled_user_id`, and
`enrolled_keyword_text` must still be populated so the profile builder can create it.

## 1.3 Audio Feature Extraction

### 1.3.1 Two implementations — training vs deployment

Both use **identical algorithms** (same window, padding, mel scale, log-clamp). Numerical parity is enforced by `tests/unit/test_features.py::test_train_deploy_parity` with absolute tolerance `3e-3` (the empirical max divergence due to PyTorch's FFT vs NumPy's `pocketfft` accumulating float32 rounding differently).

`solospeak/data/features.py` — training-time, pure PyTorch:

```python
"""Log-mel extraction, training variant.

Pure PyTorch using torch.stft — no torchaudio dependency.
"""
import torch
from torch import nn

from solospeak.utils.config import AudioConfig


def _build_mel_filterbank_torch(config: AudioConfig) -> torch.Tensor:
    """HTK mel filterbank — (n_fft//2 + 1, n_mels) float32."""
    n_freqs = config.n_fft // 2 + 1
    freq_bins = torch.linspace(0.0, config.sample_rate / 2.0, n_freqs)
    m_min = 2595.0 * torch.log10(torch.tensor(1.0 + config.fmin / 700.0))
    m_max = 2595.0 * torch.log10(torch.tensor(1.0 + config.fmax / 700.0))
    mel_pts = torch.linspace(m_min.item(), m_max.item(), config.n_mels + 2)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    f = freq_bins.unsqueeze(1)
    lo, center, hi = hz_pts[:-2].unsqueeze(0), hz_pts[1:-1].unsqueeze(0), hz_pts[2:].unsqueeze(0)
    lower = (f - lo) / (center - lo).clamp(min=1e-10)
    upper = (hi - f) / (hi - center).clamp(min=1e-10)
    return torch.clamp(torch.minimum(lower, upper), min=0.0).float()


class LogMelExtractor(nn.Module):
    fb: torch.Tensor  # (n_fft//2+1, n_mels) registered buffer

    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("fb", _build_mel_filterbank_torch(config))

    def _fix_time_dim(self, log_mel: torch.Tensor) -> torch.Tensor:
        """Center-crop or right-pad to config.window_frames."""
        target = self.config.window_frames
        frames = log_mel.shape[-1]
        if frames == target:
            return log_mel
        if frames > target:
            start = (frames - target) // 2
            return log_mel[..., start:start + target]
        return torch.nn.functional.pad(log_mel, (0, target - frames))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """(B, T=25600) -> (B, 1, n_mels=80, window_frames=160)."""
        c = self.config
        window = torch.hann_window(c.win_length, device=waveform.device, dtype=waveform.dtype)
        stft = torch.stft(
            waveform,
            n_fft=c.n_fft,
            hop_length=c.hop_length,
            win_length=c.win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = stft.abs().pow(2)
        mel = torch.einsum("bft,fm->bmt", power, self.fb)
        log_mel = torch.log(mel.clamp(min=1e-6))
        log_mel = self._fix_time_dim(log_mel)
        return log_mel.unsqueeze(1)
```

`solospeak/data/features_deploy.py` — deployment-time, no PyTorch import:

```python
"""Log-mel extraction, deployment variant.

Pure NumPy implementation of the same algorithm as LogMelExtractor. This file
is used by the CLI demo and by ONNX Runtime validation before the mel tensor is
fed to the exported model.
"""
import numpy as np

from solospeak.utils.config import AudioConfig


def _build_mel_filterbank_numpy(config: AudioConfig) -> np.ndarray:
    """HTK mel filterbank — (n_fft//2 + 1, n_mels) float32."""
    n_freqs = config.n_fft // 2 + 1
    freq_bins = np.linspace(
        0.0, config.sample_rate / 2.0, n_freqs, dtype=np.float32,
    )
    m_min = np.float32(2595.0) * np.log10(np.float32(1.0 + config.fmin / 700.0))
    m_max = np.float32(2595.0) * np.log10(np.float32(1.0 + config.fmax / 700.0))
    mel_pts = np.linspace(float(m_min), float(m_max), config.n_mels + 2, dtype=np.float32)
    hz_pts = np.float32(700.0) * (
        np.float32(10.0) ** (mel_pts / np.float32(2595.0)) - np.float32(1.0)
    )
    f = freq_bins[:, None]
    lo, center, hi = hz_pts[:-2][None, :], hz_pts[1:-1][None, :], hz_pts[2:][None, :]
    lower = (f - lo) / np.maximum(center - lo, np.float32(1e-10))
    upper = (hi - f) / np.maximum(hi - center, np.float32(1e-10))
    return np.maximum(np.minimum(lower, upper), np.float32(0.0)).astype(np.float32)


class LogMelExtractorDeploy:
    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self.fb = _build_mel_filterbank_numpy(config)

    def _fix_time_dim(self, log_mel: np.ndarray) -> np.ndarray:
        """Center-crop or right-pad to config.window_frames."""
        target = self.config.window_frames
        frames = log_mel.shape[-1]
        if frames == target:
            return log_mel
        if frames > target:
            start = (frames - target) // 2
            return log_mel[..., start:start + target]
        return np.pad(log_mel, ((0, 0), (0, target - frames)), mode="constant")

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """(T=25600,) float32 -> (1, n_mels=80, window_frames=160)."""
        c = self.config
        x = waveform.astype(np.float32, copy=False)
        if x.ndim != 1:
            raise ValueError(f"Expected mono waveform shape (T,), got {x.shape}.")
        if c.win_length != c.n_fft:
            raise ValueError("Deployment extractor requires win_length == n_fft for parity.")

        pad = c.n_fft // 2
        x = np.pad(x, pad_width=pad, mode="reflect")

        n = np.arange(c.win_length, dtype=np.float32)
        window = np.float32(0.5) * (
            np.float32(1.0) - np.cos(np.float32(2.0 * np.pi / c.win_length) * n)
        )
        n_frames = 1 + (len(x) - c.n_fft) // c.hop_length
        starts = np.arange(n_frames)[:, None] * c.hop_length
        idx = starts + np.arange(c.n_fft)[None, :]
        frames = x[idx].astype(np.float32)
        frames *= window[None, :]

        spectrum = np.fft.rfft(frames, n=c.n_fft, axis=-1)
        power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)
        mel = (power @ self.fb).astype(np.float32)              # (frames, mels)
        log_mel = np.log(np.maximum(mel, np.float32(1e-6))).astype(np.float32).T
        return self._fix_time_dim(log_mel)[None, :, :]
```

🧪 **CRITICAL TEST:** `tests/unit/test_features.py::test_train_deploy_parity` runs both
extractors on a synthetic 1.6 s waveform and asserts
`np.testing.assert_allclose(train, deploy, atol=3e-3)`.

### 1.3.2 Fixed-length window for deployment

The deployed ONNX model takes a fixed `(1, 1, 80, 160)` mel input derived from exactly
25,600 audio samples (1.6 s at 16 kHz). Training-time data is padded or center-cropped
to `AudioConfig.window_samples` via `solospeak/utils/audio.py::pad_or_crop_to_window`.
The feature extractor then center-crops/right-pads the mel axis to
`AudioConfig.window_frames=160`.
Streaming inference uses the same 1.6 s context with a 160 ms hop so the demo can meet
the post-window latency target.

This decision is final: **no variable-length inference**. It simplifies ONNX export, ONNX Runtime memory layout, and on-device buffering.

## 1.4 Augmentation Pipeline

### 1.4.1 Augmentation operator inventory

Operators are split into two families by domain. The waveform augmenters run before
feature extraction; the mel augmenter runs after. The DataLoader pipeline applies waveform
augmenters first, then calls `LogMelExtractor`, then applies mel augmenters.

**Waveform augmenters** — interface: `__call__(wav: Tensor) -> Tensor`

| Operator | Purpose | Probability | Parameters |
|---|---|---|---|
| `AddNoise` | MUSAN noise mixing at random SNR | 0.7 | snr_db ∈ [−5, 30] |
| `ConvolveRIR` | Room impulse response convolution | 0.5 | RIR sampled from OpenSLR-28 |
| `GainJitter` | Random linear gain | 0.3 | gain ∈ [−6, +6] dB |
| `TimeShift` | Circular shift within window | 0.3 | shift ∈ [−100, +100] ms |
| `PitchShift` | Small pitch shift | 0.1 | semitones ∈ [−1, +1] |

**Mel augmenters** — interface: `__call__(mel: Tensor) -> Tensor` (operates on log-mel spectrogram, NOT raw waveform)

| Operator | Purpose | Probability | Parameters |
|---|---|---|---|
| `SpecAugment` | Time + frequency masking on mel | 0.5 | 2 masks each, ≤ 20% coverage |

### 1.4.2 Curriculum SNR (novelty claim 4)

`solospeak/data/augmentation.py`:

```python
class CurriculumAugmenter:
    """Progressively widens noise/distance range during training.

    Unlike uniform sampling, we start with mild augmentation and widen to the
    full range by 50% of total training steps.
    """
    def __init__(self, total_steps: int,
                 snr_final: tuple[int, int] = (-5, 30),
                 distance_final: tuple[float, float] = (0.5, 5.0)):
        self.total_steps = total_steps
        self.snr_final = snr_final
        self.distance_final = distance_final

    def current_ranges(self, step: int) -> dict[str, tuple[int, int] | tuple[float, float]]:
        progress = min(1.0, step / (0.5 * self.total_steps))
        snr_min = int(30 - progress * (30 - self.snr_final[0]))
        snr_max = self.snr_final[1]
        dist_max = 0.5 + progress * (self.distance_final[1] - 0.5)
        # Key names match DataConfig field names exactly: snr_range_db and distance_range_m.
        return {"snr_range_db": (snr_min, snr_max),
                "distance_range_m": (0.5, dist_max)}
```

### 1.4.3 Hard negative mining

`solospeak/data/hard_negatives.py`:

```python
"""Phonetic + speaker hard-negative mining.

Hard negatives are computed at the start of each epoch (not per-batch) to
keep the training loop fast.
"""

def phone_edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance over phoneme token lists."""
    prev = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        cur = [i]
        for j, token_b in enumerate(b, start=1):
            cost = 0 if token_a == token_b else 1
            cur.append(min(
                prev[j] + 1,       # deletion
                cur[j - 1] + 1,    # insertion
                prev[j - 1] + cost # substitution
            ))
        prev = cur
    return prev[-1]


def phonetic_hard_negatives(keyword: str, vocab: list[str], k: int = 10) -> list[str]:
    """Use g2p_en to find phonetically close words."""
    from g2p_en import G2p
    g2p = G2p()
    target_phones = g2p(keyword)
    distances = [(w, phone_edit_distance(target_phones, g2p(w))) for w in vocab]
    distances.sort(key=lambda x: x[1])
    return [w for w, _ in distances[1:k+1]]


def speaker_hard_negatives(anchor_id: str, embeddings: np.ndarray,
                           speaker_ids: list[str], k: int = 10) -> list[str]:
    """Find speakers whose ECAPA-TDNN embedding is closest to the anchor."""
    anchor_idx = speaker_ids.index(anchor_id)
    anchor_vec = embeddings[anchor_idx]
    scores = embeddings @ anchor_vec
    top_idx = np.argsort(-scores)[1:k+1]
    return [speaker_ids[i] for i in top_idx]
```

**ECAPA-TDNN dependency.** Speaker hard-negative mining requires precomputed speaker
embeddings. We use `speechbrain.inference.speaker.EncoderClassifier` with the
`speechbrain/spkrec-ecapa-voxceleb` model. The environment pins `torch==2.3.1`,
`torchaudio==2.3.1`, and `speechbrain==1.0.3`; do not downgrade SpeechBrain to 1.0.0 or
leave torchaudio unconstrained, because that pairing is known to produce brittle import
and runtime behavior around the pretrained inference API.

`scripts/precompute_speaker_embeddings.py`:
- Loads ECAPA-TDNN once.
- Processes every speaker in `train_speaker.csv` (one ~3 s clip per speaker is enough).
- Saves to `data/processed/speaker_embeddings.npy` (~30 minutes one-time).
- Also saves `data/processed/speaker_embedding_ids.json` for index-to-ID mapping.

## 1.5 DataLoader Assembly

### 1.5.1 Datasets

`solospeak/data/datasets.py`:

```python
"""PyTorch Dataset classes for each training stage.

Stage 1: GSCDataset           — 35-class KWS classification
Stage 2-4: DualHeadDataset    — yields (wav, keyword_label, speaker_label) triplets
Stage 5: QuadrantDataset      — yields trial audio + quadrant label + profile templates
"""
```

Each dataset yields a `dict`. Exact schemas per stage:

- **GSCDataset** (Stage 1):
  - `wav: Tensor(25600,)` — fixed-length, float32, [-1, 1]
  - `keyword_label: int` — integer class index into `gsc_vocab.json`
  - `keyword_text: str` — raw GSC command string (e.g. `"yes"`)

- **DualHeadDataset** (Stages 2–4):
  - `wav: Tensor(25600,)` — fixed-length, float32, [-1, 1]
  - `keyword_label: int` — index into `keyword_vocab.json`, or `WORD_IGNORE_INDEX = -100`
    when the row comes from a speaker-only manifest whose transcript is not a wake phrase
  - `speaker_label: int` — index into `speaker_vocab.json`
  - `speaker_id_raw: str` — raw speaker ID string (for logging/debugging only)
  - `keyword_text: str` — raw phrase string (for logging/debugging only)

- **QuadrantDataset** (Stage 5):
  - `wav: Tensor(25600,)` — fixed-length, float32, [-1, 1]
  - `quadrant_label: int` — 1 for Q1 (accept), 0 for Q2/Q3/Q4 (reject)
  - `quadrant_class: str` — one of `"Q1_accept"`, `"Q2_imposter"`, `"Q3_wrong_word"`, `"Q4_background"`
  - `user_id: str` — enrolled user ID this sample is evaluated against
  - `keyword_text: str` — trial utterance keyword/text
  - `enrolled_keyword_text: str` — enrolled keyword for this profile
  - `profile_id: str` — key for the enrolled profile
  - `trial_source: str` — `"real"`, `"augmented"`, `"tts"`, or `"background"`
  - `q3_gate_eligible: bool` — true iff the row is allowed in the hard quadrant gate
  - `synthesis_backend: str` — backend name for synthetic rows, else `""`
  - `speaker_verification_score: float | None` — cosine score for synthetic right-voice checks
  - `content_template: Tensor(128,)` — L2-normalized content enrollment template for this user
  - `speaker_template: Tensor(128,)` — L2-normalized speaker enrollment template for this user

**`profile_path` blank-case contract.** The spec notes that `profile_path` may be blank
before enrollment profiles have been generated. `QuadrantDataset` MUST raise
`FileNotFoundError` if a row's `profile_path` is blank or the pointed-to `.npz` file
does not exist. Do NOT silently return zero-filled templates — a blank path means the
dataset was instantiated before enrollment profiles were built, which is a usage error.
`QuadrantDataset` should only be constructed after `scripts/pack_audio_lmdb.py` and the
enrollment profile builder have both completed successfully.

### 1.5.2 Class-aware sampler for SupCon

`solospeak/data/samplers.py`:

```python
"""Class-aware batch sampler.

SupCon needs at least 2 positives per class in a batch. A pure random sampler
gives many singleton classes per batch, which makes SupCon's positive set
empty for those classes and silently drops their gradient.

This sampler guarantees `min_positives_per_class` positives for each class
that appears in a batch.
"""
class ClassAwareBatchSampler(Sampler[list[int]]):
    # Must inherit from torch.utils.data.Sampler[list[int]] so that DataLoader
    # recognises it as a batch sampler and does not apply default sampling on top.
    def __init__(
        self,
        labels: list[int],
        batch_size: int,
        num_classes_per_batch: int = 8,
        num_samples_per_class: int = 16,
        seed: int = 42,
    ) -> None: ...
    def __iter__(self) -> Iterator[list[int]]: ...
    def __len__(self) -> int: ...
```

Used by both content (label = keyword class) and speaker (label = integer speaker label)
DataLoaders in Stages 2–4.

Invariant: `num_classes_per_batch * num_samples_per_class == batch_size`, and
`num_samples_per_class >= 2`. If a class has fewer than `num_samples_per_class` examples,
sample with replacement inside that class.

### 1.5.3 Batch collation

`🔴 BLOCKING`. All wavs are pre-padded to fixed length in `__getitem__`, so collation is straightforward stacking — no variable-length collator needed.

Three collators are defined — one per dataset type. Pass the appropriate one as `collate_fn` to `DataLoader`.

```python
def parse_bool_csv(value: DatasetValue) -> bool:
    """Parse canonical manifest booleans. Never replace this with bool(value)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"Expected manifest boolean 'true'/'false', got {value!r}")


def collate_gsc(batch: list[DatasetItem]) -> BatchDict:
    """For GSCDataset (Stage 1)."""
    return {
        "wav": torch.stack([b["wav"] for b in batch]),
        "keyword_label": torch.tensor([b["keyword_label"] for b in batch], dtype=torch.long),
    }


def collate_dual_head(batch: list[DatasetItem]) -> BatchDict:
    """For DualHeadDataset (Stages 2–4)."""
    return {
        "wav": torch.stack([b["wav"] for b in batch]),
        "keyword_label": torch.tensor([b["keyword_label"] for b in batch], dtype=torch.long),
        "speaker_label": torch.tensor([b["speaker_label"] for b in batch], dtype=torch.long),
    }


def collate_quadrant(batch: list[DatasetItem]) -> BatchDict:
    """For QuadrantDataset (Stage 5).

    Templates are stacked as tensors; string fields are kept as lists because
    torch.stack does not handle strings.
    """
    return {
        "wav": torch.stack([b["wav"] for b in batch]),
        "quadrant_label": torch.tensor([b["quadrant_label"] for b in batch], dtype=torch.long),
        "quadrant_class": [b["quadrant_class"] for b in batch],   # list[str]
        "user_id": [b["user_id"] for b in batch],                 # list[str]
        "keyword_text": [b["keyword_text"] for b in batch],       # list[str]
        "enrolled_keyword_text": [b["enrolled_keyword_text"] for b in batch],
        "profile_id": [b["profile_id"] for b in batch],           # list[str]
        "trial_source": [b["trial_source"] for b in batch],       # list[str]
        "q3_gate_eligible": torch.tensor(
            [parse_bool_csv(b["q3_gate_eligible"]) for b in batch], dtype=torch.bool,
        ),
        "synthesis_backend": [str(b["synthesis_backend"]) for b in batch],
        "speaker_verification_score": torch.tensor([
            float("nan") if b["speaker_verification_score"] is None
            else float(b["speaker_verification_score"])
            for b in batch
        ], dtype=torch.float32),
        "content_template": torch.stack([b["content_template"] for b in batch]),
        "speaker_template": torch.stack([b["speaker_template"] for b in batch]),
    }
```

**Usage rule.** `collate_fixed_length` is the legacy name kept for backward compatibility; it is an alias for `collate_dual_head`. New code should call the named variant directly.

### 1.5.4 Acceptance criteria — Phase 1

- 🧪 Smoke path: `make download-data-smoke && make prepare-manifests-smoke` completes on
  a fresh clone without network access or large corpora.
- 🧪 Full path: `make download-data` completes; idempotent on re-run. This is not a CI
  requirement and may require tens/hundreds of GB depending on selected corpora.
- 🧪 `make prepare-manifests` produces 7 manifest files + `STATS.md` + `STATS.json`.
- 🧪 `make pack-lmdb` produces `data/processed/audio.lmdb`, rewrites packed manifest
  paths to `lmdb://...`, and is idempotent on re-run.
- 🧪 `data/manifests/keyword_vocab.json`, `gsc_vocab.json`, and `speaker_vocab.json`
  exist and are used by datasets to emit integer labels.
- 🧪 Smoke `train_gsc.csv` contains 350 rows, exactly 35 classes, and
  `STATS.json["n_gsc_classes"] == 35`.
- 🧪 Speaker-manifest rows with OOV transcripts emit `keyword_label == -100`, and word CE
  losses ignore those rows instead of creating an `UNK` class.
- 🧪 `tests/integration/test_data_pipeline.py::test_no_speaker_overlap` passes.
- 🧪 `tests/unit/test_features.py::test_train_deploy_parity` passes (`atol=3e-3`).
- 🧪 Benchmark: `DataLoader(DualHeadDataset, batch_size=128, num_workers=4)` achieves `MIN ≥ 80 samples/sec` on CPU. (TARGET ≥ 200 samples/sec; failing TARGET is acceptable, failing MIN is not.)
- 🧪 Augmented batch passes shape, dtype, and value-range asserts.

---

# Phase 2: Core Model

**Duration:** Days 9–14.
**Goal:** All neural modules implemented, unit-tested, and callable end-to-end on a batch of mel spectrograms.
**Exit criterion:** `model(mel)` returns correctly-shaped `(z_c, z_s)` with parameter counts matching the table in 2.1.6 within ±2%; backward pass works; ONNX export succeeds.

> **READ THIS BEFORE IMPLEMENTING PHASE 2.**
>
> The original v1 of this document said "BC-ResNet-8 has ~1M params" and "BC-ResNet-N has base 2N channels" — but did NOT define the block layers. An implementing agent had to reverse-engineer the architecture from the parameter count, which created a loop where the agent oscillated between depthwise-separable convs (too few params), full convs (too many), and bottleneck designs (something else again).
>
> v2 fixes this by defining the EXACT block layers, EXACT channel progression, and shipping with ANALYTICALLY COMPUTED parameter counts. The test ranges in 2.1.6 are derived from the architecture, not invented. **Do not change the architecture to hit a different parameter target. Implement what is specified, verify the params match section 2.1.6, and move on.**
>
> This is a SoloSpeak-specific architecture. It is INSPIRED BY but NOT IDENTICAL TO the BC-ResNet paper (Kim et al. 2021). Do not consult the paper to "correct" this spec — the spec is the contract.
> Internally the class is named `SoloSpeakResNet` to avoid implying that it implements
> broadcasted residuals. The config aliases remain `bcresnet1/5/8/10/16` for continuity.

## 2.1 Backbone Implementation — SoloSpeakResNet

### 2.1.1 Architecture overview

SoloSpeakResNet is a 4-stage residual CNN with three downsampling transitions between
stages, followed by a frequency pool that collapses the frequency axis. It is a standard
ResNet-style design, deliberately NOT using the broadcasted-residual pattern from the
original BC-ResNet paper. We keep the `bcresnet*` config variant names only as aliases for
the size ladder; the implementation class name is `SoloSpeakResNet`.

### 2.1.2 Exact block definitions

`solospeak/models/backbones/bcresnet.py`:

```python
"""SoloSpeakResNet backbone.

A 4-stage residual CNN. Plain Conv2d blocks (no depthwise separable, no
broadcast paths). Project-specific architecture inspired by but not identical
to Kim et al. 2021.

INPUT:  (B, 1, 80, T)        — log-mel, 80 mel bands
OUTPUT: (B, C_out, 1, T/8)   — frequency collapsed by AdaptiveAvgPool

For T = 160 (the production deployment size), output time dim = 20.
"""
from typing import TypedDict

import torch
from torch import nn


class _VariantSpec(TypedDict):
    base: int
    num_blocks: list[int]


VARIANTS: dict[str, _VariantSpec] = {
    "bcresnet1":  {"base":  8, "num_blocks": [1, 1, 1, 1]},
    "bcresnet5":  {"base": 16, "num_blocks": [2, 2, 3, 3]},
    "bcresnet8":  {"base": 24, "num_blocks": [2, 2, 3, 2]},
    "bcresnet10": {"base": 32, "num_blocks": [2, 2, 2, 2]},
    "bcresnet16": {"base": 40, "num_blocks": [2, 2, 3, 3]},
}
# Channel progression for all variants: [base, 2*base, 3*base, 4*base]
# Output channels = 4 * base (= last stage channels).


class NormalBlock(nn.Module):
    """Two 3x3 convs with a residual connection.

    PARAMS: 18*C^2 + 4*C  (two 3x3 convs without bias, plus two BN with 2*C each).

    Layers:
        Conv2d(C, C, kernel=3, padding=1, bias=False)   ->  9*C^2
        BatchNorm2d(C)                                   ->  2*C
        ReLU(inplace=True)
        Conv2d(C, C, kernel=3, padding=1, bias=False)   ->  9*C^2
        BatchNorm2d(C)                                   ->  2*C
        Add residual (input shortcut)
        ReLU(inplace=True)
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

    def fuse_model(self) -> None:
        """Fuse Conv+BN pairs for export/quantization.

        ReLU is intentionally left unfused inside residual blocks because one
        module instance is used in two places. Conv+BN fusion gives the export
        speedup we need without changing the forward graph semantics.
        """
        torch.ao.quantization.fuse_modules(
            self, [["conv1", "bn1"], ["conv2", "bn2"]], inplace=True,
        )


class TransitionBlock(nn.Module):
    """Halves both frequency and time dimensions and changes channel count.

    PARAMS: 9*C_in*C_out + 9*C_out^2 + C_in*C_out + 6*C_out

    Layers:
        Conv2d(C_in, C_out, kernel=3, stride=2, padding=1, bias=False)
        BatchNorm2d(C_out)
        ReLU(inplace=True)
        Conv2d(C_out, C_out, kernel=3, padding=1, bias=False)
        BatchNorm2d(C_out)
        Shortcut: Conv2d(C_in, C_out, kernel=1, stride=2, bias=False) + BatchNorm2d
        Add
        ReLU(inplace=True)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

    def fuse_model(self) -> None:
        """Fuse Conv+BN pairs for export/quantization."""
        torch.ao.quantization.fuse_modules(
            self, [["conv1", "bn1"], ["conv2", "bn2"]], inplace=True,
        )
        torch.ao.quantization.fuse_modules(self.shortcut, [["0", "1"]], inplace=True)


class SoloSpeakResNet(nn.Module):
    """SoloSpeak residual CNN backbone.

    Stride schedule (for input T=160, F=80):
        Input:                         (B,  1, 80, 160)
        Stem (stride (2,1) in (F,T)):  (B, c0, 40, 160)
        Stage 0 (no downsample):       (B, c0, 40, 160)
        Trans1 (stride 2,2):           (B, c1, 20,  80)
        Stage 1:                       (B, c1, 20,  80)
        Trans2 (stride 2,2):           (B, c2, 10,  40)
        Stage 2:                       (B, c2, 10,  40)
        Trans3 (stride 2,2):           (B, c3,  5,  20)
        Stage 3:                       (B, c3,  5,  20)
        AdaptiveAvgPool2d((1, None)):  (B, c3,  1,  20)

    where c0..c3 = [base, 2*base, 3*base, 4*base].

    Final output time dimension = T/8 (three stride-2-in-time transitions).
    """

    def __init__(self, variant: str = "bcresnet8") -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}. Choose from {list(VARIANTS)}.")
        spec = VARIANTS[variant]
        base = spec["base"]
        num_blocks = spec["num_blocks"]
        channels = [base * (i + 1) for i in range(4)]
        self._output_channels = channels[-1]

        # Stem: stride 2 in frequency only, stride 1 in time
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=5, stride=(2, 1), padding=2, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Build 4 stages with transitions between them
        stages: list[nn.Module] = []
        for i, n in enumerate(num_blocks):
            if i > 0:
                stages.append(TransitionBlock(channels[i - 1], channels[i]))
            for _ in range(n):
                stages.append(NormalBlock(channels[i]))
        self.stages = nn.Sequential(*stages)

        # Collapse frequency axis only (keep time).
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

    @property
    def output_channels(self) -> int:
        """Channels at the backbone output. Used by SoloSpeakModel to size heads."""
        return self._output_channels

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """(B, 1, 80, T) -> (B, C_out, 1, T/8)."""
        x = self.stem(mel)
        x = self.stages(x)
        x = self.freq_pool(x)
        return x

    def fuse_model(self) -> None:
        """Fuse all supported Conv+BN patterns before ONNX export/quantization."""
        self.eval()
        torch.ao.quantization.fuse_modules(self.stem, [["0", "1", "2"]], inplace=True)
        for block in self.stages:
            if hasattr(block, "fuse_model"):
                block.fuse_model()


# Backward-compatible alias for older tests/scripts. New code should import
# SoloSpeakResNet.
BCResNet = SoloSpeakResNet
```

### 2.1.3 Channel progression

For every variant, the four-stage channel progression is exactly `[base, 2*base, 3*base, 4*base]`:

| Variant | base | Stage channels | Output channels |
|---|---|---|---|
| `bcresnet1`  |  8 |  8 / 16 / 24 / 32 |  32 |
| `bcresnet5`  | 16 | 16 / 32 / 48 / 64 |  64 |
| `bcresnet8`  | 24 | 24 / 48 / 72 / 96 |  96 |
| `bcresnet10` | 32 | 32 / 64 / 96 / 128 | 128 |
| `bcresnet16` | 40 | 40 / 80 / 120 / 160 | 160 |

### 2.1.4 Stride schedule and output shape

For input `(B, 1, 80, T)`:
- Stem stride `(2, 1)` halves F: `(B, c0, 40, T)`.
- Three TransitionBlocks each stride `(2, 2)` halve both F and T.
- Final `AdaptiveAvgPool2d((1, None))` collapses F to 1.
- Output: `(B, c3, 1, T/8)`.

For the production input `T = 160`, output time dim is exactly **20**. The doc nowhere uses "T/16" or "4 transition blocks" — both were errors in v1.

### 2.1.5 Public API: `output_channels`

`SoloSpeakResNet.output_channels` is a property returning the last-stage channel count. `SoloSpeakModel` uses this to size the heads. **Do NOT hardcode a `_backbone_channels = {...}` lookup table anywhere else in the codebase.**

### 2.1.6 Parameter count specification

These numbers are computed analytically from the block formulas in 2.1.2 and verified by `tests/unit/test_backbones.py::test_param_counts`. Implementations that match the spec MUST produce these exact counts.

| Variant | num_blocks | Computed Params | Test Range (±2%) |
|---|---|---|---|
| `bcresnet1`  | [1, 1, 1, 1] |    65,032 |    63,731 –   66,333 |
| `bcresnet5`  | [2, 2, 3, 3] |   512,720 |   502,466 –  522,974 |
| `bcresnet8`  | [2, 2, 3, 2] |   985,080 |   965,378 – 1,004,782 |
| `bcresnet10` | [2, 2, 2, 2] | 1,583,136 | 1,551,473 – 1,614,799 |
| `bcresnet16` | [2, 2, 3, 3] | 3,193,160 | 3,129,297 – 3,257,023 |

🧪 **Test:** `tests/unit/test_backbones.py::test_param_counts`

```python
import pytest
from solospeak.models.backbones.bcresnet import SoloSpeakResNet

EXPECTED = {
    "bcresnet1":     65_032,
    "bcresnet5":    512_720,
    "bcresnet8":    985_080,
    "bcresnet10": 1_583_136,
    "bcresnet16": 3_193_160,
}

@pytest.mark.parametrize("variant,expected", EXPECTED.items())
def test_param_counts(variant, expected):
    model = SoloSpeakResNet(variant)
    actual = sum(p.numel() for p in model.parameters())
    # ±2% tolerance accommodates innocuous implementation choices (e.g. whether
    # to count BN running buffers as params). The numbers above EXCLUDE running
    # buffers because BN running_mean / running_var have requires_grad=False
    # and are not in .parameters().
    lo = int(expected * 0.98)
    hi = int(expected * 1.02)
    assert lo <= actual <= hi, f"{variant}: got {actual}, expected {expected} ±2%"
```

### 2.1.7 Forward shape contract

🧪 **Test:** `tests/unit/test_backbones.py::test_forward_shape`

```python
def test_forward_shape():
    model = SoloSpeakResNet("bcresnet8")
    mel = torch.randn(4, 1, 80, 160)
    out = model(mel)
    assert out.shape == (4, 96, 1, 20), f"Got {out.shape}"
    assert model.output_channels == 96
```

🧪 **Test:** `tests/unit/test_backbones.py::test_backward`

```python
def test_backward():
    model = SoloSpeakResNet("bcresnet8")
    mel = torch.randn(2, 1, 80, 160, requires_grad=True)
    out = model(mel)
    loss = out.sum()
    loss.backward()
    assert mel.grad is not None
    assert torch.isfinite(mel.grad).all()
```

🧪 **Test:** `tests/unit/test_backbones.py::test_onnx_export`

```python
@pytest.mark.onnx
def test_onnx_export(tmp_path):
    model = SoloSpeakResNet("bcresnet8").eval()
    mel = torch.randn(1, 1, 80, 160)
    onnx_path = tmp_path / "bcresnet.onnx"
    torch.onnx.export(model, mel, onnx_path, opset_version=17,
                      input_names=["mel"], output_names=["feat"])
    assert onnx_path.exists()
```

🧪 **Test:** `tests/unit/test_backbones.py::test_fuse_model_export`

```python
@pytest.mark.onnx
def test_fuse_model_export(tmp_path):
    model = SoloSpeakResNet("bcresnet8").eval()
    model.fuse_model()
    mel = torch.randn(1, 1, 80, 160)
    out = model(mel)
    assert out.shape == (1, 96, 1, 20)
    onnx_path = tmp_path / "bcresnet_fused.onnx"
    torch.onnx.export(model, mel, onnx_path, opset_version=17,
                      input_names=["mel"], output_names=["feat"])
    assert onnx_path.exists()
```

### 2.1.8 Acceptance criteria — Phase 2.1

- 🧪 All five variants instantiate without error.
- 🧪 Param counts match table in 2.1.6 within ±2%.
- 🧪 `model(torch.randn(4, 1, 80, 160)).shape == (4, output_channels, 1, 20)`.
- 🧪 `loss.backward()` produces finite gradients on input.
- 🧪 ONNX export at opset 17 succeeds.
- 🧪 `model.fuse_model()` preserves the shape contract and the fused model exports.

**STOP. Do not advance to 2.2 until all tests above pass.** If a param count is off,
double-check that BN bias/weight count is included (each BN contributes `2 * C`
parameters). Do NOT modify the architecture to hit a different number — the architecture
is the contract.

## 2.2 Heads — Content and Speaker

### 2.2.1 `solospeak/models/heads.py`

```python
"""Dual orthogonal heads: content (what was said) and speaker (who said it).

Each head:
    Input:  backbone feature map (B, C, 1, T')
    Output: (B, 128), L2-normalized

Identical architecture for both — the orthogonality is enforced by the
training loss, not by structural difference.
"""
import torch
import torch.nn.functional as F
from torch import nn


class EmbeddingHead(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int = 256,
                 output_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(input_channels, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 1, T')
        x = self.pool(x).flatten(1)              # (B, C)
        x = self.drop(self.act(self.fc1(x)))      # (B, hidden_dim)
        x = self.fc2(x)                           # (B, output_dim)
        return F.normalize(x, p=2, dim=-1)        # L2-normalized
```

### 2.2.2 Why two separate heads, not one shared?

`[DECISION]` Separate weight matrices per head, shared backbone features.

The orthogonality loss operates on the cross-covariance of the two output spaces across the batch. If the heads shared weights, `z_c == z_s` by definition, and the loss would be vacuous.

### 2.2.3 Auxiliary classification heads (training-only, stripped at export)

```python
class AuxiliaryHeads(nn.Module):
    """Used in Stages 2-4. Stripped at ONNX export.

    The class counts come from the manifest STATS.json (see TrainingConfig
    in section 0.2.3) — never hardcoded.
    """

    def __init__(self, embed_dim: int, n_words: int, n_speakers: int) -> None:
        super().__init__()
        self.aux_word = nn.Linear(embed_dim, n_words)
        self.aux_speaker = nn.Linear(embed_dim, n_speakers)

    def forward(self, z_c: torch.Tensor, z_s: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.aux_word(z_c), self.aux_speaker(z_s)
```

### 2.2.4 Acceptance criteria — Phase 2.2

- 🧪 `EmbeddingHead(96)(feat).norm(dim=-1)` is `1.0 ± 1e-5` for all rows in the batch.
- 🧪 Content head + speaker head together add ~115K params with `bcresnet8` (= 2 × 57,728; computed in section 2.4.1).
- 🧪 Gradient flows through L2 normalization; no NaNs from accidental zero vectors (handled by `F.normalize`'s built-in epsilon).

## 2.3 Losses

### 2.3.1 Supervised Contrastive Loss

`solospeak/losses/supcon.py`:

```python
"""Supervised Contrastive Loss (Khosla et al. 2020).

For embeddings z in R^(B, D) (L2-normalized) and labels y in {0..K-1}^B:

    L_i = -1/|P(i)| * sum_{p in P(i)} log( exp(z_i.z_p / tau)
                                          / sum_{a != i} exp(z_i.z_a / tau) )

where P(i) = {j != i : y_j == y_i}.

Returns the mean of L_i over anchors with non-empty P(i). Anchors with empty
positive sets contribute 0 to the loss but emit a warning if more than 25%
of anchors lack positives — that means the batch sampler is misconfigured.
"""
import torch
from torch import nn


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeddings: (B, D), L2-normalized
        # labels: (B,)
        ...
```

### 2.3.2 Orthogonality Loss

`solospeak/losses/orthogonality.py`:

```python
"""Frobenius-norm cross-covariance penalty.

Drives E[z_c z_s^T] (centered) to zero — content and speaker embeddings
become decorrelated in expectation.
"""
import torch


def orthogonality_loss(z_c: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
    """(B, D), (B, D) -> scalar."""
    B = z_c.shape[0]
    z_c_centered = z_c - z_c.mean(dim=0, keepdim=True)
    z_s_centered = z_s - z_s.mean(dim=0, keepdim=True)
    C = (z_c_centered.T @ z_s_centered) / max(B - 1, 1)   # (D, D)
    return (C ** 2).sum()
```

### 2.3.3 Gradient Reversal Layer + Adversaries

`solospeak/losses/adversarial.py`:

```python
"""DANN-style gradient reversal."""
import torch
from torch import nn


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx,  # type: ignore[override]
                x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_  # type: ignore[attr-defined]
        return x.view_as(x)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx,  # type: ignore[override]
                 grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None  # type: ignore[attr-defined]


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(x, lambda_)


class AdversarialProbeHead(nn.Module):
    """Trained to predict the 'wrong' attribute from an embedding, behind grad-reverse.

    For Stage 3+ disentanglement:
        content_adversary: predicts speaker-ID from z_c (should fail)
        speaker_adversary: predicts word-class from z_s (should fail)

    Architecture:
        Linear(embed_dim, 256) -> ReLU -> Linear(256, n_classes)

    Larger than v1's single linear layer — empirically a single linear is too
    weak to push the encoder hard enough.
    """

    def __init__(self, embed_dim: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
```

### 2.3.4 Combined multi-loss

`solospeak/losses/combined.py`:

```python
"""Stage-aware weighted loss aggregator."""
```

Stage-specific loss compositions:
- Stage 1: cross-entropy on GSC classification head only.
- Stage 2: `1.0 * L_supcon_c + 1.0 * L_supcon_s + 0.5 * L_aux_ce`.
- Stage 3: Stage 2 + ramped `lambda_ortho * L_ortho + lambda_adv * L_adv`.
- Stage 4: same as Stage 3, with heavy augmentation in the data pipeline.
- Stage 5: BCE on fusion head only (encoder frozen).
- Stage 6: QAT-aware distillation on the clean deployable model; no aux/adversarial
  heads are required.

`CombinedLoss.forward()` receives logical streams, not a single flat batch:

```python
losses = combined_loss(
    outputs={
        "content": content_out,
        "speaker": speaker_out,
    },
    batches={
        "content": content_batch,
        "speaker": speaker_batch,
    },
    step=step,
)
```

For Stage 1 use the key `"gsc"`, for Stage 5 use `"quadrant"`, and for Stage 6 use
`"distill"` plus optional `"quadrant"`. Do not infer ownership from whatever labels happen
to be present in one merged batch.

**Aux CE routing rule.** `L_aux_ce` is not "CE on every label in every interleaved
batch." It is the sum of only the valid, task-owned CE terms:

```python
from solospeak.utils.types import WORD_IGNORE_INDEX  # defined once in types.py — do not re-define

L_aux_word = cross_entropy(
    content_out["word_logits"],
    content_batch["keyword_label"],
    ignore_index=WORD_IGNORE_INDEX,
)
L_aux_speaker = cross_entropy(
    speaker_out["speaker_logits"],
    speaker_batch["speaker_label"],
)
L_aux_ce = L_aux_word + L_aux_speaker
```

In Stages 3-4, adversarial CE follows the same ownership:
`content_adversary(z_c)` predicts speaker IDs on the speaker batch, while
`speaker_adversary(z_s)` predicts word IDs only on rows whose `keyword_label != -100`
(normally the content batch). Any CE term that consumes `keyword_label` MUST pass
`ignore_index=WORD_IGNORE_INDEX` or apply the equivalent mask.

### 2.3.5 Acceptance criteria — Phase 2.3

- 🧪 `SupConLoss` returns a finite, non-zero scalar on a hand-crafted batch with two classes of two samples each (i.e. the simplest case where positives exist).
- 🧪 `SupConLoss` returns `0` (warning emitted) when EVERY label is unique — there are no positives at all, so loss is undefined and should be 0 by convention.
- 🧪 `orthogonality_loss(z_c, z_s)` decreases monotonically over 100 gradient steps when both `z_c` and `z_s` are trainable parameters initialized to correlated random vectors. (We do NOT test "loss = 0 when orthogonal per-dimension" because centered cross-covariance does not equal element-wise orthogonality.)
- 🧪 `grad_reverse(x, 0.5)` flips the sign of the gradient and scales by 0.5 on backward.

## 2.4 SoloSpeakModel Assembly

`solospeak/models/solospeak.py`:

```python
"""Full SoloSpeak model: backbone + content head + speaker head + fusion MLP."""
import torch
from torch import nn

from solospeak.models.backbones.bcresnet import SoloSpeakResNet
from solospeak.models.heads import EmbeddingHead
from solospeak.models.fusion import GatedFusionMLP
from solospeak.utils.config import SoloSpeakConfig


class SoloSpeakModel(nn.Module):
    """📋 CONTRACT

    Forward signature:
        input:  mel of shape (B, 1, 80, T)
        output: (z_c, z_s) — both (B, 128), L2-normalized

    The fusion MLP is a separate submodule. It is NOT part of forward();
    use forward_fusion(s_c, s_s) to apply it.

    Naming convention:
        self.fusion_mlp  — the GatedFusionMLP module (attribute)
        self.forward_fusion(...) — the method that computes fusion output

    The two never share a name. v1 had a name collision; v2 fixes it by
    using forward_fusion() as the method and fusion_mlp as the attribute.
    """

    def __init__(self, config: SoloSpeakConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = SoloSpeakResNet(config.backbone.variant)
        c_out = self.backbone.output_channels        # property — no hardcoded table
        self.content_head = EmbeddingHead(
            c_out, config.heads.hidden_dim, config.heads.content_dim, config.heads.dropout,
        )
        self.speaker_head = EmbeddingHead(
            c_out, config.heads.hidden_dim, config.heads.speaker_dim, config.heads.dropout,
        )
        self.fusion_mlp = GatedFusionMLP(config.fusion)

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(mel)
        return self.content_head(feat), self.speaker_head(feat)

    def forward_fusion(self, s_c: torch.Tensor, s_s: torch.Tensor) -> torch.Tensor:
        return self.fusion_mlp(s_c, s_s)
```

### 2.4.1 Parameter budget verification

For `bcresnet8` (the default):

| Component | Params |
|---|---|
| SoloSpeakResNet backbone | 985,080 |
| Content head (96 → 256 → 128) | 57,728 |
| Speaker head (96 → 256 → 128) | 57,728 |
| Fusion MLP (361, see 2.5) | 361 |
| **Total deployable** | **1,100,897** |

The "0.9M adversarial heads" mentioned in v1 are training-only and live in a separate `TrainingWrapper` module. They are NOT part of `SoloSpeakModel` and are NOT in the deployed ONNX graph.

🧪 **Test:** `tests/unit/test_solospeak.py::test_total_param_count`

```python
def test_total_param_count_bcresnet8():
    cfg = SoloSpeakConfig.from_yaml("configs/defaults.yaml")
    model = SoloSpeakModel(cfg)
    n = sum(p.numel() for p in model.parameters())
    assert 1_080_000 <= n <= 1_125_000, f"Got {n}"
```

## 2.5 Gated Fusion MLP

`solospeak/models/fusion.py`:

```python
"""Gated fusion MLP: (s_c, s_s) -> accept/reject probability.

Engineered features (6 inputs):
    s_c             — cosine similarity, content
    s_s             — cosine similarity, speaker
    s_c * s_s       — product (both-must-match indicator)
    |s_c - s_s|     — disagreement
    s_c^2           — nonlinearity on content
    s_s^2           — nonlinearity on speaker

Architecture (hidden_dims = [20, 10] from FusionConfig):
    Linear(6, 20) + ReLU
    Linear(20, 10) + ReLU
    Linear(10, 1)  + Sigmoid

Param count: (6*20 + 20) + (20*10 + 10) + (10*1 + 1) = 140 + 210 + 11 = 361.

The doc occasionally refers to this as "the ~360-param fusion MLP". v1 said
"400 params" — that was wrong; the correct number is 361.
"""
import torch
from torch import nn

from solospeak.utils.config import FusionConfig


class GatedFusionMLP(nn.Module):
    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = 6
        for h in config.hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, s_c: torch.Tensor, s_s: torch.Tensor) -> torch.Tensor:
        features = torch.stack([
            s_c, s_s,
            s_c * s_s,
            (s_c - s_s).abs(),
            s_c ** 2,
            s_s ** 2,
        ], dim=-1)
        return self.net(features).squeeze(-1)
```

🧪 **Test:** `tests/unit/test_fusion.py::test_param_count`

```python
def test_fusion_param_count():
    from solospeak.utils.config import FusionConfig
    mlp = GatedFusionMLP(FusionConfig())
    n = sum(p.numel() for p in mlp.parameters())
    assert n == 361, f"Got {n}"
```

## 2.6 Silero-VAD Wrapper

`solospeak/models/vad.py`:

```python
"""Silero-VAD wrapper for streaming inference.

Wraps the Silero VAD v4 ONNX model (silero_vad_v4.onnx) so the streaming
pipeline can gate mel-extraction on speech activity.

Pinned model: a specific Silero ONNX asset from `snakers4/silero-vad`, shipped in the
OTA package as `silero_vad_v4.onnx`. Record the upstream URL, file size, SHA-256, and
input names in `artifacts/MANIFEST.json`. The wrapper below matches the current official
ONNX streaming signature (`input`, `state`, `sr`) and keeps the required 64-sample
context internally. If the pinned asset exposes a different signature (for example older
`h`/`c` recurrent-state inputs), update this wrapper and its tests before using it.
Ref: https://github.com/snakers4/silero-vad

The wrapper is intentionally thin: it exposes a single `is_speech(frame)` method.
The ONNX model expects 512-sample frames at 16 kHz (32 ms); the streaming ring
buffer (streaming.py) must supply exactly 512 new samples per call. The wrapper prepends
the 64-sample context required by the ONNX graph.
"""
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import ClassVar


class SileroVAD:
    """Thin wrapper around Silero VAD v4 ONNX model.

    Usage:
        vad = SileroVAD(Path("artifacts/silero_vad_v4.onnx"))
        speech = vad.is_speech(audio_frame_512_samples)
    """

    FRAME_SAMPLES: ClassVar[int] = 512        # required by the v4 model at 16 kHz
    CONTEXT_SAMPLES: ClassVar[int] = 64       # official ONNX wrapper context at 16 kHz
    SAMPLE_RATE: ClassVar[int] = 16000

    def __init__(self, onnx_path: Path, threshold: float = 0.5) -> None:
        self.threshold = threshold
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(str(onnx_path), sess_options=opts)
        input_names = {inp.name for inp in self._session.get_inputs()}
        expected = {"input", "state", "sr"}
        if not expected.issubset(input_names):
            raise ValueError(
                f"Unexpected Silero ONNX inputs {sorted(input_names)}; expected {sorted(expected)}. "
                "Pin the official state/context model or update SileroVAD."
            )
        # Stateful recurrent state + streaming context required across frames.
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT_SAMPLES), dtype=np.float32)

    def is_speech(self, frame: np.ndarray) -> bool:
        """Return True if the 512-sample frame contains speech.

        Args:
            frame: (512,) float32, normalised to [-1, 1].

        Returns:
            bool — True if VAD confidence ≥ self.threshold.
        """
        if frame.shape != (self.FRAME_SAMPLES,):
            raise ValueError(
                f"VAD expects exactly {self.FRAME_SAMPLES} samples, got {frame.shape}"
            )
        x = frame[None, :].astype(np.float32)          # (1, 512)
        x = np.concatenate([self._context, x], axis=1) # (1, 576)
        sr = np.array(self.SAMPLE_RATE, dtype=np.int64)
        out, self._state = self._session.run(
            None,
            {"input": x, "state": self._state, "sr": sr},
        )
        self._context = x[:, -self.CONTEXT_SAMPLES:]
        return bool(out.item() >= self.threshold)

    def reset_state(self) -> None:
        """Reset hidden state between independent audio streams."""
        self._state[:] = 0.0
        self._context[:] = 0.0
```

🧪 **Test:** `tests/unit/test_solospeak.py::test_silero_vad_wrapper`
- Skip (`@pytest.mark.onnx`) if `artifacts/silero_vad_v4.onnx` does not exist.
- Construct a 512-sample silent frame (all zeros) and a 512-sample sine-wave frame.
- Assert `is_speech` returns `False` on silence (confidence near 0).
- Assert `reset_state()` does not raise.

### 2.6.1 Acceptance criteria — Phase 2.6

- 🧪 `SileroVAD` imports without error even when the ONNX file is absent (import-time check only).
- 🧪 `SileroVAD.FRAME_SAMPLES == 512` and `SileroVAD.SAMPLE_RATE == 16000`.
- 🧪 `SileroVAD.CONTEXT_SAMPLES == 64`, and constructing the wrapper against the pinned
  ONNX file validates that the graph exposes `input`, `state`, and `sr`.
- 🧪 `is_speech` raises `ValueError` for wrong-length input.

---

## 2.7 Acceptance criteria — Phase 2

- 🧪 `SoloSpeakModel(default_config)(mel_batch)` runs without error.
- 🧪 Output shapes are `(B, 128)` for both `z_c` and `z_s`.
- 🧪 Output embeddings have unit L2 norm (`atol=1e-5`).
- 🧪 Backward pass works.
- 🧪 Total deployable param count for `bcresnet8`: 1,080,000 – 1,125,000.
- 🧪 `torch.onnx.export(model, dummy_mel, "out.onnx", opset_version=17)` succeeds for `dummy_mel` of shape `(1, 1, 80, 160)`.

---

# Phase 3: Training Pipeline

**Duration:** Days 15–32 (≈18 days; v1 said 14 — too tight given Kaggle session limits and the cost of a single failed run).
**Goal:** All 6 stages implemented and individually runnable; the full chain runs end-to-end producing a Stage-6 checkpoint.
**Exit criterion:** `make stage-6` produces `checkpoints/stage6_qat.pt` whose dev-set metrics meet the **MIN** thresholds in the per-stage gates below. The TARGET thresholds are aspirations for the final report, not advancement gates.

## 3.0 Gate philosophy (read first)

v1 used aggressive single-threshold gates that would block the project on a 0.1 pp shortfall. v2 splits each gate into:

- **MIN.** Must pass to advance to the next stage. The model is learning and improving; data and code are correct.
- **TARGET.** What the final report aims for. Failing the TARGET is a writeup concern, not a build-blocker.

If a stage hits MIN but not TARGET, advance and keep training in parallel; revisit later if compute remains.

## 3.1 Training Infrastructure

### 3.1.1 Stage base class

`solospeak/training/stages/base.py`:

```python
"""Abstract base for training stages.

Every stage implements:
    prepare_data()       -> set up data loaders
    build_model()        -> initialize or load model
    compute_loss()       -> stage-specific loss
    on_epoch_end()       -> eval, logging, save checkpoints
    go_no_go_check()     -> run gate criterion, returns (passed_min, passed_target)
    run()                -> full training loop; returns path to saved checkpoint
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import torch
from torch import nn
from torch.utils.data import DataLoader

from solospeak.utils.config import SoloSpeakConfig
from solospeak.utils.types import LossDict, MetricsDict, StageBatch


class TrainingStage(ABC):
    # ClassVar annotations are required for mypy strict — these are subclass-level
    # constants, not per-instance attributes.
    stage_id: ClassVar[int]
    stage_name: ClassVar[str]
    min_gate_metric: ClassVar[str]
    min_gate_threshold: ClassVar[float]
    target_gate_threshold: ClassVar[float]

    def __init__(self, config: SoloSpeakConfig) -> None:
        self.config = config
        self.step = 0

    @abstractmethod
    def prepare_data(self) -> tuple[DataLoader, ...]: ...
    @abstractmethod
    def build_model(self) -> nn.Module: ...
    @abstractmethod
    def compute_loss(self, batches: StageBatch, step: int) -> LossDict: ...
    @abstractmethod
    def on_epoch_end(self, epoch: int) -> MetricsDict: ...
    @abstractmethod
    def go_no_go_check(self, metrics: MetricsDict) -> tuple[bool, bool]:
        """Returns (passed_min, passed_target)."""
        ...

    @abstractmethod
    def run(self) -> Path:
        """Full training loop. Returns path to saved checkpoint.

        Responsibilities of each concrete stage implementation:
          1. Call prepare_data() and build_model().
          2. Iterate over epochs; build a StageBatch and call compute_loss() with grad clip.
          3. Call on_epoch_end() after each epoch; log via W&B callbacks.
          4. Write a checkpoint every config.training.checkpoint_every_steps steps.
          5. Call go_no_go_check() at end; raise RuntimeError on MIN-gate failure.
          6. Return the Path of the final saved checkpoint.
        """
        ...
```

**StageBatch convention.** Every concrete stage passes batches to `compute_loss()` as a
dict keyed by logical stream:

- Stage 1: `{"gsc": gsc_batch}`
- Stages 2–4: `{"content": content_batch, "speaker": speaker_batch}`
- Stage 5: `{"quadrant": quadrant_batch}`
- Stage 6: `{"distill": distill_batch}` and, if available, `{"quadrant": quadrant_batch}`

This avoids the old single-batch ambiguity that made the Stage 2 interleaved content /
speaker routing easy to implement incorrectly.

### 3.1.2 Trainer orchestrator

`solospeak/training/trainer.py` runs stages sequentially, supports `--resume-from`, and aborts on MIN-gate failure (not TARGET-gate failure).

### 3.1.3 Training-only wrapper

`solospeak/training/training_wrapper.py` owns every module that is needed during
training but must not appear in the deployed ONNX graph.

```python
"""Training wrapper around SoloSpeakModel.

SoloSpeakModel is the deployable model: backbone + content head + speaker head +
fusion MLP. This wrapper adds auxiliary word/speaker classifiers and adversarial
probe heads for Stages 2-4. Export code must load only wrapper.model.state_dict()
into a clean SoloSpeakModel.
"""
import torch
from torch import nn

from solospeak.losses.adversarial import AdversarialProbeHead, grad_reverse
from solospeak.models.heads import AuxiliaryHeads
from solospeak.models.solospeak import SoloSpeakModel
from solospeak.utils.config import SoloSpeakConfig


class TrainingWrapper(nn.Module):
    def __init__(self, config: SoloSpeakConfig) -> None:
        super().__init__()
        if config.training.n_aux_word_classes is None:
            raise ValueError("training.n_aux_word_classes must be injected from STATS.json")
        if config.training.n_aux_speaker_classes is None:
            raise ValueError("training.n_aux_speaker_classes must be injected from STATS.json")
        self.model = SoloSpeakModel(config)
        embed_dim = config.heads.content_dim
        n_words = config.training.n_aux_word_classes
        n_speakers = config.training.n_aux_speaker_classes
        self.aux_heads = AuxiliaryHeads(embed_dim, n_words, n_speakers)
        self.content_adversary = AdversarialProbeHead(embed_dim, n_speakers)
        self.speaker_adversary = AdversarialProbeHead(embed_dim, n_words)

    def forward(self, mel: torch.Tensor, adv_lambda: float = 0.0) -> dict[str, torch.Tensor]:
        z_c, z_s = self.model(mel)
        word_logits, speaker_logits = self.aux_heads(z_c, z_s)
        return {
            "z_c": z_c,
            "z_s": z_s,
            "word_logits": word_logits,
            "speaker_logits": speaker_logits,
            "speaker_from_content_logits": self.content_adversary(grad_reverse(z_c, adv_lambda)),
            "word_from_speaker_logits": self.speaker_adversary(grad_reverse(z_s, adv_lambda)),
        }

    def deployable_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only the clean SoloSpeakModel weights for Stage 5/6/export."""
        return self.model.state_dict()
```

**Checkpoint rule.** Stage 2-4 checkpoints store both `wrapper_state` and
`model_state`. `model_state` is the deployable `SoloSpeakModel` state from
`deployable_state_dict()`. Phase 5 export MUST ignore `wrapper_state`.

### 3.1.4 Checkpoint handoff

```
checkpoints/
├── stage1_backbone.pt
├── stage2_dualhead.pt
├── stage3_disentangle.pt
├── stage4_robust.pt
├── stage5_fusion.pt
├── stage6_qat.pt
└── latest.pt        # symlink
```

Each checkpoint stores `{model_state, optimizer_state, config_snapshot, step, metrics, stage_origin}`. Later stages assert that `stage_origin` matches the expected predecessor.

**Kaggle-session resilience.** Checkpoints are written every 500 steps to local session storage AND uploaded to a `solospeak-checkpoints` Kaggle dataset every 2,000 steps. `run_stage.py` automatically resumes from the latest available checkpoint on session start.

### 3.1.5 W&B integration

`solospeak/training/callbacks.py` logs:
- Every N steps: `train/loss_<component>` for each loss term, `train/lr`, `train/grad_norm`, `gpu/memory_allocated_mb`.
- Every epoch: `dev/ta_clean`, `dev/ta_noisy_avg`, `dev/fa_per_hour_per_user`, `dev/probe_speaker_on_content`, `dev/probe_word_on_speaker`.

### 3.1.6 Temporary TA protocol before fusion training

Stages 2–4 gate on task accuracy before the learned fusion MLP has been trained. To keep
those gates reproducible, they use a fixed proxy decision protocol:

1. Build simulated dev profiles from `dev_content.csv`: 5 enrollment utterances per
   `(speaker_id, keyword_text)` profile and held-out utterances for Q1/Q2/Q3/Q4 trials.
2. Compute `content_template` and `speaker_template` exactly as in 3.6.2.
3. For each trial, compute cosine scores `s_c` and `s_s`.
4. Use `proxy_score = min(s_c, s_s)` as the pre-fusion accept score.
5. Calibrate a per-profile `tau_proxy` on a held-out calibration split by choosing the
   lowest threshold that keeps estimated FA/hr/user ≤ 2.0.
6. Report `dev/ta_clean` and `dev/ta_noisy_avg` from this proxy. These metrics are used
   only for Stage 2–4 gates and mode-collapse guards; Stage 5 replaces the proxy with
   the learned `GatedFusionMLP`.

This protocol prevents an implementing agent from inventing a different temporary
decision rule for each stage.

## 3.2 Stage 1 — Backbone Pretraining

**Purpose.** Warm-start SoloSpeakResNet on supervised KWS before dual-head training. Prevents Stage 2 divergence.

**Data.** GSC v2, 35-class classification.
**Loss.** Cross-entropy.
**Architecture.** SoloSpeakResNet backbone + linear classifier (the classifier is discarded at end of stage; only the backbone weights are passed to Stage 2).
**Class count.** Load `TrainingConfig.n_gsc_classes` from
`data/manifests/STATS.json["n_gsc_classes"]`; this must be 35 for both smoke and full
GSC v2. Do not reuse `n_aux_word_classes` for the Stage-1 classifier.
**Duration estimate.** ~12 hours on A100, ~36 hours on T4. Budget 3 calendar days for retries.
**Hyperparameters.** batch=256, lr=3e-3 cosine-decayed, warmup=1000 steps, epochs=30, augmentation = `GainJitter` + `TimeShift` only (no noise yet).

**Gate.**
- **MIN:** GSC v2 dev accuracy ≥ 92%.
- **TARGET:** ≥ 96%.

**Expected curves.** epoch 3 ≥ 80%, epoch 10 ≥ 92%, epoch 20 ≥ 95%, epoch 30 ≥ 96%.

If MIN not hit: see Appendix E.1.

## 3.3 Stage 2 — Dual-Head Joint Training

**Purpose.** Train content and speaker heads jointly. Backbone fine-tunes from Stage 1.

**Data.**
- Content: `train_content.csv` (LibriPhrase).
- Speaker: `train_speaker.csv` (VoxCeleb2 or Common Voice fallback).
- **Interleaved batching.** Each step draws one ClassAwareBatch (size 128) from each
  loader. Both batches go through the encoder once each. SupCon-content and aux word CE
  run on the content batch only. SupCon-speaker and aux speaker CE run on the speaker
  batch only. **Total effective batch per step: 256.** The training loop passes
  `{"content": content_batch, "speaker": speaker_batch}` to `compute_loss()`; do not
  concatenate the two streams before loss routing.

**Loss.** `1.0 * L_supcon_c + 1.0 * L_supcon_s + 0.5 * L_aux_ce`.

**Class-aware sampler is mandatory** — random sampling produces empty positive sets for SupCon and silently kills the gradient on the affected anchors.

**Aux head class counts** are loaded at startup from `data/manifests/STATS.json` and
injected into `TrainingConfig.n_aux_word_classes` / `n_aux_speaker_classes`.
**Never hardcode** these. `n_aux_word_classes` is the LibriPhrase wake-phrase count, not
the GSC class count and not a transcript vocabulary for VoxCeleb/Common Voice.

**Duration estimate.** ~5 days on T4.
**Hyperparameters.** lr=1e-3 (lower than Stage 1, the backbone is pretrained); backbone LR = 0.1× head LR (discriminative LR).

**Gate.**
- **MIN:** Dev clean TA ≥ 90% using the proxy protocol in 3.1.6.
- **TARGET:** ≥ 96%.

## 3.4 Stage 3 — Disentanglement Turn-On

**Purpose.** Force `z_c` and `z_s` to be orthogonal (the core novelty).

**Data.** Same as Stage 2.
**Loss.**
```
L = 1.0 * L_supcon_c + 1.0 * L_supcon_s + 0.5 * L_aux_ce
  + lambda_ortho(step) * L_ortho
  + lambda_adv(step)   * L_adversarial

lambda_ortho(step) = 0.1 * min(1.0, step / 5000)
lambda_adv(step)   = 0.1 * min(1.0, step / 5000)
```

**Adversarial probe heads.** Two MLPs (256 hidden, see 2.3.3) trained behind
grad-reverse. Discarded at end of stage. Use the same valid-label routing as 2.3.4:
speaker-ID adversarial CE is computed on speaker-batch labels, and word adversarial CE is
computed only on labels where `keyword_label != -100`.

### 3.4.1 Disentanglement metric — relative reduction

`[DECISION]` v1's "probe accuracy ≤ 20%" gate was poorly defined (chance is 1/N_speakers, often much smaller than 20%). v2 uses **relative reduction** vs a Stage-2 baseline:

1. After Stage 2, train probes on z_c and z_s, record accuracies `acc_c_pre`, `acc_s_pre`.
2. After Stage 3, retrain probes (fresh weights, same protocol), record `acc_c_post`, `acc_s_post`.
3. Compute relative reduction:
   ```
   reduction_c = (acc_c_pre - acc_c_post) / acc_c_pre
   reduction_s = (acc_s_pre - acc_s_post) / acc_s_pre
   ```

**Probe protocol (fixed for reproducibility):**
- Architecture: `Linear(128, 256) → ReLU → Linear(256, n_classes)`.
- Optimizer: AdamW, lr=1e-3, 10 epochs.
- Train data: balanced sample of 50 utterances per class; max 200 classes (subsample if more).
- Eval data: held-out speaker-disjoint set.

**Gate.**
- **MIN:** `reduction_c ≥ 0.30` AND `reduction_s ≥ 0.30` (probe accuracy drops by at least 30% relative).
- **TARGET:** `reduction_c ≥ 0.60` AND `reduction_s ≥ 0.60`.

**Mode collapse guard.** Also assert `dev clean TA ≥ 0.95 * (Stage 2 TA)` using the
same proxy protocol from 3.1.6 — i.e. disentanglement did not destroy task accuracy by
more than 5%.

### 3.4.2 Failure modes

- **Probes don't drop:** see Appendix E.2.
- **Mode collapse (probes drop, but TA also drops):** reduce `lambda_adv` from 0.1 to 0.05; extend ramp to 10,000 steps.

## 3.5 Stage 4 — Robustness Training

**Purpose.** Handle noise (−5 to 30 dB SNR) and distance (0.5 to 5 m) without accuracy loss.

**Data.** Stage 3 data + heavy augmentation:
- `AddNoise` (p=0.9)
- `ConvolveRIR` (p=0.7)
- `CurriculumAugmenter` with `total_steps = total Stage 4 steps`.

**Loss.** Same as Stage 3.
**Duration estimate.** ~3 days.

**Gate.**
- **MIN:** Dev noisy macro-TA (mean over 8 SNR buckets) ≥ 80% using the proxy protocol
  from 3.1.6.
- **TARGET:** ≥ 88%.

### 3.5.1 Hard-negative refresh

Every epoch: refresh the hard-negative pairs based on the previous epoch's most-confused word/speaker pairs.

## 3.6 Stage 5 — Fusion MLP Training

**Purpose.** Train the 361-param fusion MLP to map `(s_c, s_s) -> P(accept)`.

### 3.6.1 Q1–Q4 dataset construction

`[DECISION]` Construct each of the four quadrants explicitly. Encoder is FROZEN during Stage 5 — only the fusion MLP trains.

For 20 simulated users built from LibriPhrase dev set:

- **Q1 (true positive):** `(user_keyword, user_voice)` — label 1.
  - 5 enrollment templates per user; 10 test utterances per user; cosine similarity to the user's mean template.
- **Q2 (imposter):** `(user_keyword, other_voice)` — label 0.
  - Same keyword spoken by a non-enrolled speaker; similarity computed against the user's templates.
- **Q3 (wrong word, right voice):** `(phonetic_neighbor, user_voice)` — label 0.
  - Phonetic neighbors generated via `g2p_en` (minimum phoneme edit distance ≤ 2).
    Prefer real utterances from the enrolled speaker. If real utterances do not exist,
    synthetic rows are allowed only under the TTS policy below.
- **Q4 (background):** `(other_keyword, other_voice)` — label 0.

**Target: 5,000 samples per quadrant. Class-balanced.** Achieved via the following explicit rules:

- **Q1** (20 users × 10 utterances = 200 direct samples): augment each sample 25× using the waveform augmenters from 1.4.1 (AddNoise, ConvolveRIR, GainJitter, TimeShift, PitchShift) to reach ~5,000 samples.
- **Q2** (same keyword, non-enrolled speaker): for each of the 20 enrolled users, draw 10 non-enrolled speakers from the LibriPhrase dev set × 10 utterances each = 2,000 samples; augment 2.5× to reach ~5,000.
- **Q3** (phonetic neighbor, enrolled voice): for each user expand across up to 5 phonetic neighbors × available utterances. If real utterances are insufficient, generate synthetic rows only under the TTS policy below. Augment until 5,000 is reached. Record the real/synth source per sample and set `q3_gate_eligible` according to the hard-gate rule below.
- **Q4** (other keyword, other speaker): draw freely from all non-enrolled keyword × non-enrolled speaker pairs in the dev set; subsample to exactly 5,000.

**Stage-5 split rule — no fusion-data leakage.** Build candidate quadrant examples from
LibriPhrase dev, then split by `profile_id` (not by individual augmented row) into:

- `stage5_train_quadrants.csv` — trains only the fusion MLP.
- `stage5_calibration_quadrants.csv` — early stopping and global/default-threshold
  sanity checks only; it must not provide profile-specific thresholds for the dev gate.
- `stage5_dev_quadrants.csv` — the only split used for the Stage-5 MIN/TARGET gate.

All augmentations and synthetic variants inherit the split of their source
`profile_id` / base row. Do not allow augmented siblings, TTS variants, or held-out
utterances from the same enrolled profile to cross between train/calibration/dev; that
would leak profile-specific similarity statistics into the gate.

**Threshold ownership rule.** Stage 5 trains the fusion MLP and may use
`stage5_calibration_quadrants.csv` for early stopping, model selection, or choosing a
single global fallback/default threshold. It does NOT persist per-profile `tau` values
for Stage-5 dev profiles. Per-profile thresholds are computed only when a profile is
created for enrollment/evaluation, using that profile's own held-out calibration
negatives as described in 3.6.2 and 4.1.2.

**Stage-5 dev threshold rule.** The Stage-5 quadrant gate uses one global threshold
`tau_stage5`, not per-profile thresholds. Compute `tau_stage5` by sweeping thresholds
from 0.01 to 0.99 on `stage5_calibration_quadrants.csv` and choosing the value that
maximizes macro-average quadrant accuracy over Q1/Q2/Q3/Q4 after excluding
`q3_gate_eligible=false` Q3 rows. Break ties by choosing the highest threshold, which is
the lower-false-accept option. Apply this single threshold unchanged to
`stage5_dev_quadrants.csv`. If the calibration split is unavailable in a smoke test, use
`tau_stage5 = 0.5` and mark the metric as smoke-only.

Full (non-smoke) calibration denominator rule: `stage5_calibration_quadrants.csv` must
contain at least 500 `Q3_wrong_word` rows with `q3_gate_eligible=true`, drawn from at
least 5 distinct `profile_id` values. If it does not, data construction fails before
threshold selection. Do not let the global threshold be chosen mostly by Q1/Q2/Q4.

**TTS policy for Q3.** Parler-TTS is a descriptive/speaker-prompted TTS model, not an
arbitrary zero-shot voice-cloning backend for enrolled users. Therefore:

- Real Q3 rows (`trial_source=real`) are the primary "wrong word, right voice" metric.
- A synthesized row may count as right-voice Q3 only if the synthesis backend actually
  conditions on the enrolled user's audio (for example a supported voice-cloning or
  fine-tuned TTS path) and the generated clip passes a speaker-similarity check against
  the user's speaker template.
- Parler-TTS-only rows are allowed as a disclosed stress test for wrong-word content
  confusion, but they must remain in the `q3_rejection_synth` bucket and must not be
  mixed into the hard right-voice Q3 gate unless the speaker-similarity check passes.

**Q3 hard-gate rule.** The Stage-5 Q3 MIN/TARGET gate and the headline
`q3_rejection` metric include only:
- real right-voice Q3 rows (`trial_source=real`), and
- synthetic Q3 rows whose synthesis backend conditioned on enrolled-user audio AND whose
  output passed the speaker-similarity verification.

Generic Parler-TTS rows that are descriptive-prompt-only are reported in
`q3_rejection_synth` as a stress bucket, but they are excluded from the hard right-voice
Q3 gate and from the headline `q3_rejection` numerator/denominator.

The manifest columns are the source of truth:
- `q3_gate_eligible=true` for real right-voice Q3 rows and for verified synthetic rows
  that count in the hard gate.
- `q3_gate_eligible=false` for generic TTS stress rows.
- `synthesis_backend` is blank for real rows and names the backend for synthetic rows.
- `speaker_verification_score` is blank for real rows and stores the cosine similarity
  used to verify synthetic right-voice rows.

Minimum denominator rule for full (non-smoke) validation:
`stage5_dev_quadrants.csv` must contain at least 500 `Q3_wrong_word` rows with
`q3_gate_eligible=true` after augmentation, drawn from at least 5 distinct `profile_id`
values. `test_kpi.csv` must contain at least 100 eligible Q3 rows. If either denominator
is smaller, data construction fails and the gate is NO-GO; do not silently pass Q3 on an
empty or stress-only subset. Smoke manifests may use the tiny counts from 1.2.5 and must
mark any Q3 metric as smoke-only.

**Implementation dependency.** Stage 5 may use `solospeak/enrollment/tts_augmentation.py`
to synthesize Q3 neighbors when real utterances are unavailable, subject to the TTS
policy above. Create the minimal TTS helper by Level 4; the reusable enrollment template
and calibration helpers also land by Level 4 so Stage 5 does not duplicate template math.

The generated Stage-5 quadrant manifest uses the canonical schema from 1.2.1. In
particular, `speaker_id` / `keyword_text` describe the trial audio, while
`enrolled_user_id`, `enrolled_keyword_text`, `profile_id`, and `profile_path` describe
the enrollment target. Q3 synthesized rows must set `trial_source=tts`; real Q3 rows set
`trial_source=real`. The `q3_gate_eligible`, `synthesis_backend`, and
`speaker_verification_score` columns decide whether each row belongs to the hard
right-voice Q3 gate or only to the disclosed synthetic stress bucket.

**Loss.** Binary cross-entropy on fusion output.
**Duration.** ~12 hours (the fusion MLP is tiny).

**Gate.**
- **MIN:** All four hard-gate quadrant accuracies ≥ 85% on
  `stage5_dev_quadrants.csv` using the single global `tau_stage5`; Q3 uses the
  `q3_gate_eligible=true` subset above, while generic Parler-TTS stress rows are
  reported separately.
- **TARGET:** Q1 ≥ 96%, Q2/Q3/Q4 ≥ 95%.

### 3.6.2 Per-user threshold calibration

The fusion threshold τ is NOT learned as part of the fusion MLP. Stage 5 may report a
global fallback/default threshold for sanity checks, but per-user calibration happens
when an enrollment/eval profile is created (Phase 5 / KPI evaluation). The Stage-5
calibration split must not overwrite the per-profile calibration procedure below.

**Enrollment/template math.**
1. For each accepted enrollment recording, compute `(z_c_i, z_s_i)`.
2. `content_template = normalize(mean_i(z_c_i))`.
3. `speaker_template = normalize(mean_i(z_s_i))`.
4. At inference, compute `s_c = cosine(z_c, content_template)` and
   `s_s = cosine(z_s, speaker_template)`.
5. `fusion_score = GatedFusionMLP(s_c, s_s)`.
6. Calibrate per-user `tau` from held-out enrollment negatives by choosing the lowest
   threshold that keeps estimated FA/hr/user ≤ 2.0 while maximizing held-out Q1 accepts.
   If no held-out negatives are available, use `FusionConfig.tau_on` as the conservative
   default.

## 3.7 Stage 6 — Quantization-Aware Fine-Tuning

**Purpose.** Fine-tune the model so the final INT8 export matches FP32 within tolerance.

**Data.** Stage-4-style augmented waveform windows for embedding stability plus the
Stage-5 quadrant/calibration rows when fusion-score supervision is available.
**Loss.** Clean-model distillation against the Stage-5 FP32 teacher:
- embedding similarity distillation on `z_c` and `z_s` against a fixed reference bank;
- optional BCE on `fusion_score` for rows with `quadrant_label`;
- optional MSE/KL on teacher vs student `(content_score, speaker_score, fusion_score)`.

Stage 6 does **not** reuse Stage-4 aux/adversarial CE. The QAT-prepared module is a clean
`SoloSpeakModel`, so training-only auxiliary and adversarial heads are absent by design.
**Duration.** ~12 hours.

**Gate.**
- **MIN:** Temporary ONNX Runtime INT8 vs FP32 TA degradation < 1.0 pp on dev set.
- **TARGET:** < 0.3 pp.

`[DECISION] QAT vs PTQ.` v1 mixed PyTorch QAT and ONNX Runtime static PTQ. v2 picks ONE primary path:

**Primary path: PyTorch fake-quant QAT fine-tune → save QAT-aware FP32 checkpoint →
fused FP32 ONNX export → ONNX Runtime static PTQ over the QAT-aware weights.**

The PyTorch QAT step adapts the FP32 weights to be quantization-friendly using fake-quant
observers. The production INT8 model is NOT the converted PyTorch quantized module. The
actual INT8 quantization is performed by ONNX Runtime's static quantizer (per-channel for
weights, per-tensor for activations). This combines the accuracy benefit of QAT with the
production-friendly tooling of ONNX RT.

```python
# Stage 6: short fake-quant QAT fine-tune on the clean deployable model
model_fp32.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
model_prepared = torch.ao.quantization.prepare_qat(model_fp32, inplace=False)
# ... ~5 epochs of distillation/fusion fine-tuning ...
# Copy the trained float weights into a clean, non-prepared SoloSpeakModel with the
# same architecture. This strips FakeQuant/Observer modules while preserving the
# quantization-adapted FP32 weights. Do NOT call convert() for the checkpoint that
# Phase 5 exports.
model_export_fp32 = SoloSpeakModel(config)
copy_float_weights_from_prepared(model_prepared, model_export_fp32)
torch.save({
    "model_state": model_export_fp32.state_dict(),
    "qat_trained": True,
    "contains_fake_quant_modules": False,
    "converted_int8_pytorch": False,
}, "checkpoints/stage6_qat.pt")
```

`copy_float_weights_from_prepared()` copies parameters/buffers whose names correspond to
the original clean modules and skips `activation_post_process`, `fake_quant`, and observer
state. Unit test it by asserting the clean model exports to ONNX and has no modules whose
class name contains `FakeQuant` or `Observer`.

**`copy_float_weights_from_prepared` contract** — lives in `solospeak/training/stages/stage6_qat.py`:

```python
def copy_float_weights_from_prepared(
    prepared: nn.Module,
    target: nn.Module,
) -> None:
    """Copy float weights from a QAT-prepared model into a clean target model.

    Iterates `target.named_parameters()` and `target.named_buffers()`. For each
    name, looks up the matching parameter/buffer in `prepared` (same dotted path)
    and copies its data. Names that include 'activation_post_process', 'fake_quant',
    or 'observer' in any path component are skipped — they do not exist in `target`.

    Raises ValueError if a name present in `target` is missing from `prepared`
    (which indicates an architecture mismatch — fix the model, not this function).
    """
    target_params = dict(target.named_parameters())
    target_bufs   = dict(target.named_buffers())
    src_params = dict(prepared.named_parameters())
    src_bufs   = dict(prepared.named_buffers())

    for name, param in target_params.items():
        if name not in src_params:
            raise ValueError(f"Prepared model missing parameter: {name!r}")
        param.data.copy_(src_params[name].data)

    for name, buf in target_bufs.items():
        if name not in src_bufs:
            raise ValueError(f"Prepared model missing buffer: {name!r}")
        buf.data.copy_(src_bufs[name].data)
```

🧪 **Test:** `tests/unit/test_deployment.py::test_copy_float_weights_from_prepared`
- Prepare a tiny `SoloSpeakModel` with `prepare_qat`, mutate its float weights, call
  `copy_float_weights_from_prepared`, assert the target weights match the mutated values.
- Assert the target has no modules whose `type(m).__name__` contains `"FakeQuant"` or `"Observer"`.

**Stage-6 gate implementation.** To measure the gate, Stage 6 runs a temporary
export-and-quantize check into `artifacts/tmp_stage6_int8.onnx` using the same calibration
settings as Phase 5, also exports `artifacts/tmp_stage6_fp32.onnx`, and calls the same
INT8-vs-FP32 degradation checker with both paths. It records the degradation in the
checkpoint metrics, then deletes or overwrites both temporary artifacts. Phase 5 still
owns the final production export.

## 3.8 Stage Runner Script

`scripts/run_stage.py`:

```python
"""CLI entry point for any single training stage.

Usage:
    python -m scripts.run_stage --stage 1 --config configs/training/stage1_backbone_pretrain.yaml
    python -m scripts.run_stage --stage 3 --resume-from checkpoints/stage2_dualhead.pt
    python -m scripts.run_stage --all
"""
```

`--all` runs stages 1→6 sequentially, aborting on MIN-gate failure but NOT on TARGET-gate failure.

### 3.8.1 Acceptance criteria — Phase 3

- 🧪 `make stage-1` completes; GSC dev acc ≥ MIN(92%).
- 🧪 Each subsequent stage loads its predecessor's checkpoint successfully.
- 🧪 Stage 3 disentanglement reduction ≥ MIN(0.30) on both probes.
- 🧪 Stage 4 noisy macro-TA ≥ MIN(80%).
- 🧪 Stage 5 per-quadrant accuracy ≥ MIN(85%).
- 🧪 Stage 6 temporary ORT INT8 vs FP32 degradation < MIN(1.0 pp).

---

# Phase 4: Evaluation Infrastructure

**Duration:** Days 33–37.
**Goal:** Full KPI suite, subgroup eval, probe verification, and ablation tooling are implemented and reproducible.
**Exit criterion:** `make eval` runs in ≤ 1 hour on a single GPU and produces
`reports/kpi_final.json`, `reports/probes.json`, and `reports/subgroup_report.md`.
`make ablation` is a separate long-running reporting command; it produces
`reports/ablation_table.md` but is not part of the one-hour eval contract.

## 4.1 Full KPI Suite

### 4.1.1 `solospeak/eval/kpi_suite.py`

```python
"""Full Samsung KPI evaluation.

One call, one JSON output:
    ta_clean                 float
    ta_noisy                 dict[int, float]   # per-SNR
    ta_noisy_macro           float              # mean across SNRs
    fa_per_hour_per_user     float
    fa_per_hour_device       dict[int, float]   # N enrolled users -> fa/hr; N ∈ {1, 4, 8}
    q2_rejection             float
    q3_rejection             float              # real + speaker-verified synth weighted average
    q3_rejection_real        float              # real utterances only
    q3_rejection_synth       float              # all synthesized Q3 stress rows, separate from hard gate
    q4_rejection             float
    distance_ta              dict[float, float]
    param_count              int
    xrt_fp32                 float
    xrt_int8                 float
    per_demographic          SubgroupDict           # subgroup -> metrics, or null if unavailable
"""
```

### 4.1.2 Measurement protocols

**TA Clean.** ~50 (user, keyword) pairs from `test_kpi.csv` (Q1 rows ≈ 485 total; see manifest example in 1.2.5); 10 utterances per pair ≈ 485 Q1 evaluation samples. Per-user threshold τ calibrated on a held-out rejection sample. Report mean ± std over 3 seeds.

**Note on sample count.** `test_kpi.csv` has ~1,940 rows (485 per quadrant), covering ~50 keywords × ~98 speakers. "500 pairs" referred to Q1+Q2+Q3+Q4 combined, not 500 Q1 pairs. The correct interpretation is ~50 enrolled (user, keyword) pairs × ~10 utterances each = ~485 Q1 samples.

**TA Noisy.** Same ~50 enrolled (user, keyword) pairs (~485 Q1 samples, as clarified above), mixed with MUSAN babble/traffic/music at SNRs `{-5, 0, 5, 10, 15, 20, 25, 30}` dB. Report per-SNR and macro-mean.

**FA per hour.** v1 was ambiguous about per-user vs device. v2 reports both:
- `fa_per_hour_per_user = total_wakes_for_user / 10.0` for a single-enrolled-user setup.
- `fa_per_hour_device = sum_over_enrolled_users(fa_per_hour_per_user_i)` — what the device experiences with N enrolled users in parallel-listening mode. Report for N ∈ {1, 4, 8}.

**Q2 Rejection.** Test utterances where the correct keyword is spoken by a non-enrolled speaker. Rejection rate = fraction correctly rejected.

**Q3 Rejection.** Test utterances where a phonetically similar but different keyword is
spoken by the enrolled user. The manifest columns are the source of truth:
- `q3_rejection_real`: rows with `quadrant_class == "Q3_wrong_word"` and
  `trial_source == "real"`.
- `q3_rejection_synth`: rows with `quadrant_class == "Q3_wrong_word"` and
  `trial_source == "tts"`, reported as a separate synthetic stress bucket.
- headline `q3_rejection`: rows with `quadrant_class == "Q3_wrong_word"` and
  `q3_gate_eligible == true`.

Do not recompute speaker verification during KPI aggregation. The profile/template
builder owns verification and records the result in `q3_gate_eligible` and
`speaker_verification_score`.

**Q4 Rejection.** Other keyword by other speaker. Rejection rate.

## 4.2 Probe Verification (Disentanglement Proof)

`solospeak/eval/probes.py` runs the protocol from 3.4.1 on the final model and reports:
- `acc_c_post` — speaker probe on `z_c`
- `acc_s_post` — word probe on `z_s`
- `reduction_c`, `reduction_s` (vs Stage 2 baseline, also recomputed here).

## 4.3 Ablation Runner

### 4.3.1 The seven ablation configurations

| # | Config | What's Ablated | Purpose |
|---|---|---|---|
| 1 | `full` | Nothing (baseline) | Reference |
| 2 | `no_ortho` | Orthogonality loss off | Novelty 1 contribution |
| 3 | `no_adv` | Adversarial probes off | Ortho-only vs ortho+adv |
| 4 | `no_disent` | Both 2 and 3 | Full disent contribution |
| 5 | `no_tts_enroll` | No TTS aug at enrollment | Novelty 3 |
| 6 | `no_gated_fusion` | `min(s_c, s_s)` instead of MLP | Novelty 2 |
| 7 | `no_curriculum` | Uniform SNR sampling | Novelty 4 |

### 4.3.2 Compute strategy — be honest about cost

`[DECISION]` v1 said "21 runs (7 × 3 seeds)" requires 42 GPU-days. That is impossible at hackathon scale. v2's plan:

- **Default:** 7 × 1 seed = 7 ablation runs, each forking from the same Stage 2 checkpoint and re-running Stages 3–5. Each takes ~2 days of GPU time. Sequential on one Kaggle T4: 14 days. With two parallel notebooks: ~7 days.
- **If schedule allows:** add 2 additional seeds for the `full` and `no_disent` configs only — these are the comparisons that matter most for the novelty claim. Total compute: 7 + 4 = 11 runs.
- **Variance disclosure:** the report MUST state "single-seed ablations" wherever multi-seed is not done. No silent variance-hiding.

### 4.3.3 Output: `reports/ablation_table.md`

Markdown table, one row per ablation, columns: clean TA, noisy macro TA, FA/hr/user, Δ vs full. Include a "seeds" column documenting how many seeds backed the row.

**Command separation.** `make eval` MUST NOT launch ablation training. `make ablation`
is the only command that runs `scripts/run_ablation.py`, and it is allowed to take days.
The final report should include `reports/ablation_table.md` when available; if compute
runs out, report the completed subset with the missing rows marked `not_run`.

## 4.4 Subgroup Evaluation

`[DECISION]` v1 required gender / age / accent stratification. Several of these depend on metadata that may be missing or noisy (especially in the Common Voice fallback). v2 marks subgroup eval as **OPTIONAL but recommended**:

- Always run: `keyword_syllable_count` (2 / 3 / 4+).
- Run if VoxCeleb metadata available: `gender` (M/F).
- Run if available with explicit data-card disclosure: `age_bucket`, `accent_bucket`.

Missing subgroups are reported as `null` in `reports/kpi_final.json` and noted in `docs/fairness_report.md`. **Do not invent metadata.**

## 4.5 xRT Benchmark

`solospeak/eval/xrt.py`:

```python
"""Real-time factor benchmark.

xRT = inference_time_per_window / window_duration

Window = 1.6 s (160 frames at 10 ms hop).

Targets:
    HARD GATE (deployment validation):  xrt_int8 < 0.20
    STRETCH:                            xrt_int8 < 0.08

Measurement environments (any one is sufficient for the HARD gate):
    1. ARM Cortex-A78 via Android device + ONNX Runtime (preferred)
    2. Apple M-series CPU (ARM, single-thread)  — ships in the demo notebook
    3. Raspberry Pi 4 (Cortex-A72)               — fallback
    4. x86 with 2.0x correction factor           — last resort

Report p50, p95, p99 over 100 windows.
"""
```

The HARD validation gate in Phase 5.3 is `xrt_int8 < 0.20`. The STRETCH target of 0.08 is reported but does not block.

### 4.5.1 Acceptance criteria — Phase 4

- 🧪 `make eval` finishes in ≤ 1 hour on a single GPU and does not launch ablation training.
- 🧪 `reports/kpi_final.json` has all required fields.
- 🧪 `reports/probes.json` and `reports/subgroup_report.md` are written.
- 🧪 `make ablation` writes `reports/ablation_table.md` with 7 rows when all configured ablations finish. This is report-critical but not part of the `make eval` runtime gate.
- 🧪 All KPIs meet the relevant MIN gates from Phase 3 and `ValidationGates` on the test set.
- 🧪 Probe reduction ≥ MIN(0.30).

---

# Phase 5: Deployment Pipeline

**Duration:** Days 38–41.
**Goal:** Validated INT8 ONNX artifact passing all gates; live demo runs on real audio.
**Exit criterion:** `make export` produces `artifacts/solospeak_int8.onnx` (size ≤ 5 MB) that passes all 9 required validation gates plus Gate 10 (INT8 vs FP32 degradation, using the freshly exported `artifacts/solospeak_fp32.onnx` as `fp32_onnx_path`); the CLI demo emits a wake event within 250 ms after the decisive window is available.

## 5.1 ONNX Export

### 5.1.1 `solospeak/deployment/export_onnx.py`

```python
"""PyTorch -> FP32 ONNX export.

Inputs:
    checkpoint: stage6_qat.pt
    input shape: (1, 1, 80, 160)   — fixed, NO dynamic time axis

Outputs:
    artifacts/solospeak_fp32.onnx   (~4-5 MB, varies with variant)

Stripping:
    - Auxiliary classification heads removed
    - Adversarial probe heads removed
    - GradientReversal layers removed
    - BatchNorm fused into preceding Conv via the model's fuse_model() method
    - The fusion MLP is exported in the same .onnx file as an optional score path

Why fixed time dim:
    - Simplifies ONNX Runtime memory layout
    - Required for INT8 quantization stability
    - Matches the streaming inference loop's 1.6s window exactly
"""
```

Opset 17. Dynamic axes: batch dimension only.

Export uses a small `ExportWrapper` around `SoloSpeakModel`: it calls
`model.forward(mel)` to get `(z_c, z_s)`, computes cosine scores against the provided
templates, then calls `model.forward_fusion(s_c, s_s)`. The training-time
`SoloSpeakModel.forward()` contract remains unchanged.

`ExportWrapper` is defined inside `solospeak/deployment/export_onnx.py` (not in
`solospeak/models/`) and is instantiated only during the export step. It is never saved
as a checkpoint:

```python
# solospeak/deployment/export_onnx.py — ExportWrapper (defined at module level)
import torch
import torch.nn.functional as F
from torch import nn

from solospeak.models.solospeak import SoloSpeakModel


class ExportWrapper(nn.Module):
    """Wraps SoloSpeakModel for ONNX export.

    Accepts three inputs (mel, content_template, speaker_template), runs the
    encoder, computes safe cosine scores, and returns all five outputs.
    Never instantiated outside of export_onnx.py.
    """

    def __init__(self, model: SoloSpeakModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        mel: torch.Tensor,               # (B, 1, 80, 160)
        content_template: torch.Tensor,  # (B, 128) L2-normalized
        speaker_template: torch.Tensor,  # (B, 128) L2-normalized
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_c, z_s = self.model(mel)
        # Safe cosine: clamp denominator so zero-template callers don't produce NaN.
        ct = content_template / content_template.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        st = speaker_template / speaker_template.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        s_c = (z_c * ct).sum(dim=-1)           # (B,)
        s_s = (z_s * st).sum(dim=-1)           # (B,)
        fusion_score = self.model.forward_fusion(s_c, s_s)  # (B,)
        return z_c, z_s, s_c, s_s, fusion_score
```

🧪 **Test:** `tests/integration/test_onnx_roundtrip.py` must verify that the ONNX graph
exported from `ExportWrapper` has exactly the five output names listed below and that a
zero-template forward pass does not produce `NaN` in any output.

**ONNX I/O contract.**
- Inputs:
  - `mel`: `(B, 1, 80, 160)`
  - `content_template`: `(B, 128)` L2-normalized enrollment template
  - `speaker_template`: `(B, 128)` L2-normalized enrollment template
- Outputs:
  - `z_c`, `z_s`: `(B, 128)` L2-normalized embeddings
  - `content_score`, `speaker_score`, `fusion_score`: `(B,)`

Score computation uses safe cosine normalization with denominator clamped at `1e-8`, so
for pure embedding extraction callers may pass zero templates and ignore the score
outputs. The deployed artifact remains a single `solospeak_int8.onnx` file.

**BatchNorm fusion contract.** Every block class exposes `fuse_model()` and performs
fusion locally, for example:

```python
torch.ao.quantization.fuse_modules(self.stem, [["0", "1", "2"]], inplace=True)
torch.ao.quantization.fuse_modules(block, [["conv1", "bn1"], ["conv2", "bn2"]], inplace=True)
torch.ao.quantization.fuse_modules(block.shortcut, [["0", "1"]], inplace=True)
```

Do not use a top-level example like `[['conv1', 'bn1']]`; those names do not exist on the
assembled model.

### 5.1.2 Op compatibility

Pre-checked against opset 17:
- `F.normalize`: supported.
- `GELU`: supported.
- `AdaptiveAvgPool2d` with static `output_size`: supported.

If export fails for any other reason, see Appendix E.5.

## 5.2 INT8 Quantization

### 5.2.1 `solospeak/deployment/quantize.py`

```python
"""Static post-training quantization via ONNX Runtime.

Inputs:
    solospeak_fp32.onnx
    Calibration data: 500 resolved dev trials from Stage-5/KPI manifests

Outputs:
    artifacts/solospeak_int8.onnx (~1.2-2.5 MB, varies with variant)
    reports/quantization_report.json (per-layer scales, activation histograms)

Settings:
    - Per-channel quantization for Conv weights
    - Per-tensor for activations
    - Primary setting quantizes Conv nodes only:
        op_types_to_quantize=["Conv"]
      This keeps the embedding heads and fusion MLP in FP32 while quantizing the
      latency/size-dominant backbone convolutions.
    - Fusion MLP is never quantized (too small, preserves precision)
"""
```

**Quantization exclusion contract.** The default implementation must call ONNX Runtime
static quantization with `op_types_to_quantize=["Conv"]`; do not include `"Gemm"` or
`"MatMul"` in the default quantized op list. If a later experiment quantizes linear
layers too, the exporter must first assign stable ONNX node-name prefixes:

```text
backbone/...
content_head/...
speaker_head/...
fusion_mlp/...
score_path/...
```

Then `quantize.py` must pass every `fusion_mlp/...` and `score_path/...` node in
`nodes_to_exclude`, and `reports/quantization_report.json` must list the excluded nodes.
Do not rely on exporter-generated anonymous node names such as `/net/net.0/Gemm`; those
are not a stable API.

**Calibration I/O contract.** The exported ONNX graph has three inputs, so calibration
samples are dictionaries, not bare mel arrays:

```python
{
    "mel": np.ndarray,               # (1, 1, 80, 160), float32
    "content_template": np.ndarray,  # (1, 128), float32, L2-normalized
    "speaker_template": np.ndarray,  # (1, 128), float32, L2-normalized
}
```

Use real profile templates from `profile_path` whenever available. For embedding-only
calibration rows without profiles, pass zero templates and mark the row as such in the
quantization report. The ORT calibration reader must yield these full input dictionaries;
do not pass `Iterable[np.ndarray]` unless the exported graph has been changed back to a
single-input embedding-only graph.

### 5.2.2 Accuracy verification after quantization

Run the Phase 4 KPI suite on the INT8 model. Report TA degradation vs FP32. If > 1.0 pp, return to Stage 6 for more QAT (see Appendix E.6).

## 5.3 Artifact Validation Gates

`solospeak/deployment/validate_artifact.py`:

```python
def validate(onnx_path: Path, eval_set: EvalSet,
             config: DeploymentConfig,
             gates: ValidationGates,
             fp32_onnx_path: Path | None = None) -> ValidationReport:
    """Run validation gates on the INT8 artifact.

    Gates 1–9 are always checked. Gate 10 (INT8 vs FP32 TA degradation) fires when
    fp32_onnx_path is provided. Gate 10 is optional only for quick local/debug runs;
    final Phase-5 release validation and Stage-6 temporary dev-set checks MUST pass it.
    """
    results = []
    results.append(check_filesize(onnx_path, max_mb=gates.max_filesize_mb))
    results.append(check_onnx_opset(onnx_path, min_opset=gates.min_opset))
    results.append(check_mobile_ops(onnx_path))
    results.append(check_xrt(onnx_path, max_xrt=gates.max_xrt))  # HARD gate, not 0.08
    results.append(check_ta_clean(onnx_path, eval_set, min_ta=gates.min_ta_clean))
    results.append(check_ta_noisy(onnx_path, eval_set, min_ta=gates.min_ta_noisy_macro))
    results.append(check_fa_rate(
        onnx_path, eval_set, max_fa_per_hr_per_user=gates.max_fa_per_hr_per_user,
    ))
    results.append(check_param_count(onnx_path, max_params=gates.max_param_count))
    results.append(check_output_shapes(onnx_path))
    # Gate 10: required for final release validation; optional only for quick/debug runs.
    if fp32_onnx_path is not None:
        results.append(check_int8_vs_fp32_degradation(
            onnx_path, fp32_onnx_path, eval_set,
            max_degradation_pp=gates.max_int8_vs_fp32_degradation_pp,
        ))

    report = ValidationReport(results)
    if not report.all_passed:
        raise ArtifactValidationError(report)
    return report
```

`scripts/export_and_validate.py` MUST load `ValidationGates` through
`load_eval_config_bundle()` and pass the resulting object into `validate()`. For the
final artifact produced by `make export`, it MUST call
`validate(..., gates=gates, fp32_onnx_path=fp32_path)`. Calling `validate()` without
`fp32_onnx_path` is reserved for fast local checks while iterating on gates 1–9.

`[DECISION]` v1 used aspirational thresholds (0.99 clean TA, 0.08 xRT, 1.0 FA/hr) as HARD blockers. That guarantees the artifact will fail. v2 uses MIN gates for validation and tracks TARGET/STRETCH separately in the report. The TARGET numbers go in the report; the HARD gates here decide whether the bits are safe to flash.

## 5.4 OTA Package Builder

`solospeak/deployment/ota_package.py` produces (signing is a Samsung-internal step, hackathon ships unsigned):

```
solospeak_ota_v1.0.0.zip
├── MANIFEST.json
├── solospeak_int8.onnx
├── silero_vad_v4.onnx
└── README.txt
```

## 5.5 Live Demo — CLI

`demo/cli/live_demo.py`:

```python
"""Live wake-word demo via laptop microphone.

Pipeline:
    sounddevice InputStream (16 kHz mono)
    -> ring buffer (1.6 s context, 160 ms hop / 90% overlap)
    -> Silero-VAD gate
    -> LogMelExtractorDeploy
    -> ONNX Runtime inference (INT8)
    -> cosine + fusion vs enrolled templates
    -> hysteresis (tau_on / tau_off + 250 ms refractory)
    -> emit WakeEvent

Modes:
    --enroll   interactive enrollment
    --listen   continuous listening
    --eval     run eval set, show confusion matrix
"""
```

UX: visible VAD meter + per-frame `(s_c, s_s, fusion)` scores so judges see the model thinking.

**Latency definition.** The model needs 1.6 s of context, so "wake latency" is measured
from the moment the decisive audio window is available, not from the first phoneme of the
keyword. With a 160 ms hop, the algorithmic cadence is ≤ 160 ms plus feature/inference time;
the acceptance threshold is ≤ 250 ms for this post-window latency.

## 5.6 Android Demo (Stretch)

`🟡 STRETCH` — defer to post-submission if schedule slips. CLI demo is sufficient.

### 5.6.1 Acceptance criteria — Phase 5

- 🧪 `make export` produces `artifacts/solospeak_int8.onnx` ≤ 5 MB.
- 🧪 All 9 required validation gates pass, and final release validation also passes Gate 10 by supplying `artifacts/solospeak_fp32.onnx` as `fp32_onnx_path`.
- 🧪 `python demo/cli/live_demo.py --enroll --user pranav --keyword "hey prism"` succeeds.
- 🧪 Live demo triggers on real voice within 250 ms post-window latency.

---

# Phase 6: Production Hardening

**Duration:** Days 42–44 (overlaps Phase 7 — production-readiness items that judges will see in the report).
**Goal:** Every section in this phase has either (a) a working stub the judge can run or (b) an honest "planned post-hackathon" doc with design rationale. Most items are `🟡 STRETCH` or doc-only.
**Exit criterion:** All files in `solospeak/security/` and `solospeak/observability/` exist with at least one runnable function each, and `docs/` contains the seven required documents listed in 6.5.

## 6.1 Security

### 6.1.1 Threat model — `docs/threat_model.md`

A hand-authored markdown document covering, at minimum:

- Replay attacks (recorded user voice played back through speaker).
- Imposter attacks (voice cloning, deepfake, mimicry).
- Adversarial audio (FGSM-style perturbations on mel input).
- Profile theft (template extraction from device storage).
- Multi-user confusion (one user's keyword matched against another user's profile).

For each threat: attack vector, current mitigation in v1.0, planned mitigation, residual risk.

### 6.1.2 Replay attack baseline — `solospeak/security/replay_eval.py`

```python
"""Replay attack evaluation.

Procedure:
    1. Record the user enrolling (5 utterances).
    2. Play those 5 recordings through a Bluetooth speaker at 0.5/1/2/3 m.
    3. Re-record with the laptop mic.
    4. Run inference. Report: how many replays trigger wake?

For v1.0 we report the baseline number with NO anti-spoofing. Anti-spoofing
head is planned for Q3 2026 (post-hackathon).
"""
```

`🟡 STRETCH` — record once, report a single number. The aim is to show the metric exists, not to drive it down.

### 6.1.3 Adversarial audio baseline — `solospeak/security/adversarial_eval.py`

```python
"""FGSM attack on mel spectrogram.

For each test utterance:
    1. Compute gradient of fusion output w.r.t. mel input.
    2. Perturb mel by epsilon * sign(gradient).
    3. Find minimum epsilon that flips the decision.

Report: median epsilon across the test set as the "adversarial robustness budget."
"""
```

`🟡 STRETCH` — single number for the report.

### 6.1.4 Threat model code stub — `solospeak/security/threat_model.py`

Just a module-level docstring listing the threats and a function `documented_threats() -> list[str]` that returns them. Useful for runtime introspection ("what threats does this build defend against?") but otherwise informational.

## 6.2 Observability

### 6.2.1 Local metrics struct — `solospeak/observability/metrics.py`

```python
"""On-device metric collection. No data leaves the device by default."""
from dataclasses import dataclass, field

@dataclass
class SoloSpeakLocalMetrics:
    wake_events_total: int = 0
    wake_events_per_user: dict[str, int] = field(default_factory=dict)
    inference_latency_ms_p50: float = 0.0
    inference_latency_ms_p99: float = 0.0
    fa_events_estimated_per_hr: float = 0.0
    enrollment_attempts: int = 0
    enrollment_failures: int = 0
    model_version: str = "solospeak-v1.0.0"

    def to_dict(self) -> dict[str, int | float | str | dict[str, int]]:
        return {
            "wake_events_total": self.wake_events_total,
            "wake_events_per_user": self.wake_events_per_user,
            "inference_latency_ms_p50": self.inference_latency_ms_p50,
            "inference_latency_ms_p99": self.inference_latency_ms_p99,
            "fa_events_estimated_per_hr": self.fa_events_estimated_per_hr,
            "enrollment_attempts": self.enrollment_attempts,
            "enrollment_failures": self.enrollment_failures,
            "model_version": self.model_version,
        }
```

The CLI demo prints these every 60 s.

### 6.2.2 Differential privacy — `solospeak/observability/dp_noise.py`

```python
"""Laplace mechanism. Used IF telemetry is ever uploaded.

In v1.0 no telemetry is uploaded. This module documents the planned mechanism.
"""
import numpy as np


def add_dp_noise(value: float, sensitivity: float, epsilon: float) -> float:
    """Add Laplace(0, sensitivity/epsilon) noise to a scalar metric."""
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    scale = sensitivity / epsilon
    return float(value + np.random.laplace(0.0, scale))
```

🧪 **Test:** Statistical test on 10000 samples that the empirical variance matches `2 * (sensitivity/epsilon)^2`.

## 6.3 Fairness Documentation

### 6.3.1 `docs/fairness_report.md`

Required sections:

- **Subgroup KPI table.** Copied from `reports/subgroup_report.md`.
- **Dataset skew acknowledgement.** VoxCeleb2 is heavily skewed toward English-speaking adult men in their 30s–40s. Report the imbalance numerically.
- **Mandatory subgroup result.** `keyword_syllable_count` macro-TA gap between best and worst bucket. This is the only mandatory subgroup metric (per Phase 4); all others are best-effort.
- **Planned mitigations.** Concrete dataset additions (e.g., "fold in a pinned Common Voice multilingual release in v1.1") with a target date.

### 6.3.2 Enrollment UX guards (in `live_demo.py`)

In `--enroll` mode:

- Refuse keywords < 2 syllables (use `g2p_en` to count vowel phones). Message: "Please choose a 2-syllable or longer keyword for reliability."
- Refuse keywords > 6 syllables. Message: "Please choose a shorter keyword (≤ 6 syllables) for low-latency wake."
- After successful enrollment, print: "SoloSpeak has learned how YOU say this phrase. It will not respond to other people saying it."

These are not security guarantees — they are usability guards.

## 6.4 Rollback

### 6.4.1 Two-slot deployment simulation

`demo/cli/live_demo.py` accepts `--model-slot {current,previous}`. Both slots are real ONNX files in `artifacts/`:

- `artifacts/solospeak_int8.onnx` (current)
- `artifacts/solospeak_int8_previous.onnx` (previous, optional — copy of an earlier checkpoint export)

Switching `--model-slot` at startup demonstrates the rollback architecture without needing real OTA infrastructure. The actual OTA dual-slot mechanism is Samsung-internal.

## 6.5 Documentation Inventory

By end of Phase 6, `docs/` contains:

| File | Purpose | Source |
|---|---|---|
| `architecture.md` | This document | hand-authored |
| `reproducibility.md` | Exact steps to reproduce all results from a fresh clone | hand-authored |
| `threat_model.md` | 6.1.1 above | hand-authored |
| `fairness_report.md` | 6.3.1 above | hand-authored |
| `ops_runbook.md` | What to do if a deployed model misbehaves | hand-authored |
| `deployment_guide.md` | OTA, rollback, canary outline | hand-authored |
| `api_reference.md` | Public function signatures | auto-gen from docstrings via `pdoc` |

### 6.5.1 Acceptance criteria — Phase 6

- 🧪 All seven `docs/*.md` files exist and pass `markdown-link-check`.
- 🧪 `python -m solospeak.security.replay_eval` produces a numeric output.
- 🧪 `python -m solospeak.security.adversarial_eval` produces a numeric output.
- 🧪 `python -c "from solospeak.observability.metrics import SoloSpeakLocalMetrics; print(SoloSpeakLocalMetrics().to_dict())"` succeeds.
- 🧪 Live demo with `--model-slot previous` boots and accepts audio.

---

# Phase 7: Demo & Submission

**Duration:** Days 42–46 (parallel with Phase 6).
**Goal:** Recording-quality demo video, final Phase-2 report, clean repo, submission email sent.
**Exit criterion:** Submission email sent ≥ 48 h before the Jun 22 2026 deadline (target: Jun 20).

## 7.1 Demo Video

### 7.1.1 Script (10 minutes total)

```
0:00–0:30   Title card. One-sentence pitch: "SoloSpeak wakes only when you say your keyword."
0:30–2:00   Enrollment demo. 30 s of real enrollment with voice-over.
2:00–3:30   Positive tests. User says keyword at 0.5 m / 2 m / 4 m. Wake fires each time.
3:30–5:00   Negative tests:
              - Friend says keyword (Q2 imposter) → rejected
              - User says phonetic neighbor (Q3 wrong-word) → rejected
              - TV/radio plays in background (Q4 background) → no wake
5:00–7:00   Architecture walkthrough with animated slides.
7:00–8:00   KPI dashboard with achieved numbers.
8:00–9:00   Samsung ecosystem fit: Bixby, Buds, SmartThings.
9:00–10:00  Closing: repo URL, license, contact.
```

### 7.1.2 Technical requirements

- 1080p, 30 fps minimum.
- Headset mic for narration; laptop mic only for the demo audio itself (so the model is hearing realistic audio).
- Captions auto-generated via Whisper, hand-corrected.
- Upload to YouTube as **unlisted**; include the link in the submission email.

### 7.1.3 What NOT to fake

- Do not edit out failed wakes. If the model fails on take 1, retake honestly.
- Do not synthesize the friend's voice for Q2; record a real second person.
- Do not boost the TA numbers in the dashboard slide; they must match `reports/kpi_final.json`.

## 7.2 Final Report

### 7.2.1 Phase-2 report structure (~25 pages PDF)

1. Executive summary (1 page).
2. Problem analysis (2 pages).
3. System architecture, with diagrams (3 pages).
4. Training methodology (3 pages).
5. Scalability & production readiness — Phase 6 content (3 pages).
6. Evaluation results, including subgroup table (4 pages).
7. Ablation analysis (2 pages, with seed-count caveat).
8. Samsung ecosystem fit (2 pages).
9. Honest limitations (2 pages — required, not optional).
10. Appendix: full KPI tables, subgroup eval, ablation matrix, threat model summary.

### 7.2.2 Writing rules

- Every quantitative claim has a number with provenance (config + commit hash + seed).
- Every limitation is named explicitly and not buried.
- Every figure has a caption that is readable standalone.
- Disclose seed counts: if ablations ran with 1 seed, say so.

## 7.3 Repository Cleanup

Pre-submission checklist:

- [ ] `README.md` shows the final KPI table prominently.
- [ ] README links: repo → demo video → docs → release artifacts.
- [ ] All test suites green (`make test`).
- [ ] Stale branches deleted.
- [ ] Final checkpoints in a GitHub Release (not git LFS — too quota-hungry).
- [ ] `artifacts/solospeak_int8.onnx` is the version reported in the final KPI table.
- [ ] LICENSE is Apache-2.0.
- [ ] `CITATION.cff` present.
- [ ] No `TODO` / `FIXME` / `XXX` comments on `main`.
- [ ] No hard-coded local paths (e.g. `/Users/pranav/...`).
- [ ] Docker image builds from a clean machine (`docker build -t solospeak .` succeeds).

### 7.3.1 GitHub Release

Tag the final commit `v1.0.0-phase2`. Attach:

- `stage6_qat.pt` (Stage-6 PyTorch checkpoint).
- `solospeak_int8.onnx` (deployment artifact).
- `solospeak_int8_previous.onnx` (rollback slot, for the demo).
- `kpi_final.json`, `probes.json`, `ablation_table.md`, `subgroup_report.md`.
- A link to the YouTube demo video.

## 7.4 Submission Package

### 7.4.1 Email template — copy-paste ready

```
To: ennovatex.io@samsung.com
Subject: AX Hackathon Phase 2 Submission | 04 | Resonant

Hello Samsung ennovateX team,

Please find enclosed our Phase 2 submission for Problem #04 — Speech Disentanglement.

Team: Resonant
Participant: Pranav Angrish (pangrish_be22@thapar.edu)
Institute: Thapar Institute of Engineering & Technology

Attachments / links:
  - Final report PDF: resonant-04-solospeak-final.pdf  (attached)
  - GitHub repository: https://github.com/<user>/solospeak
  - Demo video: https://youtu.be/<id>
  - Trained artifacts: https://github.com/<user>/solospeak/releases/tag/v1.0.0-phase2

Headline results (full tables in the report):
  - TA Clean: <X.X%>  (MIN gate 0.92, TARGET 0.96)
  - TA Noisy macro: <X.X%>  (MIN gate 0.80, TARGET 0.88)
  - FA per hour per user: <X.X>
  - Q2 imposter rejection: <X.X%>
  - Q3 phonetic-neighbor rejection: <X.X%>
  - Model size: 1.1M params, <X.X> MB INT8
  - xRT on ARM Cortex-A78: <X.XX>  (HARD gate 0.20, STRETCH 0.08)
  - Disentanglement: speaker-probe-on-z_c reduced by <Y%> vs Stage 2 baseline
  - All 4 quadrants (Q1/Q2/Q3/Q4) explicitly evaluated

Looking forward to Phase 3.

Best,
Pranav Angrish
```

Replace `<...>` placeholders with real numbers from `reports/kpi_final.json` before sending.

### 7.4.2 Submission timing

- Target send time: **Jun 20 2026, 18:00 IST** (≈48 h before the Jun 22 deadline).
- If schedule slips: send by **Jun 22 12:00 IST** at the latest (6+ h timezone buffer).
- If no acknowledgement within 24 h: re-send from the same email address with subject prefix `[RESEND]`.

### 7.4.3 Acceptance criteria — Phase 7

- 🧪 Submission email sent before the cutoff with all four links live.
- 🧪 Demo video plays end-to-end without buffering at 1080p.
- 🧪 GitHub Release `v1.0.0-phase2` exists with all six artifacts.
- 🧪 Cloning the repo on a fresh machine and running `make install-dev && make test` succeeds.

---

# Appendix A: Complete File Inventory

Every file that must exist in the final repo, in dependency order. A file's level is the earliest phase it can be created. Files at lower levels depend only on files at lower or equal levels.

## Level 0 — Repo scaffolding (Day 0)

```
README.md
LICENSE
CITATION.cff
pyproject.toml
requirements.txt
requirements-dev.txt
Makefile
Dockerfile
.gitignore
.pre-commit-config.yaml
.github/workflows/ci.yml
.github/workflows/eval-regression.yml
scripts/__init__.py
```

## Level 1 — Configuration & types (Days 1–2)

```
configs/defaults.yaml
configs/backbone/bcresnet1.yaml
configs/backbone/bcresnet5.yaml
configs/backbone/bcresnet8.yaml
configs/backbone/bcresnet10.yaml
configs/backbone/bcresnet16.yaml
configs/training/stage1_backbone_pretrain.yaml
configs/training/stage2_dual_head.yaml
configs/training/stage3_disentangle.yaml
configs/training/stage4_robustness.yaml
configs/training/stage5_fusion.yaml
configs/training/stage6_qat.yaml
configs/eval/full_kpi_suite.yaml
configs/eval/ablations.yaml
solospeak/__init__.py
solospeak/utils/__init__.py
solospeak/utils/types.py
solospeak/utils/config.py
solospeak/utils/seeding.py
solospeak/utils/audio.py
solospeak/utils/logging.py
```

## Level 2 — Data (Days 3–8)

```
data/README.md
data/licenses/.gitkeep
scripts/download_datasets.py
scripts/prepare_libriphrase.py
scripts/prepare_manifests.py
scripts/pack_audio_lmdb.py
scripts/precompute_speaker_embeddings.py
solospeak/data/__init__.py
solospeak/data/features.py
solospeak/data/features_deploy.py
solospeak/data/datasets.py
solospeak/data/augmentation.py
solospeak/data/hard_negatives.py
solospeak/data/splits.py
solospeak/data/samplers.py
tests/unit/test_features.py
tests/unit/test_augmentation.py
tests/unit/test_splits.py
tests/integration/test_data_pipeline.py
tests/fixtures/audio/reference_1p6sec.wav
data/manifests/STATS.md
data/manifests/STATS.json
data/manifests/keyword_vocab.json
data/manifests/gsc_vocab.json
data/manifests/speaker_vocab.json
```

## Level 3 — Models (Days 9–14)

```
solospeak/models/__init__.py
solospeak/models/backbones/__init__.py
solospeak/models/backbones/bcresnet.py
solospeak/models/heads.py
solospeak/models/fusion.py
solospeak/models/solospeak.py
solospeak/models/vad.py
solospeak/losses/__init__.py
solospeak/losses/supcon.py
solospeak/losses/orthogonality.py
solospeak/losses/adversarial.py
solospeak/losses/combined.py
tests/unit/test_backbones.py
tests/unit/test_heads.py
tests/unit/test_fusion.py
tests/unit/test_losses.py
tests/unit/test_solospeak.py
```

## Level 4 — Training (Days 15–32)

```
solospeak/training/__init__.py
solospeak/training/trainer.py
solospeak/training/schedulers.py
solospeak/training/callbacks.py
solospeak/training/training_wrapper.py
solospeak/training/stages/__init__.py
solospeak/training/stages/base.py
solospeak/training/stages/stage1_backbone.py
solospeak/training/stages/stage2_dual_head.py
solospeak/training/stages/stage3_disentangle.py
solospeak/training/stages/stage4_robustness.py
solospeak/training/stages/stage5_fusion.py
solospeak/training/stages/stage6_qat.py
solospeak/enrollment/__init__.py
solospeak/enrollment/tts_augmentation.py
solospeak/enrollment/calibration.py
solospeak/enrollment/templates.py
scripts/run_stage.py
tests/integration/test_training_smoke.py
```

**`run_proxy_ta` must be available at Level 4.** Stages 2–4 call `run_proxy_ta` from
their `on_epoch_end()` implementations. `kpi_suite.py` is a Level 5 file, so the agent
MUST implement `run_proxy_ta` by Level 4 using one of these two approaches (choose one
and stay consistent):

- **Preferred:** create `solospeak/training/proxy_eval.py` at Level 4 containing only
  `run_proxy_ta` (signature and contract from Appendix B / §3.1.6). When Level 5 arrives,
  `kpi_suite.py` imports and re-exports it: `from solospeak.training.proxy_eval import run_proxy_ta`.
- **Alternative:** create `solospeak/eval/kpi_suite.py` as a Level 4 partial stub that
  implements only `run_proxy_ta`; complete the rest of the file at Level 5.

Do NOT leave `run_proxy_ta` unimplemented while writing Stage 2–4 training code — the
`on_epoch_end()` gate check will be a no-op or crash without it.

## Level 5 — Evaluation (Days 33–37)

```
solospeak/eval/__init__.py
solospeak/eval/kpi_suite.py
solospeak/eval/subgroup.py
solospeak/eval/ablations.py
solospeak/eval/probes.py
solospeak/eval/xrt.py
scripts/run_eval.py
scripts/run_ablation.py
tests/integration/test_eval_smoke.py
```

**Note on `reports/` files.** `reports/kpi_final.json`, `reports/probes.json`,
`reports/ablation_table.md`, and `reports/subgroup_report.md` are **generated output
files**, not source files. Do NOT create placeholder versions during repo scaffolding.
They are produced by `make eval` and `make ablation` and are committed only when those
commands have run successfully with a trained model. Create the `reports/` directory with
a `.gitkeep` at scaffold time.

## Level 6 — Deployment (Days 38–41)

```
solospeak/deployment/__init__.py
solospeak/deployment/export_onnx.py
solospeak/deployment/quantize.py
solospeak/deployment/validate_artifact.py
solospeak/deployment/ota_package.py
solospeak/enrollment/service.py
solospeak/inference/__init__.py
solospeak/inference/streaming.py
solospeak/inference/hysteresis.py
solospeak/inference/multi_user.py
scripts/export_and_validate.py
demo/cli/live_demo.py
tests/integration/test_onnx_roundtrip.py
tests/unit/test_enrollment.py
tests/unit/test_inference.py
tests/unit/test_deployment.py
```

**Note on `artifacts/solospeak_int8.onnx`.** This is a **generated artifact** produced
by `make export`. Do NOT create a placeholder file during scaffolding. Create the
`artifacts/` directory with a `.gitkeep`. The final ONNX file is attached to the GitHub
Release, not committed to git.

## Level 7 — Production hardening & docs (Days 42–44)

```
solospeak/security/__init__.py
solospeak/security/replay_eval.py
solospeak/security/adversarial_eval.py
solospeak/security/threat_model.py
solospeak/observability/__init__.py
solospeak/observability/metrics.py
solospeak/observability/dp_noise.py
docs/architecture.md
docs/reproducibility.md
docs/threat_model.md
docs/fairness_report.md
docs/ops_runbook.md
docs/deployment_guide.md
docs/api_reference.md
```

## Level 8 — Skills (any time)

```
skills/data_curation/SKILL.md
skills/training/SKILL.md
skills/evaluation/SKILL.md
skills/deployment/SKILL.md
skills/debugging/SKILL.md
```

Each `SKILL.md` is a short markdown playbook for a human operator or AI agent working in
that domain. They are **documentation only** — no code, no imports. Required sections per file:

| File | Purpose |
|---|---|
| `skills/data_curation/SKILL.md` | How to add a new corpus: license check, download script pattern, manifest columns, LMDB packing |
| `skills/training/SKILL.md` | How to run a stage, resume a checkpoint, tune hyperparameters, interpret W&B curves |
| `skills/evaluation/SKILL.md` | How to run `make eval`, interpret KPI JSON fields, add a new metric to `kpi_suite.py` |
| `skills/deployment/SKILL.md` | How to export ONNX, run INT8 quantization, validate gates, build the OTA package |
| `skills/debugging/SKILL.md` | Common failure modes and their fixes (mirrors Appendix E but with more operational detail) |

These files are populated over the course of the project; stubs with just the required
section headers are sufficient at scaffold time.

---

# Appendix B: Function Contracts Reference

All public function signatures in one place. If any of these change, update this appendix in the same PR.

## Data

```python
# solospeak/data/features.py
class LogMelExtractor(nn.Module):
    def forward(self, waveform: torch.Tensor) -> torch.Tensor: ...
    # (B, 25600) -> (B, 1, 80, 160)

# solospeak/data/features_deploy.py
class LogMelExtractorDeploy:
    def __call__(self, waveform: np.ndarray) -> np.ndarray: ...
    # (25600,) float32 -> (1, 80, 160) float32

# solospeak/data/datasets.py
def parse_bool_csv(value: DatasetValue) -> bool: ...
# Parses canonical manifest booleans. Accepts bool, "true", or "false"; raises otherwise.

class GSCDataset(Dataset):
    def __getitem__(self, idx: int) -> DatasetItem: ...
    # {'wav': Tensor(25600,), 'keyword_label': int, 'keyword_text': str}
    # keyword_label indexes gsc_vocab.json

class DualHeadDataset(Dataset):
    def __getitem__(self, idx: int) -> DatasetItem: ...
    # {'wav': Tensor(25600,), 'keyword_label': int, 'speaker_label': int,
    #  'speaker_id_raw': str, 'keyword_text': str}
    # keyword_label may be WORD_IGNORE_INDEX (-100) for speaker-only rows

class QuadrantDataset(Dataset):
    def __getitem__(self, idx: int) -> DatasetItem: ...
    # {'wav': Tensor(25600,), 'quadrant_label': int,  # 1=accept 0=reject
    #  'quadrant_class': str,       # "Q1_accept"|"Q2_imposter"|"Q3_wrong_word"|"Q4_background"
    #  'user_id': str,
    #  'keyword_text': str,
    #  'enrolled_keyword_text': str,
    #  'profile_id': str,
    #  'trial_source': str,         # "real"|"augmented"|"tts"|"background"
    #  'q3_gate_eligible': bool,
    #  'synthesis_backend': str,
    #  'speaker_verification_score': float | None,
    #  'content_template': Tensor(128,),   # L2-normalized enrollment template
    #  'speaker_template': Tensor(128,)}   # L2-normalized enrollment template

# solospeak/data/splits.py
def assign_split(speaker_id: str, seed: int = 42,
                 train_pct: float = 0.85, dev_pct: float = 0.10
                 ) -> Literal["train", "dev", "test"]: ...

# solospeak/data/samplers.py
class ClassAwareBatchSampler(Sampler[list[int]]):
    def __init__(self, labels: list[int], batch_size: int,
                 num_classes_per_batch: int = 8,
                 num_samples_per_class: int = 16,
                 seed: int = 42) -> None: ...
    def __iter__(self) -> Iterator[list[int]]: ...

# solospeak/data/hard_negatives.py
def phone_edit_distance(a: list[str], b: list[str]) -> int: ...
def phonetic_hard_negatives(keyword: str, vocab: list[str], k: int = 10) -> list[str]: ...
def speaker_hard_negatives(anchor_id: str, embeddings: np.ndarray,
                            speaker_ids: list[str], k: int = 10) -> list[str]: ...
```

## Models

```python
# solospeak/models/backbones/bcresnet.py
class SoloSpeakResNet(nn.Module):
    def __init__(self, variant: str = "bcresnet8") -> None: ...
    def forward(self, mel: torch.Tensor) -> torch.Tensor: ...
    def fuse_model(self) -> None: ...
    @property
    def output_channels(self) -> int: ...

BCResNet = SoloSpeakResNet  # backward-compatible alias only

# solospeak/models/vad.py
class SileroVAD:
    FRAME_SAMPLES: ClassVar[int]   # 512
    CONTEXT_SAMPLES: ClassVar[int] # 64
    SAMPLE_RATE: ClassVar[int]     # 16000
    def __init__(self, onnx_path: Path, threshold: float = 0.5) -> None: ...
    def is_speech(self, frame: np.ndarray) -> bool: ...
    # frame: (512,) float32 — raises ValueError for wrong shape
    def reset_state(self) -> None: ...

# solospeak/models/heads.py
class EmbeddingHead(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int = 256,
                 output_dim: int = 128, dropout: float = 0.1) -> None: ...
    def forward(self, feat: torch.Tensor) -> torch.Tensor: ...
    # (B, C, 1, T') -> (B, output_dim) L2-normalized

class AuxiliaryHeads(nn.Module):
    def __init__(self, embed_dim: int, n_words: int, n_speakers: int) -> None: ...
    # n_words and n_speakers loaded at runtime from STATS.json
    def forward(self, z_c: torch.Tensor, z_s: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]: ...

# solospeak/models/fusion.py
class GatedFusionMLP(nn.Module):
    def __init__(self, config: FusionConfig) -> None: ...
    def forward(self, s_c: torch.Tensor, s_s: torch.Tensor) -> torch.Tensor: ...
    # (B,), (B,) -> (B,) probability in [0, 1]

# solospeak/models/solospeak.py
class SoloSpeakModel(nn.Module):
    fusion_mlp: GatedFusionMLP   # attribute
    def forward(self, mel: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]: ...   # (z_c, z_s)
    def forward_fusion(self, s_c: torch.Tensor, s_s: torch.Tensor
                        ) -> torch.Tensor: ...
```

## Losses

```python
# solospeak/losses/supcon.py
class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None: ...
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor
                ) -> torch.Tensor: ...

# solospeak/losses/orthogonality.py
def orthogonality_loss(z_c: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor: ...

# solospeak/losses/adversarial.py
def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor: ...

class AdversarialProbeHead(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int) -> None: ...
    def forward(self, z: torch.Tensor) -> torch.Tensor: ...

# solospeak/losses/combined.py
class CombinedLoss(nn.Module):
    def __init__(self, weights: LossWeights, stage: int) -> None: ...
    def forward(self, outputs: dict[str, BatchDict],
                batches: StageBatch, step: int) -> LossDict: ...
    # Returns: {'total': Tensor, 'supcon_c': Tensor, 'supcon_s': Tensor,
    #           'aux_ce': Tensor, 'ortho': Tensor, 'adv': Tensor}
    # Stage keys: "gsc" (Stage 1), "content"/"speaker" (Stages 2-4),
    # "quadrant" (Stage 5), "distill" plus optional "quadrant" (Stage 6).
```

## Training

```python
# solospeak/training/stages/base.py
class TrainingStage(ABC):
    stage_id: ClassVar[int]
    stage_name: ClassVar[str]
    min_gate_metric: ClassVar[str]
    min_gate_threshold: ClassVar[float]
    target_gate_threshold: ClassVar[float]

    def prepare_data(self) -> tuple[DataLoader, ...]: ...   # @abstractmethod
    def build_model(self) -> nn.Module: ...                  # @abstractmethod
    def compute_loss(self, batches: StageBatch, step: int) -> LossDict: ...  # @abstractmethod
    def on_epoch_end(self, epoch: int) -> MetricsDict: ...  # @abstractmethod
    def go_no_go_check(self, metrics: MetricsDict) -> tuple[bool, bool]: ...  # @abstractmethod
    def run(self) -> Path: ...                               # @abstractmethod

# solospeak/training/trainer.py
class Trainer:
    def __init__(self, config: SoloSpeakConfig) -> None: ...
    def run_stage(self, stage_id: int) -> Path: ...
    def run_all_stages(self) -> list[Path]: ...
    def resume_from_stage(self, stage_id: int) -> None: ...

# solospeak/training/training_wrapper.py
class TrainingWrapper(nn.Module):
    model: SoloSpeakModel
    aux_heads: AuxiliaryHeads
    content_adversary: AdversarialProbeHead
    speaker_adversary: AdversarialProbeHead
    def __init__(self, config: SoloSpeakConfig) -> None: ...
    def forward(self, mel: torch.Tensor, adv_lambda: float = 0.0) -> dict[str, torch.Tensor]: ...
    def deployable_state_dict(self) -> dict[str, torch.Tensor]: ...

# solospeak/training/stages/stage6_qat.py
def copy_float_weights_from_prepared(
    prepared: nn.Module,
    target: nn.Module,
) -> None: ...
# Copies float weights from a QAT-prepared model into a clean target model.
# Raises ValueError if a name in target is absent from prepared (architecture mismatch).
# See §3.7 for full spec and test contract.
```

## Evaluation

```python
# solospeak/eval/kpi_suite.py
def run_kpi_suite(model_path: Path, eval_config: EvalConfig) -> KPIResult: ...

def run_proxy_ta(model: torch.nn.Module, eval_config: EvalConfig,
                 noisy: bool = False) -> dict[str, float]: ...
# Stage 2-4 only. Uses the fixed pre-fusion proxy protocol from §3.1.6:
# proxy_score = min(content cosine, speaker cosine), per-profile tau_proxy calibration.

# solospeak/eval/probes.py
def verify_disentanglement(encoder: nn.Module, probe_data: DataLoader,
                            stage2_baseline: ProbeBaseline | None = None) -> ProbeResult: ...
# Returns: {'acc_c_post': float,       # speaker-probe accuracy on z_c
#           'acc_s_post': float,       # word-probe accuracy on z_s
#           'reduction_c': float | None,  # relative reduction vs Stage-2 baseline on z_c
#           'reduction_s': float | None}  # relative reduction vs Stage-2 baseline on z_s
# reduction_* are None when stage2_baseline is not provided.
# Key names match the Phase 4.2 spec exactly; do not use 'reduction_vs_baseline'.

# solospeak/eval/xrt.py
def measure_xrt(onnx_path: Path, num_clips: int = 100,
                clip_duration_s: float = 1.6, threads: int = 1) -> XRTResult: ...
# Returns: {'p50': float, 'p95': float, 'p99': float, 'platform': str}

# solospeak/eval/ablations.py
def run_ablations(config: AblationConfig) -> AblationResult: ...
```

## Deployment

```python
# solospeak/deployment/export_onnx.py
def export_onnx(checkpoint_path: Path, output_path: Path,
                opset: int = 17, fixed_time_dim: int = 160,
                include_fusion: bool = True) -> None: ...

# solospeak/deployment/quantize.py
def quantize_static(fp32_onnx: Path, int8_onnx: Path,
                    calibration_data: Iterable[dict[str, np.ndarray]]) -> None: ...

# solospeak/deployment/validate_artifact.py
def validate(onnx_path: Path, eval_set: EvalSet,
             config: DeploymentConfig,
             gates: ValidationGates,
             fp32_onnx_path: Path | None = None) -> ValidationReport: ...
# fp32_onnx_path enables Gate 10 (INT8 vs FP32 TA degradation check).
# Gate 10 is required for final Phase-5 validation and Stage-6 temporary dev-set checks;
# omitting fp32_onnx_path is allowed only for quick local/debug validation.
```

## Enrollment & Inference

```python
# solospeak/enrollment/service.py
class EnrollmentService:
    def enroll(self, user_id: str, keyword_text: str,
               recordings: list[np.ndarray]) -> UserProfile: ...
    def calibrate_threshold(self, profile: UserProfile,
                             rejection_samples: list[np.ndarray]) -> float: ...
    # rejection_samples: list of rejection waveforms, each (T,) float32.
    # At least 10 samples recommended to sweep thresholds meaningfully.
    # Picks the lowest threshold that keeps estimated FA/hr/user ≤ 2.0.
    # Returns FusionConfig.tau_on as a conservative default if the list is empty.

# solospeak/inference/streaming.py
class StreamingInference:
    def __init__(self, onnx_path: Path) -> None: ...
    def step(self, audio_frame: np.ndarray) -> list[WakeEvent]: ...
    def enroll_user(self, profile: UserProfile) -> None: ...
    def remove_user(self, user_id: str) -> None: ...
```


---

# Appendix C: Configuration Schema

The main schema is defined in **0.2.3** (`SoloSpeakConfig` and its sub-models). This appendix lists the additional config types referenced from later phases.

> **⚠️ IMPLEMENTING AGENT — READ BEFORE WRITING CODE.**
> The block below is **NOT a separate file**. It is a set of additions that must be
> merged directly into `solospeak/utils/config.py` below the existing definitions from
> section 0.2.3. Do NOT create a new file. Merge the new imports (`dataclass`, `field`,
> `TypeVar`) into the existing top-level import block and append the new classes below
> `_load_yaml_with_base`. All classes in this block reference `_load_yaml_with_base`,
> which is already defined in 0.2.3 — it will only be in scope if the merge is done
> correctly.

```python
# solospeak/utils/config.py — additions to the file from 0.2.3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, Field


ConfigT = TypeVar("ConfigT", bound="_YamlConfig")


class _YamlConfig(BaseModel):
    """Shared YAML loader for non-training config models.

    Uses the same `__base__` inheritance mechanism as SoloSpeakConfig.from_yaml()
    from section 0.2.3. Do not hand-roll separate YAML parsing in eval, ablation, or
    deployment scripts.
    """

    @classmethod
    def from_yaml(cls: type[ConfigT], path: Path | str) -> ConfigT:
        data = _load_yaml_with_base(Path(path))
        return cls.model_validate(data)

    def to_yaml(self, path: Path | str) -> None:
        with open(Path(path), "w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)


@dataclass(frozen=True)
class EvalSet:
    """Resolved eval manifests and any loaded calibration metadata."""
    test_kpi_manifest: Path
    test_fa_manifest: Path
    keyword_vocab: Path = field(
        default_factory=lambda: Path("data/manifests/keyword_vocab.json")
    )
    speaker_vocab: Path = field(
        default_factory=lambda: Path("data/manifests/speaker_vocab.json")
    )


@dataclass(frozen=True)
class ValidationResult:
    name: str
    passed: bool
    value: float | str
    threshold: float | str
    message: str = ""


@dataclass
class ValidationReport:
    results: list[ValidationResult]

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)


class ArtifactValidationError(RuntimeError):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__("artifact validation failed")


class EvalConfig(_YamlConfig):
    test_manifests: list[Path] = Field(
        default_factory=lambda: [Path("data/manifests/test_kpi.csv"),
                                  Path("data/manifests/test_fa.csv")]
    )
    noise_snrs_db: list[int] = Field(default_factory=lambda: [-5, 0, 5, 10, 15, 20, 25, 30])
    distance_buckets_m: list[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0, 3.5, 5.0])
    fa_audio_hours: float = 10.0
    fa_n_user_buckets: list[int] = Field(default_factory=lambda: [1, 4, 8])
    n_seeds: int = 3
    subgroups_optional: list[str] = Field(default_factory=lambda: ["gender", "age", "accent"])
    subgroups_mandatory: list[str] = Field(default_factory=lambda: ["keyword_syllable_count"])
    q3_neighbors_per_keyword: int = 5
    q3_max_edit_distance: int = 2
    q3_synthesis_fallback: bool = True   # use Parler-TTS if real utterances unavailable


class AblationConfig(_YamlConfig):
    ablations: list[Literal["full", "no_ortho", "no_adv", "no_disent",
                             "no_tts_enroll", "no_gated_fusion", "no_curriculum"]] = Field(
        default_factory=lambda: ["full", "no_ortho", "no_adv", "no_disent",
                                  "no_tts_enroll", "no_gated_fusion", "no_curriculum"]
    )
    seeds: list[int] = Field(default_factory=lambda: [42])    # default: 1 seed each
    extra_seeds_for: list[str] = Field(default_factory=lambda: ["full", "no_disent"])
    extra_seeds: list[int] = Field(default_factory=lambda: [137, 2718])
    base_checkpoint: Path = Path("checkpoints/stage2_dualhead.pt")
    max_parallel: int = 1


class DeploymentConfig(_YamlConfig):
    target_platform: Literal["arm64_android", "arm64_linux", "x86_64_linux"] = "arm64_android"
    xrt_target: float = 0.08              # our stretch goal
    onnx_opset: int = 17
    fixed_time_dim: int = 160
    streaming_hop_ms: int = 160
    calibration_num_samples: int = 500
    quant_path: Literal["qat_then_ort_ptq", "qat_only", "ptq_only"] = "qat_then_ort_ptq"
    fuse_bn_into_conv: bool = True


class ValidationGates(_YamlConfig):
    """Hard ship-gates for the deployed artifact. See Phase 5.3."""
    max_filesize_mb: float = 5.0
    min_opset: int = 17
    max_xrt: float = 0.20                 # was 0.08 in v1 (which conflicted with tests)
    min_ta_clean: float = 0.92            # TARGET is 0.96
    min_ta_noisy_macro: float = 0.80      # TARGET is 0.88
    max_fa_per_hr_per_user: float = 2.0   # was 1.0 in v1
    max_param_count: int = 3_000_000
    max_int8_vs_fp32_degradation_pp: float = 1.0   # MIN gate; TARGET 0.3


def load_eval_config_bundle(path: Path | str) -> tuple[EvalConfig, ValidationGates]:
    """Load `configs/eval/full_kpi_suite.yaml`.

    The file's top-level keys map to EvalConfig. An optional `validation_gates`
    child object overrides ValidationGates defaults. This keeps KPI protocol
    settings and ship-gate thresholds in one human-editable file without forcing
    eval scripts to parse raw dictionaries.
    """
    data = dict(_load_yaml_with_base(Path(path)))
    gates_data = data.pop("validation_gates", {}) or {}
    return EvalConfig.model_validate(data), ValidationGates.model_validate(gates_data)
```

Implementation note: merge the import additions above into the top import block from
0.2.3, then append the dataclasses/config classes below the existing `SoloSpeakConfig`
helpers. Do not paste a second `from __future__ import annotations` into the middle of
`config.py`; these contracts are valid on Python 3.10+ without postponed annotations.

The numbers above are the **MIN ship-gates**. The TARGET aspirations from earlier phases are tracked separately and reported in the final KPI table but do not block shipping.

**Config loading rule.** `scripts/run_eval.py`, `scripts/run_ablation.py`,
`scripts/export_and_validate.py`, and deployment validation code MUST load these models
through `load_eval_config_bundle()`, `AblationConfig.from_yaml()`,
`DeploymentConfig.from_yaml()`, or `ValidationGates.from_yaml()` as appropriate. They use
the same `__base__` deep-merge behavior as `SoloSpeakConfig`; no script may parse these
YAML files into untyped dictionaries and then cherry-pick keys.

Required YAML files:
- `configs/eval/full_kpi_suite.yaml` maps to `EvalConfig` plus optional
  `ValidationGates` overrides under a top-level `validation_gates` key.
- `configs/eval/ablations.yaml` maps to `AblationConfig`.
- Deployment/export commands may accept an explicit deployment YAML; if omitted, they use
  `DeploymentConfig()` and `ValidationGates()` defaults from this appendix.

---

# Appendix D: Testing Matrix

Which tests must pass at which phase, runtime budgets, and which are advancement-blockers vs informational.

| Test | Phase | Blocks Advance? | Runtime |
|---|---|---|---|
| `ruff check` | 0+ | Yes | < 10 s |
| `mypy solospeak` | 0+ | Yes | < 30 s |
| All modules importable | 0 | Yes | < 5 s |
| `SoloSpeakConfig.from_yaml` roundtrip | 0 | Yes | < 1 s |
| `make test` Phase-0 safe suite | 0 | Yes | < 2 min |
| `make test-all` full integration suite | 1+ | No for Phase 0 | varies |
| Smoke manifests (`download-data-smoke`, `prepare-manifests-smoke`) | 1 | Yes | < 30 s |
| LMDB packing idempotency (`make pack-lmdb`) | 1 | Yes for full path | varies by corpus |
| Feature extractor train/deploy parity (`atol=3e-3`) | 1 | Yes | < 10 s |
| `assign_split` produces no speaker overlap | 1 | Yes | < 30 s |
| Augmentation preserves shape/dtype/range | 1 | Yes | < 10 s |
| DataLoader benchmark — MIN ≥ 80 samples/s | 1 | Yes | < 60 s |
| DataLoader benchmark — TARGET ≥ 200 samples/s | 1 | No | < 60 s |
| `test_param_counts` for all 5 backbone variants | 2 | Yes | < 30 s |
| `test_forward_shape` `(4,1,80,160) → (4,C,1,20)` | 2 | Yes | < 5 s |
| `test_backward` produces finite gradients | 2 | Yes | < 10 s |
| `test_fuse_model_export` fused backbone exports | 2 | Yes | < 30 s |
| Head L2 norm == 1 (`atol=1e-5`) | 2 | Yes | < 5 s |
| `test_fusion_param_count == 361` | 2 | Yes | < 1 s |
| `test_total_param_count_bcresnet8 ∈ [1.08M, 1.125M]` | 2 | Yes | < 5 s |
| `torch.onnx.export` opset 17 succeeds | 2 | Yes | < 30 s |
| Stage 1 smoke (100 steps converge to declining loss) | 3 | Yes | < 5 min |
| Stage 1 MIN gate: GSC dev ≥ 0.92 | 3 | Yes | 1–3 days |
| Stage 2 MIN gate: dev clean TA ≥ 0.90 | 3 | Yes | 3–5 days |
| Stage 3 MIN gate: ≥ 30% relative probe reduction vs S2 | 3 | Yes | 2 days |
| Stage 3 mode-collapse guard: dev TA ≥ 0.95 × S2 | 3 | Yes | (in S3) |
| Stage 4 MIN gate: noisy TA macro ≥ 0.80 | 3 | Yes | 2–3 days |
| Stage 5 MIN gate: per-quadrant ≥ 0.85 | 3 | Yes | 1 day |
| Stage 6 MIN gate: temporary ORT INT8 vs FP32 Δ ≤ 1.0 pp | 3 | Yes | 1 day |
| Full KPI suite on test set | 4 | Yes | < 1 hr |
| Ablation matrix (7 × 1 seed default) | 4 | No for `make eval`; yes for final report if compute allows | ≈ 14 GPU-days |
| Subgroup eval: `keyword_syllable_count` only | 4 | Yes | < 30 min |
| Subgroup eval: gender/age/accent | 4 | No | < 30 min |
| xRT MIN gate: `< 0.20` on ARM | 4 | Yes | < 5 min |
| xRT TARGET: `< 0.08` on ARM | 4 | No | < 5 min |
| Final ONNX export | 5 | Yes | < 1 min |
| INT8 quantization | 5 | Yes | < 5 min |
| All 9 required validation gates + Gate 10 with `fp32_onnx_path` for final release | 5 | Yes | < 30 min |
| Live demo enrollment (manual) | 5 | Yes | manual |
| Live demo wake on real voice within 250 ms post-window latency (manual) | 5 | Yes | manual |
| Replay attack baseline number | 6 | No | < 30 min |
| Adversarial eval baseline number | 6 | No | < 30 min |
| All seven `docs/*.md` files exist | 6 | Yes | < 1 s |
| Submission email sent before cutoff | 7 | Yes | manual |

**Reading the table:** "Blocks Advance? = Yes" means the test is a prerequisite for moving to the next phase. "No (informational)" means the test runs and is reported, but does not block. This is deliberate: the v1 doc treated every test as blocking, which would have stopped the project at the first 0.1 pp shortfall.

---

# Appendix E: Failure Mode Playbook

If X fails during implementation, do Y. Listed in rough order of likelihood.

## E.1 Stage 1 fails to reach the MIN gate (GSC dev ≥ 0.92)

**Symptoms:** Training loss is still falling at end of epoch 30, dev acc plateaus around 0.88–0.91.

**Actions in order:**

1. Verify data: print 10 random samples, confirm labels align with class names.
2. Drop LR from 3e-3 to 1e-3 (the 3e-3 starting point may be too aggressive for this project-specific variant).
3. Disable augmentation for the first 5 epochs (warm start the backbone before adding noise).
4. Increase batch size from 256 to 512 if GPU memory allows.
5. Try `bcresnet10` (slightly more capacity) instead of `bcresnet8`.
6. Last resort: extend training by 10 epochs and accept a longer Phase 1.

## E.2 Stage 3 disentanglement fails (probe reduction < 30%)

**Symptoms:** After 2 days of Stage 3, the speaker-probe-on-z_c baseline is 0.78 and current is 0.74 — only 5% relative reduction.

**Actions in order:**

1. Verify the ramp-up: log `lambda_adv` and `lambda_ortho` per step, confirm they increase from 0 to target over 5000 steps.
2. Increase the probe head's capacity: from `Linear(128, 256) → Linear(256, n)` to `Linear(128, 512) → Linear(512, 256) → Linear(256, n)`. A weak probe cannot exert decorrelation pressure.
3. Check for mode collapse: if dev TA dropped > 5% (the mode-collapse guard), reduce `lambda_adv` from 0.1 to 0.05 and re-ramp over 10000 steps.
4. Extend Stage 3 by 1 day at the higher `lambda_adv`.
5. Documented fallback (per architecture risk register R4): drop the adversarial loss, keep orthogonality only. Report the partial disentanglement honestly in the final report.

## E.3 Stage 4 noisy TA stuck < MIN (0.80)

**Symptoms:** Clean TA = 0.96, noisy TA at SNR=0 dB = 0.62, macro = 0.74.

**Actions:**

1. Increase `AddNoise` probability from 0.7 to 0.95.
2. Add more RIR variety: download additional RIRs from OpenSLR-26.
3. Make SpecAugment more aggressive: 3 masks per axis, up to 30% coverage.
4. Mine hard negatives at the SNR buckets that fail (typically −5 dB and 0 dB).
5. Extend Stage 4 by 2 more days if compute allows.

## E.4 xRT > 0.20 on ARM (HARD gate fail)

**Symptoms:** ONNX Runtime profiling shows the model takes 0.42 × audio duration per inference.

**Actions:**

1. Profile with `onnxruntime.SessionOptions(enable_profiling=True)` to identify the slow op.
2. Common culprits: large `Softmax` / `LogSoftmax` — replace with `Sigmoid` where possible.
3. Confirm BN was fused into Conv at export. Use the backbone/export wrapper's
   `fuse_model()` method; do not use top-level `conv1`/`bn1` names because they do not
   exist on the assembled model.
4. Apply `onnxsim` graph simplifier: `python -m onnxsim model.onnx model_simplified.onnx`.
5. If still slow: drop to `bcresnet5` (≈ 50% of bcresnet8 params). Re-evaluate TA — likely loses 1–2 pp clean and 2–4 pp noisy.

## E.5 ONNX export fails with "op not supported"

**Symptoms:** `torch.onnx.export` raises an error at a specific layer.

**Actions:**

1. Confirm `opset_version=17` (most modern ops supported).
2. If `F.normalize` fails: replace with manual `x / (x.norm(dim=-1, keepdim=True).clamp(min=1e-8))`.
3. If `GELU` fails: switch to `nn.ReLU` (the backbone already uses ReLU, so this affects only heads). Re-verify accuracy.
4. If `AdaptiveAvgPool2d((1, None))` fails: use `AdaptiveAvgPool2d((1, target_t))` with the fixed `target_t` from `DeploymentConfig.fixed_time_dim // 8`.
5. Last resort: register a custom op in ONNX Runtime via the contrib opset.

## E.6 Validation gate fails on the final artifact

**Symptoms:** One of the 9 gates in `validate_artifact.py` fails on the final INT8 artifact.

**Actions:**

1. Identify which gate from the report.
2. **File size** > 5 MB: re-quantize with more aggressive settings, strip unused ops.
3. **Accuracy** below MIN: more QAT fine-tuning (extend Stage 6 by 1 day).
4. **xRT** above 0.20: see E.4.
5. **Param count** > 3M: variant too large; switch to `bcresnet8` (985K) or `bcresnet5` (513K).
6. Do NOT ship an artifact that fails any gate, even by 1%. Either fix it or document it as a known limitation in the final report.

## E.7 Kaggle disconnection mid-training

**Symptoms:** Kaggle session timed out, the run died at step 12000 of 30000.

**Actions:**

1. Verify checkpoint-every-500-steps is enabled (configured in `TrainingConfig`).
2. Re-launch the same notebook; `run_stage.py --resume-from checkpoints/stageN_step12000.pt` should pick up where it left off.
3. If Kaggle is unreliable for multi-day runs: split into shorter notebook sessions, ≤ 6 h each, with explicit checkpointing between sessions.
4. Consider parallel notebooks: 2 Kaggle accounts, alternating 12 h shifts, syncing checkpoints via a private GitHub Release.
5. Last resort: switch to Colab Pro for the affected stage.

## E.8 Solo-contestant schedule slip

**Symptoms:** Behind by > 2 days at a Phase exit gate.

**Scope cut list, in order of preference:**

1. Drop `bcresnet1` and `bcresnet16` variants from the final report (keep `5/8/10`).
2. Reduce ablations from 7 × 1 seed to 4 × 1 seed (drop `no_tts_enroll`, `no_gated_fusion`, `no_curriculum`). Document this honestly.
3. Drop the optional subgroup evals (gender/age/accent). Keep only the mandatory `keyword_syllable_count`.
4. Drop the Android demo. The CLI demo is sufficient for the demo video.
5. Drop the multilingual stretch goal entirely.
6. Drop the rollback two-slot demo from Phase 6.4 (still document it in the deployment guide).
7. Drop the adversarial robustness number; document as future work.
8. If still behind: submit on Jun 22 deadline with the BEST-effort current state, prioritizing a working live demo over feature completeness. The judges will see what works; missing items go in the limitations section honestly.

The doc deliberately does NOT instruct you to lower MIN gates as a slip remedy. If the model cannot meet a MIN gate, that is a substantive scientific result and goes in the limitations section, not the abstract.


---

# Appendix F: Changelog from v1

The 38 substantive issues identified in the v1 audit, plus v2.1–v2.4 hardening fixes,
mapped to the section number where the fix lives.

| # | v1 issue | v2 fix | Section |
|---|---|---|---|
| 1 | BC-ResNet under-specified — agent reverse-engineered architecture from param count targets | Concrete `NormalBlock` and `TransitionBlock` with exact layers + analytically computed param counts per variant. Doc explicitly states "the spec is the contract; do not consult the paper to correct it." | 2.1.2, 2.1.6 |
| 2 | Mixing paper accuracy with project-specific implementation | Doc says SoloSpeakResNet is inspired by but NOT identical to the BC-ResNet paper. Plain Conv2d, no broadcasted residual; `bcresnet*` names are config aliases only. | 2.1.1 |
| 3 | Shape contract said "T/8 from 4 transition blocks" but 4 transitions would give T/16 | Corrected: 3 transition blocks each stride (2,2) in time, giving exactly T/8. Stem stride is (2,1) — frequency only, time preserved. | 2.1.4 |
| 4 | Fusion MLP config said `[32, 16]` but implementation discussion changed it to `[20, 10]` | `FusionConfig.hidden_dims = [20, 10]` everywhere, with the param count 361 (not 400) committed. | 0.2.3, 2.5 |
| 5 | "~400 param fusion MLP" claim numerically wrong | Doc says the count is exactly 361. Tests assert `== 361`. | 2.5 |
| 6 | Naming collision: `self.fusion = GatedFusionMLP(...)` AND `def fusion(...)` | `self.fusion_mlp` is the attribute; `forward_fusion()` is the method. They never share a name. | 2.4 |
| 7 | Pydantic config used bare instances as defaults | All sub-models use `Field(default_factory=...)`. | 0.2.3 |
| 8 | Unused `field_validator` import | Removed from imports unless actually used. | 0.2.3 |
| 9 | `solospeak.__version__` test required, but `__init__.py` content unspecified | `solospeak/__init__.py` defined explicitly with `__version__ = "0.1.0"`. | 0.2.1 |
| 10 | Stubs raised `NotImplementedError` at import (caused import-time failure) | Doc clarifies: imports must succeed; constructors may raise only when instantiated. | 0.2.2 |
| 11 | LibriPhrase generation not pinned | LibriPhrase recipe pinned: LibriSpeech train-clean-100/360 + Montreal Forced Aligner alignments + minimum phrase duration 0.4 s + keyword vocab capped to 1500 unique phrases. | 1.1.2 |
| 12 | VoxCeleb single point of failure | Documented fallback: VoxCeleb2 preferred; if access/terms fail, use a pinned Common Voice English release plus LibriSpeech, marked lower-confidence for speaker identity. | 1.1.4 |
| 13 | Kaggle storage plan unrealistic | Strategy pinned: pre-process locally, pack as LMDB → upload as a single read-only Kaggle Dataset. Manifests reference the LMDB keys. | 1.1.3 |
| 14 | `train_speaker.csv` at 1M rows too ambitious | Capped at ≈ 500K rows: 50 utterances/speaker × 10K speakers, sampled from VoxCeleb2 dev (or fallback). | 1.2.2 |
| 15 | DataLoader ≥ 200 samples/sec target unrealistic with augmentation | Split into MIN ≥ 80 (advancement gate) and TARGET ≥ 200 (aspiration). | 1.5 |
| 16 | Train/deploy feature parity tolerance 1e-5 too strict for FP32 STFT | Tolerance relaxed to 3e-3, with rationale (PyTorch FFT vs NumPy pocketfft FP32 rounding diverges by ≈ 2.4e-3 max). | 1.3.1 |
| 17 | 15 training days too aggressive | Phase 3 extended to days 15–32 (≈ 18 days). Total schedule extended to day 46. | Phase 3 header |
| 18 | GO/NO-GO gates too aggressive | Every gate split into `MIN:` (advancement) and `TARGET:` (aspiration). MIN gates lowered to ship-realistic values. | 3.0, all stage sections |
| 19 | Stage 3 disentanglement metric (probe ≤ 20%) was N-classes-dependent | New metric: **relative probe-accuracy reduction vs Stage 2 baseline**. MIN 30% reduction, TARGET 60%. Probe protocol pinned: `Linear(128, 256) → ReLU → Linear(256, n)`, AdamW lr=1e-3, 10 epochs, balanced 50 utt/class, max 200 classes. | 3.4.1, 4.2 |
| 20 | SupCon test ("loss = 0 when all embeddings identical and labels identical") was mathematically wrong | Replaced with: (a) loss is finite and non-negative for any input; (b) loss is monotonically lower when positive pairs are closer; (c) gradient sign on a known-direction perturbation matches expected. | 2.3.1 |
| 21 | Orthogonality test ("loss = 0 when z_c, z_s orthogonal per-dim") fragile after centering | Replaced: loss → 0 when `z_c` and `z_s` are sampled independently from a centered distribution at large batch size (test uses B = 1024). | 2.3.2 |
| 22 | Aux head class counts hardcoded | `n_words` and `n_speakers` loaded at runtime from `data/manifests/STATS.json`. Never hardcoded. | 2.2.3 |
| 23 | Interleaved batching produced no positives for SupCon | New `ClassAwareBatchSampler`: each batch contains 8 classes × 16 samples = 128 (typical). Defined in `solospeak/data/samplers.py`. | 1.5, 3.3 |
| 24 | ECAPA-TDNN dependency not pinned | Pinned to `speechbrain/spkrec-ecapa-voxceleb`, `speechbrain==1.0.3`, and matching `torch==2.3.1` / `torchaudio==2.3.1`. Cached embeddings to `data/processed/speaker_embeddings.npy`. | 0.1.3, 1.4.3 |
| 25 | FA/hr formula ambiguous (per-user vs per-device) | Both reported. Per-user FA/hr is the primary metric; device FA/hr = sum over enrolled users for N ∈ {1, 4, 8}. | 4.1.2 |
| 26 | Q3 protocol (phonetic neighbors) unclear when real utterances unavailable | Pinned: g2p_en edit-distance ≤ 2 neighbors, prefer real utterances from LibriPhrase. If no real utterance for a neighbor, fall back to Parler-TTS synthesis (with disclosure). Two metrics reported: `q3_rejection_real` and `q3_rejection_synth`. | 4.1.2 |
| 27 | Subgroup eval requires metadata that may not exist | Subgroups split into mandatory (`keyword_syllable_count`) and optional (gender/age/accent). Optional metrics reported on a best-effort basis with explicit "metadata coverage" disclosure. | 4.4 |
| 28 | Ablation plan unrealistic (21 runs × 2 days = 42 GPU-days) | Default: 7 ablations × 1 seed (≈ 14 GPU-days sequential, ≈ 7 with 2 parallel notebooks). 2 extra seeds for `full` and `no_disent` only if time permits. Seed count disclosed in the report. | 4.3.2 |
| 29 | Variable-length training contradicted fixed-length ONNX export | Fixed-length deployment: 1.6 s audio window = 25,600 samples; feature extractors crop/pad the mel axis to 160 frames. No variable-length inference path. | 0.2.3, 1.3 |
| 30 | Window duration inconsistency (1.5 s vs 1.6 s) | Unified audio context to 1.6 s and made the STFT frame adjustment explicit (`center=True` may emit 161 frames before crop). v1's 1.5 s reference removed. | 0.2.3, 1.3 |
| 31 | QAT and ONNX Runtime PTQ paths mixed | Primary path pinned: PyTorch fake-quant QAT fine-tune → save QAT-aware FP32 checkpoint → export FP32 ONNX → ONNX Runtime static PTQ (per-channel weights, per-tensor activations, fusion MLP unquantized). | 3.7, 5.2, Appendix C |
| 32 | "BatchNorm fused into conv weights" not automatic in plain export | Explicit `fuse_model()` contract before export, with nested module paths owned by each block/stem. | 5.1 |
| 33 | Model size FP32 ~18 MB / INT8 ~4.5 MB unverified | Recomputed: deployable params 1,100,897 → FP32 ≈ 4.4 MB weights + ≈ 0.5 MB graph ≈ 5 MB total. INT8 ≈ 1.2 MB weights + 0.5 MB graph ≈ 1.7 MB total. Reported numbers updated. | 5.2 |
| 34 | xRT target inconsistency (0.2 vs 0.08) | HARD gate 0.20 (Samsung spec). STRETCH 0.08. Validation uses MIN/HARD gates only. | Appendix C, 4.5, 5.3 |
| 35 | "Single source of truth" claim was false (referenced external doc) | All architecture content from the previously-external doc is now inlined. This reference IS the single source of truth. | front matter |
| 36 | Production hardening was scope-creep for a hackathon | Phase 6 marked as mostly `🟡 STRETCH`. Each item delivers a single number, document, or stub — not a full implementation. | Phase 6 |
| 37 | Phase dependencies too strict ("do not advance until gate passes") | MIN/TARGET split. Parallel tracks encouraged: build eval/deploy in parallel with training. | 3.0 |
| 38 | Too many tests blocking advancement | Appendix D explicitly marks each test as advancement-blocker or informational. Several v1 blockers (subgroup eval, ablation matrix, full xRT) are now informational. | Appendix D |
| 39 | Old repo could keep unsupported `matchboxnet` path alive | v2.1 explicitly supports only the `SoloSpeakResNet` family. MatchboxNet must be deleted or moved to a research branch, not kept in product config. | 0.2.3 |
| 40 | `features_deploy.py` was specified only in prose | v2.1 includes the deployment log-mel implementation contract, including periodic Hann, reflect padding, HTK mel scale, log clamp, and time-axis crop/pad. | 1.3.1 |
| 41 | `fuse_model()` was required for deployment but not defined in the backbone template | v2.1 adds `fuse_model()` methods to `NormalBlock`, `TransitionBlock`, and `SoloSpeakResNet`. | 2.1.2 |
| 42 | Training-only aux/adversarial heads lacked an owner module | v2.1 defines `TrainingWrapper`, its forward outputs, and the rule that export loads only `wrapper.model` weights. | 3.1.3 |
| 43 | `make eval` and long-running ablations were conflated | v2.1 separates `make eval` (≤ 1 hour KPI/probe/subgroup) from `make ablation` (days, report-critical when compute allows). | 4.3, 4.5 |
| 44 | VoxCeleb fallback sounded secondary despite access risk | v2.1 makes the Common Voice + LibriSpeech fallback a first-class supported data path and forbids VoxCeleb-specific assumptions in training code. | 1.1.4 |
| 45 | Phase-2 acceptance criteria were misnumbered as a 2.5 subsection after Silero-VAD | v2.2 renumbers the overall Phase-2 gate to `2.7 Acceptance criteria — Phase 2`, leaving Silero-VAD's local gate at 2.6.1. | 2.7 |
| 46 | `DualHeadDataset` could emit OOV keyword labels for VoxCeleb/Common Voice transcripts | v2.2 defines `WORD_IGNORE_INDEX = -100`, requires speaker-manifest OOV keywords to use it, and routes aux/adversarial word CE only through valid wake-phrase labels. | 1.2.3, 1.5, 2.3.4, 3.3 |
| 47 | SpeechBrain 1.0.0 with torch 2.3.1 and unconstrained torchaudio was brittle | v2.2 pins `speechbrain==1.0.3` and `torchaudio==2.3.1` next to `torch==2.3.1`; ECAPA notes forbid downgrading or leaving torchaudio unconstrained. | 0.1.3, 1.4.3 |
| 48 | Smoke `train_gsc.csv` had no concrete shape or class contract | v2.2 requires 35 canonical GSC labels × 10 generated clips, a separate `gsc_vocab.json`, and `STATS.json["n_gsc_classes"] == 35`. | 1.2.3, 1.2.5, 3.2 |
| 49 | SileroVAD class constants were missing `ClassVar` in the implementation template | v2.2 imports `ClassVar` and annotates `FRAME_SAMPLES` / `SAMPLE_RATE` as class variables so mypy strict mode matches Appendix B. | 2.6, Appendix B |
| 50 | `EvalSet` dataclass default style differed from the rest of the config appendix | v2.2 uses stdlib `field(default_factory=...)` for the two default `Path` values, mirroring the surrounding default-factory pattern without relying on Pydantic. | Appendix C |
| 51 | LibriPhrase rows had `start_s` / `end_s`, but the canonical manifest schema did not | v2.3 adds `start_s` and `end_s` to every manifest row and requires loaders to crop raw utterance segments before padding. | 1.1.2, 1.2.1 |
| 52 | `test_kpi.csv` and Stage-5 quadrant rows did not identify the enrolled profile separately from the trial utterance | v2.3 adds `profile_id`, `enrolled_user_id`, `enrolled_keyword_text`, `profile_path`, and `trial_source` columns; `speaker_id` / `keyword_text` now always describe the trial audio. | 1.2.1, 1.5.1, 3.6.1 |
| 53 | LMDB packing was operationally required but had no implementation owner | v2.3 adds `scripts/pack_audio_lmdb.py`, a Makefile target, idempotency rules, manifest rewriting rules, and profile-file storage contracts. | 0.1.2, 0.1.3, 1.2.6, Appendix A |
| 54 | Stage 2-4 TA gates referenced task accuracy before the fusion MLP existed | v2.3 defines a fixed pre-fusion proxy protocol using `min(s_c, s_s)` and per-profile `tau_proxy` calibration. | 3.1.6, 3.3, 3.4, 3.5, Appendix B |
| 55 | Combined loss and stage base APIs assumed one flat batch, conflicting with interleaved content/speaker streams | v2.3 adds `StageBatch` and updates `compute_loss()` / `CombinedLoss.forward()` to receive logical streams such as `content` and `speaker`. | 0.2.2, 2.3.4, 3.1.1, Appendix B |
| 56 | Stage 6 said to use Stage-4 aux/adversarial loss while preparing a clean deployable model | v2.3 makes Stage 6 a clean-model QAT distillation stage with optional fusion BCE, explicitly excluding aux/adversarial heads. | 2.3.4, 3.7 |
| 57 | Stage-6 degradation gate could be accidentally disabled by omitting the FP32 ONNX path | v2.3 requires temporary FP32 and INT8 ONNX paths during Stage-6 validation and updates Appendix B wording. | 3.7, 5.3, Appendix B |
| 58 | Bootstrap templates had copy-paste failures | v2.3 fixes `requirements-dev.txt` syntax and Dockerfile inputs so `CMD ["make", "test"]` has `make`, `Makefile`, tests, README, and LICENSE available. | 0.1.3 |
| 59 | Old-code migration risks were scattered across the document | v2.3 adds a migration preflight checklist for stale MatchboxNet files, old config defaults, old eval hard gates, and old single-threshold training APIs. | front matter |
| 60 | Final reading order told the agent to start at `scripts/run_stage.py`, before foundational contracts existed | v2.3 changes the reading order to implement Phase 0 contracts first, then data, models, and training. | End of Reference Document |
| 61 | `pyproject.toml` template could fail editable installs in a flat repo | v2.4 adds `[tool.setuptools.packages.find]` with `solospeak*` and `scripts*` discovery. | 0.1.3 |
| 62 | LibriPhrase speaker-count targets contradicted the clean-only source corpus | v2.4 includes `train-other-500` in the default LibriPhrase recipe and download list, and lowers the expectation to ~2,300 training speakers. | 1.1.2, 1.1.3 |
| 63 | Parler-TTS pin was stale and Parler was treated like arbitrary user voice cloning | v2.4 pins `parler-tts==0.2.3` and adds a Q3 TTS policy separating real right-voice Q3, speaker-verified synthetic Q3, and generic TTS stress rows. | 0.1.3, 3.6.1, 4.1.2 |
| 64 | Silero VAD wrapper assumed an unpinned `h`/`c` ONNX signature | v2.4 requires recording the exact ONNX asset/hash/input names and updates the wrapper contract to the official `input`/`state`/`sr` state+context signature. | 2.6, Appendix B |
| 65 | Stage-5 fusion training could leak profile-specific augmented rows into the dev gate | v2.4 adds profile-level `stage5_train`, `stage5_calibration`, and `stage5_dev` quadrant splits. | 3.6.1 |
| 66 | Stage-5 needed template/calibration math before the files were scheduled to exist | v2.4 moves `enrollment/templates.py` and `enrollment/calibration.py` into Level 4 ownership. | Appendix A |
| 67 | ONNX quantization calibration accepted a single array despite a three-input exported graph | v2.4 requires calibration readers to yield full `mel` / `content_template` / `speaker_template` dictionaries. | 5.2.1, Appendix B |
| 68 | Migration preflight was advisory, so old-code repos could still start from stale APIs | Migration preflight is now a Phase-0 acceptance gate for old-code repos. | front matter, 0.1.5 |
| 69 | MUSAN category usage contradicted itself (`noise` only vs speech/music/noise eval) | MUSAN usage is now split by task: `noise` for training augmentation, all categories for FA/noisy eval, with explicit bucket resolution. | 1.1.1, 1.1.3 |
| 70 | LibriPhrase expected count could mean rows or unique `(phrase, speaker)` pairs | Expected output now means manifest rows / utterance segments, and STATS must report `row_count` separately from `unique_phrase_speaker_pairs`. | 1.1.2, 1.2.5 |
| 71 | Stage-5 calibration split could be misused for profile-specific dev thresholds | Stage-5 calibration is limited to early stopping/global fallback checks; per-profile `tau` is computed only at profile creation. | 3.6.1, 3.6.2 |
| 72 | Q3 generic TTS stress rows could be mixed into the hard right-voice gate | Q3 hard gate now includes only real right-voice rows and speaker-verified synthetic rows; descriptive Parler rows are stress-only. | 3.6.1, 4.1.2 |
| 73 | ORT quantization said "do not quantize fusion MLP" without a stable implementation rule | Default ORT quantization now quantizes Conv nodes only; any future Linear quantization must use stable node prefixes and explicit `nodes_to_exclude`. | 5.2.1 |
| 74 | Eval/deploy config models lacked a typed YAML loading contract | Appendix C now defines `_YamlConfig` and `load_eval_config_bundle()` so scripts do not parse raw dictionaries. | Appendix C |
| 75 | Appendix C could be pasted into `config.py` with a mid-file `from __future__` import | Appendix C now omits the future import and tells agents to merge imports into the top import block. | Appendix C |
| 76 | Strict mypy was required while public contracts still used bare `dict` | Shared aliases now cover dataset items, collated batches, probe outputs, xRT results, and ablation results; Appendix B uses those aliases. | 0.2.2, 1.5.3, Appendix B |
| 77 | Smoke manifest prose said "tiny" but the example mixed smoke and full row counts | Smoke `STATS` now has a separate tiny seven-manifest example, with full-path counts explicitly kept separate. | 1.2.5 |
| 78 | CI ran integration smoke tests without first generating smoke data | The CI template now runs the minimal downloader and smoke manifest generator before integration smoke tests. | 0.1.3 |
| 79 | Q3 hard-gate eligibility depended on an unspecified sidecar | The canonical manifest now carries `q3_gate_eligible`, `synthesis_backend`, and `speaker_verification_score`; Q3 denominator minimums are explicit. | 1.2.1, 3.6.1 |
| 80 | Stage-5 dev accuracy did not say which threshold to use | Stage 5 now uses a single `tau_stage5` chosen on calibration by macro quadrant accuracy and applied unchanged to dev. | 3.6.1 |
| 81 | Final INT8-vs-FP32 degradation was described as optional despite being a release gate | Gate 10 is now required for final Phase-5 release validation and optional only for quick/debug validation. | 5.3, Appendix B, Appendix D |
| 82 | CSV boolean strings could be parsed with `bool("false") == True` | Added `parse_bool_csv()` and required explicit manifest boolean parsing for `q3_gate_eligible`. | 1.2.1, 1.5.3, Appendix B |
| 83 | Stage-5 threshold calibration could be biased by too few eligible Q3 rows | Added a full-path calibration denominator rule before selecting `tau_stage5`. | 3.6.1 |
| 84 | Phase-4 Q3 aggregation could ignore the manifest eligibility column | KPI aggregation now uses `q3_gate_eligible` as the source of truth instead of recomputing verification. | 4.1.2 |
| 85 | Validation gates loaded from YAML could be ignored by `validate()` | `validate()` now accepts a `ValidationGates` object instead of constructing defaults internally. | 5.3, Appendix B |
| 86 | Missing subgroup results were specified as `null` but `SubgroupDict` did not allow `None` | `SubgroupDict` now allows unavailable subgroup entries to be `None`. | 0.2.2, 4.4 |

---

# End of Reference Document

**Total estimated production code:** ≈ 8,000 lines (excluding tests, configs, docs).
**Total estimated test code:** ≈ 3,000 lines.
**Total estimated documentation:** ≈ 2,500 lines markdown.

**Reading order for the implementing agent:**

1. **Read this entire document once** before writing any code. It is self-contained.
2. **For Phase 2 specifically:** treat Section 2.1 as the contract. Do NOT attempt to match the BC-ResNet paper's architecture or parameter counts — they refer to a different design. The variants here (`bcresnet1/5/8/10/16`) are SoloSpeak-specific names for a SoloSpeak-specific architecture defined in 2.1.2. The parameter counts in 2.1.6 are computed analytically from the block formulas in 2.1.2; if your implementation matches the formulas, you will hit the counts within ±2%.
3. **If your implementation's param count is more than ±2% off,** the bug is in your implementation, NOT in the test or the spec. Common causes: (a) using `bias=True` on Conv2d (the spec says `bias=False`), (b) miscounting BatchNorm parameters (each BN contributes `2 × channels`), (c) wrong channel progression (must be `[base, 2*base, 3*base, 4*base]`).
4. **Do not modify tests** unless you can prove this document contradicts itself. If you find a contradiction, document it in the PR description and ask before changing.
5. **Begin implementation with Phase 0 contracts:** config/types/importable stubs first,
   then data manifests/loaders, then models, then training orchestration. Do not start at
   `scripts/run_stage.py` until the foundational contracts it imports already exist.
