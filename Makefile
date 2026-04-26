.PHONY: install install-dev install-optional test test-unit test-integration test-slow test-all test-optional lint typecheck docs-check format download-data download-data-smoke prepare-manifests prepare-manifests-smoke pack-lmdb precompute-speaker-embeds eval ablation export clean

PYTEST_OPTS ?= -p no:capture

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

install-optional:
	pip install -e ".[tts,demo]"

test: lint typecheck test-unit

test-unit:
	pytest $(PYTEST_OPTS) tests/unit -v -n auto -m "not slow and not data and not tts and not demo"

test-integration:
	pytest $(PYTEST_OPTS) tests/integration -v -m "not slow and not data and not tts and not demo"

test-slow:
	pytest $(PYTEST_OPTS) tests/integration -v -m "slow or data"

test-optional:
	pytest $(PYTEST_OPTS) tests -v -m "tts or demo"

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
