.PHONY: install install-dev test test-unit test-integration test-slow lint typecheck format clean \
        download-data prepare-manifests eval ablation export

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,tts]"
	pre-commit install

test: test-unit test-integration

test-unit:
	pytest tests/unit -v -n auto -m "not slow"

test-integration:
	pytest tests/integration -v -m "not slow"

test-slow:
	pytest tests/integration -v -m slow

lint:
	ruff check solospeak tests scripts

typecheck:
	mypy solospeak

format:
	ruff format solospeak tests scripts

download-data:
	python -m scripts.download_datasets

prepare-manifests:
	python -m scripts.prepare_manifests

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
