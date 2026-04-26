FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    make \
    patch \
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
