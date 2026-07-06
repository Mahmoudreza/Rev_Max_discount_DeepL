# Dockerfile — Linux/CPU fallback for evaluation and figures.
# NOTE: This is NOT recommended for training (no MPS/CUDA; Phase 1 imitation
# on CPU takes ~3× longer than on Apple Silicon MPS).
# Primary use: CI, reproducing eval numbers, generating figures.
#
# Build:  docker build -t revmax .
# Run:    docker run -it --rm -v $(pwd)/results:/app/results revmax
#         docker run --rm revmax bash scripts/smoke_test.sh

FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source (respects .dockerignore)
COPY . .

# Run tests at build time; failures are non-fatal (|| true) so docker build
# still succeeds if a test needs a checkpoint that isn't included in image.
RUN pytest tests/ -q --tb=short || true

CMD ["/bin/bash"]
