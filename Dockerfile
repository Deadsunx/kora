# Kora serving image.
#
# Built on python:3.13-slim rather than an nvidia/cuda base image: the PyTorch
# cu128 wheels bundle the CUDA runtime libraries they need, so a CUDA base layer
# would ship a second copy of them. The host still needs an NVIDIA driver and
# the container toolkit -- run with `--gpus all`.
#
# The image is large (roughly 7 GB), and most of that is torch plus its bundled
# CUDA libraries. That is the honest cost of serving a quantised 4B model; a
# smaller image would mean a smaller model, not a cleverer Dockerfile.
#
# Model weights are NOT baked in. They are ~4 GB of Hugging Face cache and they
# change independently of this code, so they are mounted at runtime. A cold
# container with an empty cache will download them on first start.

FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    KORA_DATA_DIR=/data \
    KORA_MODELS_DIR=/models \
    KORA_RUNS_DIR=/runs

# libgomp1 is required by faiss; the rest of the runtime is pure Python wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch first, from the CUDA index, in its own layer. It is by far the largest
# dependency and it changes least often, so a code edit does not re-download it.
RUN pip install --index-url https://download.pytorch.org/whl/cu128 torch

# Then the dependency set, still without the source tree, so that editing
# src/ does not invalidate the dependency layer.
COPY pyproject.toml README.md ./
RUN mkdir -p src/kora && touch src/kora/__init__.py \
    && pip install -e ".[serve]"

COPY src/ ./src/
COPY configs/ ./configs/
RUN pip install --no-deps -e .

# Non-root. The mounted volumes must be readable by this uid.
RUN useradd --create-home --uid 10001 kora \
    && mkdir -p /cache /data /models /runs \
    && chown -R kora:kora /cache /runs /app
USER kora

EXPOSE 8000

# Reports which config is serving, so an unhealthy container is distinguishable
# from one serving the wrong system. The long start period covers model loading:
# warming the generator takes tens of seconds on a cold CUDA context.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

ENV KORA_CONFIG=/app/configs/experiments/07_rerank_fast.yaml

CMD ["uvicorn", "kora.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
