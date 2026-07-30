# GemmaApollo model server: Whisper + Qwen S2L behind HTTP (scribe
# model-server), CUDA runtime. Build from the repo root context:
#   docker compose -f docker/compose.yaml up --build
#
# The pip cu128 torch wheels bundle cuBLAS/cuDNN, and S2LEngine imports
# torch before CTranslate2 precisely so those libs are loaded first — the
# plain (non-cudnn) runtime base is sufficient.
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_LINK_MODE=copy

# deps first for layer caching — the s2l extra pulls ~3 GB of cu128 wheels
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra s2l

COPY src ./src
RUN uv sync --frozen --extra s2l

# HF weights survive rebuilds via the compose volume; the central dataset
# (/app/data/sessions, POST /log) is a host bind mount.
EXPOSE 8018
CMD ["uv", "run", "--no-sync", "scribe", "model-server", "--engine", "s2l", \
     "--host", "0.0.0.0", "--port", "8018", "--preload"]
