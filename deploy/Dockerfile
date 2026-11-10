# syntax=docker/dockerfile:1
# okchat guard API. The image holds code and Python packages only. The release (model files) and the
# Hugging Face cache (16 GB base model) are mounted at run time, never baked in.
#
#   docker build --platform linux/amd64 -f deploy/Dockerfile -t okchat-guard:1 .
#
# Pick TORCH_INDEX for the host's NVIDIA driver (https://pytorch.org/get-started/locally/).
FROM python:3.12-slim

ARG TORCH_VERSION=2.14.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf-cache \
    OKG_PROJECT_DIR=/app \
    OKG_RUN_MODE=full

# lightgbm needs OpenMP
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --index-url "${TORCH_INDEX}" "torch==${TORCH_VERSION}"
COPY deploy/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN useradd --create-home --uid 10001 okg
WORKDIR /app
COPY okg/ okg/
COPY deploy/ deploy/
# mount points: /app/releases (read-only release files), /app/feedback (logs), /data/hf-cache (model cache)
RUN mkdir -p /app/releases /app/feedback /data/hf-cache \
    && chown -R okg:okg /app/releases /app/feedback /data

USER okg
EXPOSE 8080
# The first start downloads Qwen3Guard-Gen-8B (16 GB) and loads it: allow 20 minutes before failing.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/readyz', timeout=4)"
CMD ["uvicorn", "deploy.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "1", "--timeout-graceful-shutdown", "30"]
