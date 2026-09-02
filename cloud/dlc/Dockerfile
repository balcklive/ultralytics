# Cloud YOLO training image for Alibaba Cloud PAI-DLC GPU jobs.
#
# Strategy: python:3.12-slim + uv virtualenv; torch/CUDA comes from PyPI Linux GPU
# wheels (self-contained CUDA runtime libs; the DLC GPU node only needs the NVIDIA
# driver). No dependency on PAI official image tags. The image bundles the repo
# source + deps only -- dataset / base weights / outputs all arrive via the OSS
# read-write mount handled by DLC (see docs/cloud_dlc_training.md).
#
# Build from the ultralytics checkout (repo root is the build context):
#   docker build -f cloud/dlc/Dockerfile -t <registry>/<ns>/yolo-train:<date> .
# China mirror for slow downloads:
#   docker build --build-arg PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple ...

ARG BASE_IMAGE=python:3.12-slim
ARG UV_VERSION=0.6.12
FROM ${BASE_IMAGE}

ARG PYPI_MIRROR=
ARG UV_VERSION

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    YOLO_AUTOINSTALL=false

# opencv needs libGL/glib at import; git/curl for convenience; curl not required but small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgl1 libglib2.0-0 git curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir ${PYPI_MIRROR:+--index-url ${PYPI_MIRROR}} "uv==${UV_VERSION}"

WORKDIR /workspace

# Copy manifests first so dependency layers cache across source edits.
COPY pyproject.toml uv.lock .python-version README.md LICENSE ./
# Source subset required to run tools/yolo_data.py train (no data/weights/runs).
COPY ultralytics ultralytics/
COPY tools tools/

# Resolve deps and build the project env. --frozen fails on lock files resolved only
# for another platform (e.g. the win32 dev machine); fall back to resolving for Linux.
RUN if [ -n "${PYPI_MIRROR}" ]; then export UV_DEFAULT_INDEX="${PYPI_MIRROR}"; fi; \
    uv sync --no-dev --frozen || uv sync --no-dev

COPY cloud/dlc/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]
