#!/usr/bin/env bash
# Build the cloud training image from the repo root and push it to ACR.
# Needs docker already logged in to the target ACR instance.
#
#   REGISTRY=registry.cn-hangzhou.aliyuncs.com ./cloud/dlc/build_push.sh
# Optional env: NAMESPACE(default mxdzlk) IMAGE(default yolo-train) TAG(default YYYYMMDD) PUSH(default 1)
# China mirror for torch wheels: --build-arg PYPI_MIRROR=...
set -euo pipefail

REGISTRY="${REGISTRY:?need REGISTRY, e.g. registry.cn-hangzhou.aliyuncs.com}"
NAMESPACE="${NAMESPACE:-mxdzlk}"
IMAGE="${IMAGE:-yolo-train}"
TAG="${TAG:-$(date +%Y%m%d)}"
FULL="${REGISTRY}/${NAMESPACE}/${IMAGE}:${TAG}"

docker build -f cloud/dlc/Dockerfile -t "$FULL" .
if [ "${PUSH:-1}" = "1" ]; then
    docker push "$FULL"
fi
echo "IMAGE=$FULL"
