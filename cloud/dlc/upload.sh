#!/usr/bin/env bash
# Upload a prepared dataset dir and a base weight to OSS under the mxdzlk/ prefix.
# Needs ossutil installed & configured for the target account (aliyun ossutil config).
#
#   BUCKET=mxdzlk-bucket ./cloud/dlc/upload.sh <round> <dataset_dir> <base_pt>
# Example:
#   BUCKET=mxdzlk-data ./cloud/dlc/upload.sh 20260902-r1 artifacts/cloud_round weights/20260902/best.pt
# Optional env: OSSUTIL(default ossutil) OSS_PREFIX(default oss://<bucket>/mxdzlk)
set -euo pipefail

OSSUTIL="${OSSUTIL:-ossutil}"
BUCKET="${BUCKET:?need BUCKET (bucket name); or set OSS_PREFIX to a full oss:// prefix}"
ROUND="${1:?usage: upload.sh <round> <dataset_dir> <base_pt>}"
DATASET_DIR="${2:?usage: upload.sh <round> <dataset_dir> <base_pt>}"
BASE_PT="${3:?usage: upload.sh <round> <dataset_dir> <base_pt>}"

OSS_PREFIX="${OSS_PREFIX:-oss://${BUCKET}/mxdzlk}"

[ -d "$DATASET_DIR" ] || { echo "dataset dir not found: $DATASET_DIR" >&2; exit 2; }
[ -f "$BASE_PT" ] || { echo "base weights not found: $BASE_PT" >&2; exit 2; }

# Strip ultralytics *.cache files before upload: they embed local absolute image
# paths (Windows D:\...) that are invalid in the Linux container. Ultralytics
# rebuilds the cache on first scan with container-relative paths.
find "$DATASET_DIR" -type f -name '*.cache' -delete

echo "upload dataset  $DATASET_DIR -> $OSS_PREFIX/datasets/$ROUND/"
"$OSSUTIL" cp -r -f "$DATASET_DIR" "$OSS_PREFIX/datasets/$ROUND/"
echo "upload base     $BASE_PT -> $OSS_PREFIX/models/base/"
"$OSSUTIL" cp -f "$BASE_PT" "$OSS_PREFIX/models/base/$(basename "$BASE_PT")"
echo "DONE. Mount $OSS_PREFIX read-write at /mnt/data in the DLC job; then set:"
echo "  DATA_DIR=/mnt/data/datasets/$ROUND"
echo "  BASE_PT=/mnt/data/models/base/$(basename "$BASE_PT")"
echo "  OUT_DIR=/mnt/data/out/$ROUND"
