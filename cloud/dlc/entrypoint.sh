#!/usr/bin/env bash
# Entrypoint of the cloud training image: reproduce local tools/yolo_data.py train on
# a PAI-DLC GPU node and copy the resulting weights up to the OSS output mount.
#
# Required env (set in the DLC job): DATA_DIR, BASE_PT, RUN_NAME, OUT_DIR
# Optional env: EPOCHS(80) IMGSZ(640) BATCH(-1) DEVICE(0) EXPORT_ONNX(1)
set -euo pipefail

: "${DATA_DIR:?need DATA_DIR: mounted dataset dir that contains dataset.yaml}"
: "${BASE_PT:?need BASE_PT: path to base .pt weights}"
: "${RUN_NAME:?need RUN_NAME: run/exp name}"
: "${OUT_DIR:?need OUT_DIR: writable OSS-mounted output dir}"
EPOCHS="${EPOCHS:-80}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:--1}"
DEVICE="${DEVICE:-0}"
EXPORT_ONNX="${EXPORT_ONNX:-1}"

# PAI may inject RANK=0 into a one-worker job even though no DDP process group
# is started. Ultralytics then expects a distributed sampler and crashes on the
# regular RandomSampler. Force the one-worker case back to single-GPU mode.
if [ "${WORLD_SIZE:-1}" = "1" ] || { [ -z "${WORLD_SIZE:-}" ] && [ "${RANK:-}" = "0" ]; }; then
  export RANK=-1 LOCAL_RANK=-1 WORLD_SIZE=1
fi
CACHE="${CACHE:-}"
PATIENCE="${PATIENCE:-}"
CLOSE_MOSAIC="${CLOSE_MOSAIC:-}"
COS_LR="${COS_LR:-0}"
WORKERS="${WORKERS:-}"
PRINT_ONLY="${PRINT_ONLY:-0}"

# Assemble optional train hyperparams (only emit flags, never bare values).
extra_args=()
[ -n "$CACHE" ] && extra_args+=(--cache "$CACHE")
[ -n "$PATIENCE" ] && extra_args+=(--patience "$PATIENCE")
[ -n "$CLOSE_MOSAIC" ] && extra_args+=(--close-mosaic "$CLOSE_MOSAIC")
[ "$COS_LR" = "1" ] && extra_args+=(--cos-lr)
[ -n "$WORKERS" ] && extra_args+=(--workers "$WORKERS")

if [ "$PRINT_ONLY" = "1" ]; then
  echo "PRINT python3 tools/yolo_data.py train --dataset \"$DATA_DIR\" --model \"$BASE_PT\" --epochs \"$EPOCHS\" --imgsz \"$IMGSZ\" --batch \"$BATCH\" --device \"$DEVICE\" --project \"$OUT_DIR/runs\" --name \"$RUN_NAME\" ${extra_args[*]}"
  exit 0
fi

cd /workspace
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
[ -f "$DATA_DIR/dataset.yaml" ] || { echo "dataset.yaml not found under DATA_DIR=$DATA_DIR" >&2; exit 2; }
[ -f "$BASE_PT" ] || { echo "base weights not found: BASE_PT=$BASE_PT" >&2; exit 2; }

echo "== cloud train run=${RUN_NAME} epochs=${EPOCHS} imgsz=${IMGSZ} batch=${BATCH} device=${DEVICE}"
echo "   data = ${DATA_DIR}"
echo "   base = ${BASE_PT}"

"$PY" tools/yolo_data.py train \
  --dataset "$DATA_DIR" \
  --model "$BASE_PT" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --project "$OUT_DIR/runs" \
  --name "$RUN_NAME" \
  "${extra_args[@]}"

mkdir -p "$OUT_DIR"
cp -f "$OUT_DIR/runs/$RUN_NAME/weights/best.pt" "$OUT_DIR/best.pt"

if [ "$EXPORT_ONNX" = "1" ]; then
  echo "== exporting ONNX =="
  "$PY" tools/export_onnx.py \
    --model "$OUT_DIR/best.pt" \
    --output "$OUT_DIR/best.onnx" --imgsz "$IMGSZ"
fi

# Write best.names (utf-8, index line == class id) for the C# runtime.
"$PY" - "$DATA_DIR" "$OUT_DIR" <<'PYEOF'
import sys
from pathlib import Path

import yaml

data_dir, out_dir = sys.argv[1], sys.argv[2]
names = yaml.safe_load(Path(data_dir, "dataset.yaml").read_text(encoding="utf-8"))["names"]
Path(out_dir, "best.names").write_text(
    "".join(f"{names[i]}\n" for i in range(len(names))), encoding="utf-8"
)
print(f"wrote {out_dir}/best.names ({len(names)} classes)")
PYEOF

# Force JindoFuse flush so results are visible in OSS before the job stops billing.
ls -R "$OUT_DIR" >/dev/null
touch "$OUT_DIR/$RUN_NAME.done.marker"

echo "== DONE run=${RUN_NAME} =="
ls -l "$OUT_DIR"
