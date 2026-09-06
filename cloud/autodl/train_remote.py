# cloud/autodl/train_remote.py
r"""在 AutoDL 实例上跑的 py3.8 训练模板（上传到远端后执行）。

远端环境**没有 pip 版 ultralytics**，必须把它指向项目目录：
  首行 `sys.path.insert(0, '/root/ultralytics-YOLO26')`（或 `cd /root/ultralytics-YOLO26 && python`）。
远端 base 是 **py3.8**（不要把本地 py3.12+ 工具传上来跑）。

读取 env（或 argv）：
  TRAIN_BASE   基础权重 best.pt（远端路径）
  TRAIN_DATA   dataset.yaml（远端路径）
  TRAIN_NAME   run 名（默认 unified-v1）
  TRAIN_EPOCHS 默认 100
远端跑：
  nohup /root/miniconda3/bin/python /root/autodl-tmp/train_remote.py > /root/autodl-tmp/train.log 2>&1 &
完成后日志出现 TRAIN_DONE。

详见 `docs/autodl_training.md`。
"""
import os
import sys

sys.path.insert(0, "/root/ultralytics-YOLO26")

from ultralytics import YOLO


def _get(name: str, default: str) -> str:
    return os.environ.get(name) or default


def main() -> int:
    """Run training from env-configured paths."""
    base = _get("TRAIN_BASE", "/root/ultralytics-YOLO26/best.pt")
    data = _get("TRAIN_DATA", "/root/ultralytics-YOLO26/datasets/unified_20260906/dataset.yaml")
    name = _get("TRAIN_NAME", "unified-v1")
    epochs = int(_get("TRAIN_EPOCHS", "100"))

    model = YOLO(base)
    model.train(
        data=data,
        epochs=epochs, imgsz=640, batch=-1, device=0,
        project="/root/ultralytics-YOLO26/runs", name=name,
        cache=True, patience=30, workers=8, verbose=True,
    )
    print("TRAIN_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
