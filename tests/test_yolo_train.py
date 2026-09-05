from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
_spec = importlib.util.spec_from_file_location("yolo_data", _TOOLS / "yolo_data.py")
yolo_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(yolo_data)


def test_build_train_kwargs_receives_base_args():
    args = Namespace(dataset=Path("data/wgc_review"), epochs=80, imgsz=640, batch=-1,
                     device="cpu", project="runs/train", name="full-v1")
    kwargs = yolo_data.build_train_kwargs(args)
    assert kwargs["data"].endswith("dataset.yaml")
    assert kwargs["epochs"] == 80 and kwargs["imgsz"] == 640 and kwargs["device"] == "cpu"


def test_build_train_kwargs_omits_unset_optional():
    args = Namespace(dataset=Path("d"), epochs=5, imgsz=640, batch=-1, device="cpu",
                     project="p", name="n", cache=None, patience=None,
                     close_mosaic=None, cos_lr=False, workers=None)
    kwargs = yolo_data.build_train_kwargs(args)
    for field in ("cache", "patience", "close_mosaic", "workers"):
        assert field not in kwargs
    assert "cos_lr" not in kwargs


def test_build_train_kwargs_forwards_set_optional():
    args = Namespace(dataset=Path("d"), epochs=5, imgsz=640, batch=-1, device="cpu",
                     project="p", name="n", cache="ram", patience=30,
                     close_mosaic=0, cos_lr=True, workers=8)
    kwargs = yolo_data.build_train_kwargs(args)
    assert kwargs["cache"] == "ram" and kwargs["patience"] == 30
    assert kwargs["close_mosaic"] == 0 and kwargs["cos_lr"] is True and kwargs["workers"] == 8
