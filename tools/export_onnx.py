"""Export the trained YOLO checkpoint to ONNX for the C# runtime.

Run from the ultralytics checkout, for example:
    .venv\\Scripts\\python.exe tools\\export_onnx.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("weights/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("weights/best.onnx"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    exported = Path(
        YOLO(str(args.model)).export(
            format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=True, dynamic=False
        )
    )
    if exported.resolve() != args.output.resolve():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(exported.read_bytes())
    print(f"Exported {args.model} -> {args.output}")


if __name__ == "__main__":
    main()
