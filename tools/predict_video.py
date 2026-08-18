# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run a trained YOLO model on a video and save an annotated copy.

Example:
    python tools/predict_video.py
    python tools/predict_video.py --model weights/best.pt --source weights/test_video/tmp_jump_template_player.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from ultralytics import YOLO


def predict_video(args: argparse.Namespace) -> None:
    """Predict every video frame and write an annotated video."""
    source = Path(args.source)
    model_path = Path(args.model)
    if not source.exists():
        raise SystemExit(f"Video not found: {source}")
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    output = Path(args.output) if args.output else source.with_name(f"{source.stem}_pred.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Cannot create output video: {output}")

    model = YOLO(str(model_path))
    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            result = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            annotated = result.plot(labels=True, conf=True)
            writer.write(annotated)
            frame_index += 1
            if args.show:
                cv2.imshow("YOLO video prediction", annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            if frame_index % 30 == 0:
                print(f"processed {frame_index} frames")
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()
    print(f"Saved {frame_index} frames to {output.resolve()}")


def main() -> None:
    """Parse arguments and run video prediction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="weights/best.pt", help="trained YOLO weights")
    parser.add_argument("--source", default="weights/test_video/tmp_jump_template_player.mp4", help="input video")
    parser.add_argument("--output", default=None, help="output video; defaults to <source>_pred.mp4")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--device", default=None, help="device, for example 0 or cpu")
    parser.add_argument("--show", action="store_true", help="show a live preview window")
    predict_video(parser.parse_args())


if __name__ == "__main__":
    main()
