# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
r"""Prepare and train a two-class YOLO dataset from an older monster model.

Examples:
    python tools/yolo_data.py auto-label --images data/images --old-model ..\maoxiandao\bot\resource\bundles\models\best.pt --output data/old_yolo_dataset --review-output data/old_yolo_review
    python tools/yolo_data.py annotate --dataset datasets\monster_player --split train
    python tools/yolo_data.py train --dataset datasets\monster_player --model yolo26n.pt
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = ("monster", "player")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def image_files(path: Path) -> list[Path]:
    """Return image files below a directory in stable order."""
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def yolo_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    """Convert an xyxy pixel box to normalized YOLO cx, cy, width, height."""
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
    y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)


def read_labels(path: Path, width: int, height: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    """Read YOLO labels and convert them to pixel xyxy boxes."""
    labels = []
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        cls, cx, cy, w, h = map(float, fields)
        x1, y1 = (cx - w / 2) * width, (cy - h / 2) * height
        x2, y2 = (cx + w / 2) * width, (cy + h / 2) * height
        labels.append((int(cls), (round(x1), round(y1), round(x2), round(y2))))
    return labels


def write_labels(path: Path, labels: list[tuple[int, tuple[int, int, int, int]]], width: int, height: int) -> None:
    """Write pixel xyxy boxes in normalized YOLO format."""
    lines = []
    for cls, box in labels:
        cx, cy, w, h = yolo_box(tuple(map(float, box)), width, height)
        if w > 0 and h > 0:
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def auto_label(args: argparse.Namespace) -> None:
    """Run the old model and build a dataset preserving its original classes."""
    from ultralytics import YOLO

    sources = image_files(Path(args.images))
    if not sources:
        raise SystemExit(f"No images found: {args.images}")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output is not empty: {output}. Use --overwrite to rebuild it.")
    model = YOLO(str(args.old_model))
    model_names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    review_output = Path(args.review_output) if args.review_output else None
    if review_output and review_output.exists() and any(review_output.iterdir()) and not args.overwrite:
        raise SystemExit(f"Review output is not empty: {review_output}. Use --overwrite to rebuild it.")
    rng = random.Random(args.seed)
    rng.shuffle(sources)
    val_count = round(len(sources) * args.val_fraction)
    val_paths = set(sources[:val_count])
    for source in sources:
        split = "val" if source in val_paths else "train"
        target_image = output / "images" / split / source.name
        target_label = output / "labels" / split / f"{source.stem}.txt"
        target_image.parent.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(source))
        if image is None:
            print(f"[skip] unreadable image: {source}")
            continue
        height, width = image.shape[:2]
        result = model.predict(source=image, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        labels = []
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy().tolist()
            classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
            confidences = result.boxes.conf.cpu().numpy().tolist()
            for box, cls, confidence in zip(boxes, classes, confidences):
                if cls not in model_names:
                    print(f"[warning] unknown old-model class {cls} in {source.name}; skipped")
                    continue
                pixel_box = tuple(map(round, box))
                labels.append((cls, pixel_box))
                detections.append((cls, pixel_box, confidence))
        shutil.copy2(source, target_image)
        write_labels(target_label, labels, width, height)
        if review_output:
            save_review_images(review_output, source.name, image, detections, model_names)
        print(f"[{split}] {source.name}: {len(labels)} detection(s)")
    (output / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {cls}: {name}\n" for cls, name in sorted(model_names.items())),
        encoding="utf-8",
    )
    review_message = f"\nReview images: {review_output.resolve()}" if review_output else ""
    print(f"Dataset created at {output.resolve()}{review_message}")


def save_review_images(
    review_output: Path,
    image_name: str,
    image: np.ndarray,
    detections: list[tuple[int, tuple[int, int, int, int], float]],
    names: dict[int, str],
) -> None:
    """Save an annotated source image and one crop per old-model detection."""
    annotated = image.copy()
    crop_dir = review_output / "crops"
    for index, (cls, (x1, y1, x2, y2), confidence) in enumerate(detections):
        color = (0, 220, 0)
        label = f"{cls}:{names[cls]} {confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        crop = image[max(0, y1) : min(image.shape[0], y2), max(0, x1) : min(image.shape[1], x2)]
        if crop.size:
            class_dir = crop_dir / f"{cls}_{names[cls]}"
            class_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(class_dir / f"{Path(image_name).stem}_{index:03d}.jpg"), crop)
    review_output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(review_output / image_name), annotated)


class OldLabelReviewer:
    """Review legacy detection crops and discard rejected boxes."""

    def __init__(self, dataset: Path, split: str, class_id: int) -> None:
        splits = ("train", "val") if split == "all" else (split,)
        self.images = [image for current in splits for image in image_files(dataset / "images" / current)]
        self.dataset = dataset
        self.labels_root = dataset / "labels"
        self.class_id = class_id
        self.crop_dir = dataset.parent / "old_yolo_review" / "crops" / f"{class_id}_lvwoniu_review"
        self.items: list[tuple[Path, Path, tuple[int, int, int, int]]] = []
        self.index = 0
        self.window = f"Review crops class {class_id} | k=keep, d=delete, n/space=next, q=quit"

    def label_path(self) -> Path:
        """Return the label path matching the current image."""
        split = self.images[self.index].parent.name
        return self.labels_root / split / f"{self.images[self.index].stem}.txt"

    def clean_labels(self) -> None:
        """Remove every class except the selected legacy class before review."""
        cleaned = 0
        for split in ("train", "val"):
            for path in (self.labels_root / split).glob("*.txt"):
                image = next(
                    (item for item in self.images if item.parent.name == split and item.stem == path.stem), None
                )
                if image is None:
                    continue
                frame = cv2.imread(str(image))
                if frame is None:
                    continue
                height, width = frame.shape[:2]
                labels = [label for label in read_labels(path, width, height) if label[0] == self.class_id]
                write_labels(path, labels, width, height)
                cleaned += 1
        print(f"Cleaned {cleaned} label files; kept class {self.class_id} only")

    def prepare_crops(self) -> None:
        """Create one review crop for every selected-class label."""
        self.crop_dir.mkdir(parents=True, exist_ok=True)
        self.items = []
        manifest = []
        for image_path in self.images:
            label_path = self.labels_root / image_path.parent.name / f"{image_path.stem}.txt"
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            height, width = frame.shape[:2]
            for box_index, (_class_id, box) in enumerate(read_labels(label_path, width, height)):
                x1, y1, x2, y2 = box
                crop = frame[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]
                if crop.size == 0:
                    continue
                crop_path = self.crop_dir / f"{image_path.parent.name}_{image_path.stem}_{box_index:04d}.jpg"
                cv2.imwrite(str(crop_path), crop)
                self.items.append((crop_path, label_path, box))
                manifest.append(
                    {
                        "crop": crop_path.name,
                        "label": str(label_path.relative_to(self.dataset)),
                        "class_id": self.class_id,
                        "box": list(box),
                    }
                )
        (self.crop_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Review crops created: {len(self.items)}")

    def remove_current_label(self) -> None:
        """Remove the current crop's matching box from its source label file."""
        crop_path, label_path, box = self.items[self.index]
        source_candidates = [
            image
            for image in self.images
            if image.parent.name == label_path.parent.name and image.stem == label_path.stem
        ]
        if not source_candidates:
            return
        source = cv2.imread(str(source_candidates[0]))
        if source is None:
            return
        height, width = source.shape[:2]
        labels = read_labels(label_path, width, height)
        for label_index, label in enumerate(labels):
            if label[0] == self.class_id and label[1] == box:
                labels.pop(label_index)
                break
        write_labels(label_path, labels, width, height)
        crop_path.unlink(missing_ok=True)
        print(f"deleted {crop_path.name}")

    def run(self) -> None:
        """Clean labels and run the review loop."""
        if not self.images:
            raise SystemExit("No images found in the selected split")
        self.clean_labels()
        self.prepare_crops()
        if not self.items:
            print("No selected-class crops to review")
            return
        cv2.namedWindow(self.window)
        while True:
            crop_path, _label_path, _box = self.items[self.index]
            crop = cv2.imread(str(crop_path))
            if crop is None:
                self.items.pop(self.index)
                if not self.items:
                    break
                self.index = min(self.index, len(self.items) - 1)
                continue
            max_size = max(crop.shape[:2])
            if max_size > 900:
                scale = 900 / max_size
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            canvas = cv2.copyMakeBorder(crop, 45, 0, 0, 0, cv2.BORDER_CONSTANT, value=(35, 35, 35))
            cv2.putText(
                canvas,
                f"{self.index + 1}/{len(self.items)}  {crop_path.name}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.imshow(self.window, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("d"):
                self.remove_current_label()
                self.items.pop(self.index)
                if not self.items:
                    break
                self.index = min(self.index, len(self.items) - 1)
            elif key in (ord("k"), ord("n"), 32):
                self.index = min(self.index + 1, len(self.items) - 1)
            elif key == ord("p"):
                self.index = max(self.index - 1, 0)
            elif key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return
        cv2.destroyAllWindows()


def apply_crop_review(args: argparse.Namespace) -> None:
    """Update labels from the crop files that remain after manual review."""
    crops = Path(args.crops)
    manifest_path = crops / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing crop manifest: {manifest_path}. Run review-old once to create it.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = {}
    for item in manifest:
        if item["class_id"] == args.class_id:
            grouped.setdefault(item["label"], []).append(item)
    removed = 0
    kept = 0
    for relative_label, items in grouped.items():
        label_path = Path(args.dataset) / relative_label
        source_candidates = list((Path(args.dataset) / "images" / label_path.parent.name).glob(f"{label_path.stem}.*"))
        if not source_candidates:
            continue
        source = cv2.imread(str(source_candidates[0]))
        if source is None:
            continue
        height, width = source.shape[:2]
        current = read_labels(label_path, width, height)
        accepted = [tuple(item["box"]) for item in items if (crops / item["crop"]).exists()]
        remaining = list(current)
        filtered = []
        for label in remaining:
            if label[0] == args.class_id and label[1] in accepted:
                filtered.append(label)
                accepted.remove(label[1])
            else:
                removed += 1
        kept += len(filtered)
        write_labels(label_path, filtered, width, height)
    print(f"Applied crop review: kept {kept} boxes, removed {removed} boxes")


class Annotator:
    """Small OpenCV annotator for adding player boxes to generated labels."""

    def __init__(self, dataset: Path, split: str) -> None:
        self.images = image_files(dataset / "images" / split)
        self.labels_dir = dataset / "labels" / split
        self.index = 0
        self.labels: list[tuple[int, tuple[int, int, int, int]]] = []
        self.image: np.ndarray | None = None
        self.start: tuple[int, int] | None = None
        self.window = "YOLO annotator | drag=player, right-click=delete, s=save, n/p=next, q=quit"

    def load(self) -> None:
        """Load the current image and labels."""
        self.image = cv2.imread(str(self.images[self.index]))
        if self.image is None:
            raise RuntimeError(f"Cannot read {self.images[self.index]}")
        h, w = self.image.shape[:2]
        self.labels = read_labels(self.labels_dir / f"{self.images[self.index].stem}.txt", w, h)

    def save(self) -> None:
        """Save the current labels."""
        assert self.image is not None
        h, w = self.image.shape[:2]
        write_labels(self.labels_dir / f"{self.images[self.index].stem}.txt", self.labels, w, h)
        print(f"saved {self.images[self.index].name}: {len(self.labels)} box(es)")

    def draw(self) -> np.ndarray:
        """Render labels and status text."""
        assert self.image is not None
        canvas = self.image.copy()
        for cls, (x1, y1, x2, y2) in self.labels:
            color = (0, 220, 0) if cls == 0 else (0, 120, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, CLASS_NAMES[cls], (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(
            canvas,
            f"{self.index + 1}/{len(self.images)}  monsters={sum(c == 0 for c, _ in self.labels)}  players={sum(c == 1 for c, _ in self.labels)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        return canvas

    def mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        """Handle drawing and deletion."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.start:
            x1, y1 = self.start
            box = (min(x1, x), min(y1, y), max(x1, x), max(y1, y))
            if box[2] - box[0] >= 3 and box[3] - box[1] >= 3:
                self.labels.append((1, box))
            self.start = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i, (_cls, (x1, y1, x2, y2)) in reversed(list(enumerate(self.labels))):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.labels.pop(i)
                    break

    def run(self) -> None:
        """Run the annotation loop."""
        if not self.images:
            raise SystemExit("No images found in the selected split")
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.mouse)
        while True:
            self.load()
            while True:
                cv2.imshow(self.window, self.draw())
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("s"),):
                    self.save()
                elif key in (ord("n"), 32):
                    self.save()
                    self.index = min(self.index + 1, len(self.images) - 1)
                    break
                elif key == ord("p"):
                    self.save()
                    self.index = max(self.index - 1, 0)
                    break
                elif key == ord("c"):
                    self.labels = [(cls, box) for cls, box in self.labels if cls != 1]
                elif key in (ord("q"), 27):
                    self.save()
                    cv2.destroyAllWindows()
                    return


def train(args: argparse.Namespace) -> None:
    """Start current Ultralytics training with the generated dataset."""
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(Path(args.dataset).resolve() / "dataset.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


def main() -> None:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    auto = sub.add_parser("auto-label", help="use an old YOLO model to label monsters")
    auto.add_argument("--images", type=Path, default=Path("data/images"))
    auto.add_argument("--old-model", type=Path, default=Path(r"..\maoxiandao\bot\resource\bundles\models\best.pt"))
    auto.add_argument("--output", type=Path, default=Path("data/old_yolo_dataset"))
    auto.add_argument("--review-output", type=Path, default=Path("data/old_yolo_review"))
    auto.add_argument("--conf", type=float, default=0.25)
    auto.add_argument("--imgsz", type=int, default=640)
    auto.add_argument("--val-fraction", type=float, default=0.2)
    auto.add_argument("--seed", type=int, default=42)
    auto.add_argument("--overwrite", action="store_true")
    auto.set_defaults(func=auto_label)
    annotate = sub.add_parser("annotate", help="draw player boxes on generated images")
    annotate.add_argument("--dataset", type=Path, required=True)
    annotate.add_argument("--split", choices=("train", "val"), default="train")
    annotate.set_defaults(func=lambda a: Annotator(a.dataset, a.split).run())
    review = sub.add_parser("review-old", help="keep and review one legacy model class")
    review.add_argument("--dataset", type=Path, required=True)
    review.add_argument("--class-id", type=int, default=9)
    review.add_argument("--split", choices=("all", "train", "val"), default="all")
    review.set_defaults(func=lambda a: OldLabelReviewer(a.dataset, a.split, a.class_id).run())
    apply_review = sub.add_parser("apply-crop-review", help="apply manually deleted crop files to labels")
    apply_review.add_argument("--dataset", type=Path, required=True)
    apply_review.add_argument("--crops", type=Path, default=Path("data/old_yolo_review/crops/9_lvwoniu_review"))
    apply_review.add_argument("--class-id", type=int, default=9)
    apply_review.set_defaults(func=apply_crop_review)
    fitting = sub.add_parser("train", help="train the current Ultralytics YOLO")
    fitting.add_argument("--dataset", type=Path, required=True)
    fitting.add_argument("--model", default="yolo26n.pt")
    fitting.add_argument("--epochs", type=int, default=100)
    fitting.add_argument("--imgsz", type=int, default=640)
    fitting.add_argument("--batch", type=int, default=-1)
    fitting.add_argument("--device", default=None)
    fitting.add_argument("--project", default="runs/train")
    fitting.add_argument("--name", default="monster-player")
    fitting.set_defaults(func=train)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
