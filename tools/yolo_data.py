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
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = ("monster", "player")
ENGLISH_MONSTER_NAMES = {
    "特殊小石球": "special_small_stone_ball", "蜗牛": "snail", "蓝蜗牛": "blue_snail", "蘑菇仔": "mushroom_boy",
    "木妖": "tree_stump", "红蜗牛": "red_snail", "花蘑菇": "orange_mushroom", "绿水灵": "slime", "猪猪": "pig",
    "铁甲猪": "iron_boar", "蘑菇王": "mushroom_king", "蝴蝶精": "fairy", "漂漂猪": "ribbon_pig", "蓝蘑菇": "blue_mushroom",
    "绿蘑菇": "green_mushroom", "斧木妖": "axe_stump", "刺蘑菇": "spike_mushroom", "猴子": "monkey", "无魂猴": "soulless_monkey",
    "风独眼兽": "wind_eye_beast", "巫婆": "witch", "黑木妖": "black_stump", "冰独眼兽": "ice_eye_beast", "黑斧木妖": "black_axe_stump",
    "野猪": "wild_boar", "古木妖": "ancient_stump", "木面怪人": "wood_mask_man", "石面怪人": "stone_mask_man", "石膏犬": "plaster_hound",
    "木乃伊犬": "mummy_dog", "石膏士兵": "plaster_soldier", "石膏士官": "plaster_officer", "石膏指挥官": "plaster_commander",
    "幼魔精灵2": "young_imp_spirit_2", "幼魔精灵": "young_imp_spirit", "钢甲猪": "steel_boar", "三眼章鱼": "three_eye_octopus",
    "蓝水灵": "blue_slime", "蝙蝠": "bat", "小幽灵": "small_ghost", "大幽灵": "big_ghost", "谢尔德": "shield", "青蛇": "green_snake",
    "黑石头人": "black_golem", "混种石头人": "mixed_golem", "无魂蘑菇": "soulless_mushroom", "火独眼兽": "fire_eye_beast",
    "无魂蘑菇王": "soulless_mushroom_king", "青龙": "green_dragon", "土龙": "earth_dragon", "怪猫": "monster_cat", "冰龙": "ice_dragon",
    "黑恐龙": "black_dinosaur", "月牙牛魔王": "crescent_cow_demon", "长枪牛魔王": "spear_cow_demon", "蝙蝠怪": "giant_bat",
    "火野猪": "fire_boar", "赤龙": "red_dragon", "石头人": "rock_golem", "鳄鱼": "crocodile", "黑鳄鱼": "black_crocodile",
    "火独眼兽2": "fire_eye_beast_2", "无魂蘑菇2": "soulless_mushroom_2", "刺蘑菇2": "spike_mushroom_2", "风独眼兽2": "wind_eye_beast_2",
    "火野猪2": "fire_boar_2", "猴子2": "monkey_2", "蓝蘑菇2": "blue_mushroom_2", "冰独眼兽2": "ice_eye_beast_2", "红螃蟹": "red_crab",
    "青螃蟹": "green_crab", "乌龟": "turtle",
}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

_CJK_FONT_PATHS = (
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
)
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | None] = {}
_ANNOTATOR_HINT = "画框后直接输数字设类别(Enter确认)  g=跳转  [/]=切换  c=清空当前类  左键加框/点选  拖角改框  右键删框  滚轮滚列表  s=保存  n/p=前后张  q=退出"
PANEL_WIDTH = 280
PANEL_ROW_HEIGHT = 20
PANEL_MIN_HEIGHT = 400
STATUS_HEIGHT = 46
MAX_CANVAS_WIDTH = 1400
MAX_CANVAS_HEIGHT = 900


def cjk_font(size: int) -> ImageFont.FreeTypeFont | None:
    """Return a cached font able to render CJK text, or None if no CJK font is available."""
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = None
        for path in _CJK_FONT_PATHS:
            if Path(path).exists():
                try:
                    _FONT_CACHE[size] = ImageFont.truetype(path, size)
                    break
                except OSError:
                    continue
    return _FONT_CACHE[size]


def image_files(path: Path) -> list[Path]:
    """Return image files below a directory in stable order."""
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def read_image(path: Path) -> np.ndarray | None:
    """Read an image, tolerating non-ASCII paths that cv2.imread cannot open on Windows."""
    image = cv2.imread(str(path))
    if image is not None:
        return image
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray, extension: str = ".jpg") -> None:
    """Write an image, tolerating non-ASCII paths that cv2.imwrite cannot open on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


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
    review_manifest = []
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
            review_manifest.extend(
                save_review_images(
                    review_output,
                    source.name,
                    image,
                    detections,
                    model_names,
                    str(target_label.relative_to(output)),
                )
            )
        print(f"[{split}] {source.name}: {len(labels)} detection(s)")
    (output / "dataset.yaml").write_text(
        f"path: {output.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {cls}: {name}\n" for cls, name in sorted(model_names.items())),
        encoding="utf-8",
    )
    if review_output:
        (review_output / "crops" / "manifest.json").write_text(
            json.dumps(review_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    review_message = f"\nReview images: {review_output.resolve()}" if review_output else ""
    print(f"Dataset created at {output.resolve()}{review_message}")


def save_review_images(
    review_output: Path,
    image_name: str,
    image: np.ndarray,
    detections: list[tuple[int, tuple[int, int, int, int], float]],
    names: dict[int, str],
    label_reference: str,
) -> list[dict]:
    """Save an annotated source image and one crop per old-model detection."""
    annotated = image.copy()
    crop_dir = review_output / "crops"
    manifest = []
    for index, (cls, (x1, y1, x2, y2), confidence) in enumerate(detections):
        color = (0, 220, 0)
        label = f"{cls}:{names[cls]} {confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        crop = image[max(0, y1) : min(image.shape[0], y2), max(0, x1) : min(image.shape[1], x2)]
        if crop.size:
            class_dir = crop_dir / f"{cls}_{names[cls]}"
            crop_name = f"{Path(image_name).stem}_{index:03d}.jpg"
            write_image(class_dir / crop_name, crop)
            manifest.append(
                {
                    "crop": str(Path(class_dir.name) / crop_name),
                    "label": label_reference,
                    "class_id": cls,
                    "box": [x1, y1, x2, y2],
                }
            )
    write_image(review_output / image_name, annotated)
    return manifest


def export_instances(args: argparse.Namespace) -> None:
    """Export one labeled instance image per class, grouped by the class name."""
    dataset = Path(args.dataset)
    images_root, labels_root = dataset / "images", dataset / "labels"
    config = yaml.safe_load((dataset / "dataset.yaml").read_text(encoding="utf-8"))
    raw_names = config.get("names", {})
    names = {int(k): v for k, v in raw_names.items()} if isinstance(raw_names, dict) else dict(enumerate(raw_names))
    folder_names = {cls: ENGLISH_MONSTER_NAMES.get(name, f"class_{cls}") for cls, name in names.items()}
    output = Path(args.output) if args.output else dataset / "instances"
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output is not empty: {output}. Use --overwrite to rebuild it.")
    if args.overwrite:
        shutil.rmtree(output, ignore_errors=True)
    for name in folder_names.values():
        (output / name).mkdir(parents=True, exist_ok=True)

    saved = 0
    for source in image_files(images_root):
        image = read_image(source)
        if image is None:
            continue
        height, width = image.shape[:2]
        relative = source.relative_to(images_root)
        label_path = (labels_root / relative).with_suffix(".txt")
        if not label_path.exists():
            label_path = labels_root / f"{source.stem}.txt"
        for cls, (x1, y1, x2, y2) in read_labels(label_path, width, height):
            if cls not in names or (output / folder_names[cls] / f"{folder_names[cls]}.png").exists():
                continue
            x1, y1 = max(0, x1 - args.margin), max(0, y1 - args.margin)
            x2, y2 = min(width, x2 + args.margin), min(height, y2 + args.margin)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            if max(crop.shape[:2]) < args.min_size:
                scale = args.min_size / max(crop.shape[:2])
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            write_image(output / folder_names[cls] / f"{folder_names[cls]}.png", crop, ".png")
            saved += 1
            break
    print(f"Exported {saved}/{len(names)} class images -> {output.resolve()}")


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
    manifest_paths = (
        [crops / "manifest.json"] if (crops / "manifest.json").exists() else sorted(crops.rglob("manifest.json"))
    )
    if not manifest_paths:
        raise SystemExit(f"No crop manifest found under: {crops}. Run auto-label again to create one.")
    grouped: dict[str, list[tuple[Path, dict]]] = {}
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest:
            if args.class_id is None or item["class_id"] == args.class_id:
                grouped.setdefault(item["label"], []).append((manifest_path.parent, item))
    removed = 0
    kept = 0
    for relative_label, manifest_items in grouped.items():
        label_path = Path(args.dataset) / relative_label
        source_candidates = list((Path(args.dataset) / "images" / label_path.parent.name).glob(f"{label_path.stem}.*"))
        if not source_candidates:
            continue
        source = cv2.imread(str(source_candidates[0]))
        if source is None:
            continue
        height, width = source.shape[:2]
        current = read_labels(label_path, width, height)
        accepted = Counter(
            (item["class_id"], tuple(item["box"]))
            for manifest_root, item in manifest_items
            if (manifest_root / item["crop"]).exists()
        )
        filtered = []
        for label in current:
            key = (label[0], label[1])
            if key in accepted and accepted[key] > 0:
                filtered.append(label)
                accepted[key] -= 1
            else:
                removed += 1
        kept += len(filtered)
        write_labels(label_path, filtered, width, height)
    print(f"Applied crop review: kept {kept} boxes, removed {removed} boxes")


def merge_dataset(args: argparse.Namespace) -> None:
    """Copy a source dataset's images and labels into a target dataset, reusing the target's existing data."""
    target = Path(args.target)
    source = Path(args.source)
    for name, current in (("target", target), ("source", source)):
        missing = [
            f"{name}/{sub}"
            for sub in ("images/train", "images/val", "labels/train", "labels/val")
            if not (current / sub).is_dir()
        ]
        if missing:
            raise SystemExit(f"Invalid {name} dataset, missing: {', '.join(missing)}: {current}")

    def read_names(path: Path) -> dict[int, str]:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        names = config.get("names", {})
        return {int(cls): value for cls, value in names.items()} if isinstance(names, dict) else dict(enumerate(names))

    target_names = read_names(target / "dataset.yaml")
    source_names = read_names(source / "dataset.yaml")
    if target_names != source_names:
        differences = [
            f"  {cls}: target={target_names.get(cls)} source={source_names.get(cls)}"
            for cls in sorted(set(target_names) | set(source_names))
            if target_names.get(cls) != source_names.get(cls)
        ]
        raise SystemExit("Class names differ between datasets; refusing to merge:\n" + "\n".join(differences))

    # Index target image stems so existing images can be updated in place.
    target_stems: dict[str, str] = {}
    for split in ("train", "val"):
        for image_path in image_files(target / "images" / split):
            target_stems.setdefault(image_path.stem, split)

    merged = updated = skipped = 0
    split_counts = {split: 0 for split in ("train", "val")}
    for split in ("train", "val"):
        for image_path in image_files(source / "images" / split):
            source_label = source / "labels" / split / f"{image_path.stem}.txt"
            existing_split = target_stems.get(image_path.stem)
            if existing_split is not None:
                if not args.overwrite:
                    skipped += 1
                    continue
                target_label = target / "labels" / existing_split / f"{image_path.stem}.txt"
                target_label.parent.mkdir(parents=True, exist_ok=True)
                if source_label.exists():
                    shutil.copy2(source_label, target_label)
                else:
                    target_label.write_text("", encoding="utf-8")
                updated += 1
                split_counts[existing_split] += 1
            else:
                target_image = target / "images" / split / image_path.name
                target_image.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, target_image)
                if source_label.exists():
                    target_label = target / "labels" / split / f"{image_path.stem}.txt"
                    target_label.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_label, target_label)
                else:
                    print(f"[warning] no label for {image_path.name}; copied image only")
                merged += 1
                split_counts[split] += 1
    print(
        f"Merged: {merged} new images, Updated: {updated} existing labels "
        f"(train {split_counts['train']} / val {split_counts['val']}), Skipped {skipped}"
    )


def build_73_dataset(args: argparse.Namespace) -> None:
    """Build a fresh multi-class dataset from images, inheriting player (class 0) boxes from existing datasets."""
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output is not empty: {output}. Use --overwrite to rebuild it.")
    if args.overwrite:
        shutil.rmtree(output, ignore_errors=True)
    names_config = yaml.safe_load(Path(args.names_yaml).read_text(encoding="utf-8"))
    raw = names_config.get("names", {})
    names = {int(cls): name for cls, name in raw.items()} if isinstance(raw, dict) else dict(enumerate(raw))
    if not names:
        raise SystemExit(f"No class names found in {args.names_yaml}")

    # Index inherited player labels by image stem: stem -> (split, label path).
    inherited: dict[str, tuple[str, Path]] = {}
    for dataset in (Path(path) for path in args.inherit_player):
        for split in ("train", "val"):
            image_dir = dataset / "images" / split
            label_dir = dataset / "labels" / split
            if not image_dir.is_dir() or not label_dir.is_dir():
                continue
            for label in label_dir.glob("*.txt"):
                if label.stem in inherited or not any(image_dir.glob(f"{label.stem}.*")):
                    continue
                inherited[label.stem] = (split, label)

    sources = [image for path in args.images for image in image_files(Path(path))]
    rng = random.Random(args.seed)
    rng.shuffle(sources)
    val_names = {source.name for source in sources[:round(len(sources) * args.val_fraction)]}

    inherit_map = {0: 0}
    for pair in args.inherit_monster or []:
        try:
            old_cls, new_cls = (int(part) for part in pair.split(":"))
        except ValueError:
            raise SystemExit(f"Invalid --inherit-monster pair (expected OLD:NEW), got: {pair}")
        inherit_map[old_cls] = new_cls

    inherited_boxes: Counter[int] = Counter()
    total = {split: 0 for split in ("train", "val")}
    for source in sources:
        split, label_path = inherited.get(
            source.stem, ("val" if source.name in val_names else "train", None)
        )
        target_image = output / "images" / split / source.name
        if target_image.exists():
            print(f"[skip] existing: {source.name}")
            continue
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_image)
        image = read_image(source)
        if image is None:
            print(f"[warning] unreadable image (copied, no labels): {source.name}")
            continue
        height, width = image.shape[:2]
        labels = []
        if label_path is not None:
            for cls, box in read_labels(label_path, width, height):
                if cls in inherit_map:
                    labels.append((inherit_map[cls], box))
                    inherited_boxes[inherit_map[cls]] += 1
        write_labels(output / "labels" / split / f"{source.stem}.txt", labels, width, height)
        total[split] += 1
    (output / "dataset.yaml").write_text(
        f"path: {output.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {cls}: {name}\n" for cls, name in sorted(names.items())),
        encoding="utf-8",
    )
    inherited_summary = ", ".join(f"{names.get(cls, cls)}:{count}" for cls, count in sorted(inherited_boxes.items()))
    print(
        f"Built dataset: {sum(total.values())} images (train {total['train']} / val {total['val']}), "
        f"{len(names)} classes, inherited: {inherited_summary}"
    )


def prepare_player_dataset(args: argparse.Namespace) -> None:
    """Remove the legacy background class and reserve class 0 for players."""
    dataset = Path(args.dataset)
    config_path = dataset / "dataset.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    names = config.get("names", {})
    names = {int(cls): name for cls, name in names.items()} if isinstance(names, dict) else dict(enumerate(names))
    already_prepared = names.get(0) == "player"
    monster_names = {cls: name for cls, name in names.items() if cls != 0 and name not in ("__background__", "player")}
    new_names = {0: "player", **monster_names}
    removed_background = 0
    for split in ("train", "val"):
        label_dir = dataset / "labels" / split
        for label_path in label_dir.glob("*.txt"):
            image_candidates = list((dataset / "images" / split).glob(f"{label_path.stem}.*"))
            if not image_candidates:
                continue
            image = cv2.imread(str(image_candidates[0]))
            if image is None:
                continue
            height, width = image.shape[:2]
            labels = read_labels(label_path, width, height)
            filtered = labels if already_prepared else [label for label in labels if label[0] != 0]
            removed_background += len(labels) - len(filtered)
            write_labels(label_path, filtered, width, height)
    config["names"] = {cls: name for cls, name in sorted(new_names.items())}
    config["path"] = dataset.resolve().as_posix()
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(
        f"Prepared player dataset: class 0=player, monster classes={len(monster_names)}, removed background boxes={removed_background}"
    )


class Annotator:
    """OpenCV annotator for adding, deleting, and resizing any dataset class."""

    def __init__(self, dataset: Path, split: str, player_class_id: int | None = None) -> None:
        self.images = image_files(dataset / "images" / split)
        self.labels_dir = dataset / "labels" / split
        config_path = dataset / "dataset.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        names = config.get("names", {})
        self.class_names = (
            {int(cls): name for cls, name in names.items()} if isinstance(names, dict) else dict(enumerate(names))
        )
        self.player_class_id = player_class_id if player_class_id is not None else 0
        self.class_names[self.player_class_id] = "player"
        config["names"] = {cls: name for cls, name in sorted(self.class_names.items())}
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        self.index = 0
        self.labels: list[tuple[int, tuple[int, int, int, int]]] = []
        self.image: np.ndarray | None = None
        self.start: tuple[int, int] | None = None
        self.current_class_id = self.player_class_id
        self.drag_index: int | None = None
        self.drag_corner: int | None = None
        self.goto_active = False
        self.goto_input = ""
        self.active_index: int | None = None
        self.number_active = False
        self.number_input = ""
        self.drag_original: tuple[int, int, int, int] | None = None
        self.class_ids = sorted(self.class_names)
        self.scale = 1.0
        self.display_img_w = 0
        self.display_img_h = 0
        self.panel_scroll = 0
        self.panel_scroll_max = 0
        self.panel_content_h = 0
        self.window = "YOLO annotator | g=jump class, [/] cycle, c=clear, drag=add/resize, right-click=delete, s=save, n/p=next, q=quit"

    def load(self) -> None:
        """Load the current image and labels, and update the scaled display layout."""
        self.image = read_image(self.images[self.index])
        if self.image is None:
            raise RuntimeError(f"Cannot read {self.images[self.index]}")
        h, w = self.image.shape[:2]
        self.scale = min(1.0, (MAX_CANVAS_WIDTH - PANEL_WIDTH) / w, MAX_CANVAS_HEIGHT / h)
        self.display_img_w = max(1, round(w * self.scale))
        self.display_img_h = max(1, round(h * self.scale))
        self.panel_content_h = max(self.display_img_h, PANEL_MIN_HEIGHT)
        self.panel_scroll_max = max(0, len(self.class_ids) * PANEL_ROW_HEIGHT - self.panel_content_h)
        self.panel_scroll = min(self.panel_scroll, self.panel_scroll_max)
        self.labels = read_labels(self.labels_dir / f"{self.images[self.index].stem}.txt", w, h)
        self.active_index = None
        self.number_active = False
        self.number_input = ""
        self._ensure_visible()

    def save(self) -> None:
        """Save the current labels."""
        assert self.image is not None
        h, w = self.image.shape[:2]
        write_labels(self.labels_dir / f"{self.images[self.index].stem}.txt", self.labels, w, h)
        print(f"saved {self.images[self.index].name}: {len(self.labels)} box(es)")

    def jump_to_class(self) -> None:
        """Jump the current class to the nearest valid class id typed during goto mode."""
        if self.goto_input:
            target = int(self.goto_input)
            self.current_class_id = min(self.class_ids, key=lambda class_id: abs(class_id - target))
        self.goto_active = False
        self.goto_input = ""
        self._ensure_visible()

    def _ensure_visible(self) -> None:
        """Scroll the class panel so the current class row stays visible."""
        if self.panel_content_h <= 0:
            return
        try:
            row = self.class_ids.index(self.current_class_id)
        except ValueError:
            return
        row_top = row * PANEL_ROW_HEIGHT - self.panel_scroll
        if row_top < 0:
            self.panel_scroll += row_top
        elif row_top + PANEL_ROW_HEIGHT > self.panel_content_h:
            self.panel_scroll += row_top + PANEL_ROW_HEIGHT - self.panel_content_h
        self.panel_scroll = max(0, min(self.panel_scroll_max, self.panel_scroll))

    def _commit_number(self) -> None:
        """Assign the typed class number to the active box and make it the current class."""
        if self.active_index is not None and 0 <= self.active_index < len(self.labels) and self.number_input:
            _cls, box = self.labels[self.active_index]
            new_cls = min(self.class_ids, key=lambda class_id: abs(class_id - int(self.number_input)))
            self.labels[self.active_index] = (new_cls, box)
            self.current_class_id = new_cls
        self.number_active = False
        self.number_input = ""

    def _cancel_number(self) -> None:
        """Cancel numbering and keep the active box's current class."""
        self.number_active = False
        self.number_input = ""

    def draw(self) -> np.ndarray:
        """Render the scaled image, box overlays, a class-reference panel, and status text."""
        assert self.image is not None
        height = self.panel_content_h + STATUS_HEIGHT
        width = self.display_img_w + PANEL_WIDTH
        canvas = np.full((height, width, 3), (22, 22, 22), dtype=np.uint8)
        if self.scale == 1.0:
            canvas[: self.display_img_h, : self.display_img_w] = self.image
        else:
            canvas[: self.display_img_h, : self.display_img_w] = cv2.resize(
                self.image, (self.display_img_w, self.display_img_h), interpolation=cv2.INTER_AREA
            )
        for index, (cls, (x1, y1, x2, y2)) in enumerate(self.labels):
            color = (0, 120, 255) if cls == self.player_class_id else (0, 220, 0)
            active = index == self.drag_index
            sx1, sy1 = round(x1 * self.scale), round(y1 * self.scale)
            sx2, sy2 = round(x2 * self.scale), round(y2 * self.scale)
            cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), color, 4 if active else 2)
            for corner_x, corner_y in ((sx1, sy1), (sx2, sy1), (sx1, sy2), (sx2, sy2)):
                cv2.circle(canvas, (corner_x, corner_y), 6 if active else 5, color, -1)
            if index == self.active_index:
                cv2.rectangle(canvas, (sx1 - 2, sy1 - 2), (sx2 + 2, sy2 + 2), (0, 255, 255), 2)
        current_name = self.class_names.get(self.current_class_id, "?")
        goto_preview = f"  跳转到: {self.goto_input}_" if self.goto_active else ""
        if self.number_active:
            num_preview = f"  第{self.active_index + 1}框类别: {self.number_input}_"
        elif self.active_index is not None and not self.goto_active:
            num_preview = f"  第{self.active_index + 1}框: 输数字设类别"
        else:
            num_preview = ""
        status = (
            f"{self.index + 1}/{len(self.images)}  当前类别: {self.current_class_id}:{current_name}"
            f"{goto_preview}{num_preview}  框数: {len(self.labels)}"
        )
        font = cjk_font(20)
        if font is None:
            # No CJK font available: fall back to cv2 text (Chinese names render as '?')
            cv2.putText(canvas, status, (10, height - STATUS_HEIGHT + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2)
            for cls, (x1, y1, x2, y2) in self.labels:
                color = (0, 120, 255) if cls == self.player_class_id else (0, 220, 0)
                sx1, sy1 = round(x1 * self.scale), round(y1 * self.scale)
                cv2.putText(canvas, str(cls), (sx1 + 2, max(10, sy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            return canvas
        font_small = cjk_font(15) or font
        pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).convert("RGBA")
        overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
        drawer = ImageDraw.Draw(overlay)

        # Class-reference panel on the right.
        panel_x = self.display_img_w
        drawer.rectangle((panel_x, 0, width, height - STATUS_HEIGHT), fill=(38, 38, 38, 255))
        drawer.text((panel_x + 8, 6), f"类别列表 ({len(self.class_ids)})", font=font, fill=(255, 255, 255, 255))
        for i, cls in enumerate(self.class_ids):
            row_y = 34 + i * PANEL_ROW_HEIGHT - self.panel_scroll
            if row_y + PANEL_ROW_HEIGHT < 0 or row_y >= height - STATUS_HEIGHT:
                continue
            if cls == self.current_class_id:
                drawer.rectangle((panel_x, row_y, width, row_y + PANEL_ROW_HEIGHT), fill=(0, 90, 200, 255))
                row_color = (255, 255, 255, 255)
            else:
                row_color = (215, 215, 215, 255)
            drawer.text(
                (panel_x + 8, row_y + 2),
                f"{cls}  {self.class_names.get(cls, '?')}",
                font=font_small,
                fill=row_color,
            )

        # Status bar at the bottom.
        drawer.rectangle((0, height - STATUS_HEIGHT, width, height), fill=(0, 0, 0, 185))
        drawer.text((10, height - STATUS_HEIGHT + 4), status, font=font, fill=(255, 255, 255, 255))
        drawer.text((10, height - STATUS_HEIGHT + 26), _ANNOTATOR_HINT, font=font_small, fill=(190, 190, 190, 255))

        # Box class labels on the scaled image.
        for cls, (x1, y1, x2, y2) in self.labels:
            color = (0, 120, 255) if cls == self.player_class_id else (0, 220, 0)
            rgb_color = (color[2], color[1], color[0])
            sx1, sy1 = round(x1 * self.scale), round(y1 * self.scale)
            drawer.text(
                (sx1 + 2, max(2, sy1 - font_small.size - 6)),
                self.class_names.get(cls, str(cls)),
                font=font_small,
                fill=rgb_color,
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
        pil_image = Image.alpha_composite(pil_image, overlay)
        canvas = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        return canvas

    def mouse(self, event: int, x: int, y: int, flags: int, _param: object) -> None:
        """Handle adding, resizing, deleting boxes, and class-panel selection."""
        if event == cv2.EVENT_MOUSEWHEEL:
            self.panel_scroll = max(
                0,
                min(self.panel_scroll_max, self.panel_scroll - (flags // 120) * PANEL_ROW_HEIGHT * 3),
            )
            return
        if x >= self.display_img_w:
            if event == cv2.EVENT_LBUTTONUP and self.drag_index is not None:
                self.drag_index = None
                self.drag_corner = None
                self.drag_original = None
            elif event == cv2.EVENT_LBUTTONDOWN and y < self.panel_content_h:
                row = (y - 34 + self.panel_scroll) // PANEL_ROW_HEIGHT
                if 0 <= row < len(self.class_ids):
                    self.current_class_id = self.class_ids[row]
                    self._ensure_visible()
            return
        ix = max(0, min(int(x / self.scale), self.image.shape[1]))
        iy = max(0, min(int(y / self.scale), self.image.shape[0]))
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_index = None
            self.drag_corner = None
            self.start = None
            for i, (_cls, (x1, y1, x2, y2)) in reversed(list(enumerate(self.labels))):
                corners = ((x1, y1), (x2, y1), (x1, y2), (x2, y2))
                corner_hit = None
                for corner, (corner_x, corner_y) in enumerate(corners):
                    if abs(ix - corner_x) <= 12 and abs(iy - corner_y) <= 12:
                        corner_hit = corner
                        break
                if corner_hit is not None:
                    self.drag_index = i
                    self.drag_corner = corner_hit
                    self.drag_original = (x1, y1, x2, y2)
                    return
                if x1 <= ix <= x2 and y1 <= iy <= y2:
                    self.active_index = i
                    return
            self.start = (ix, iy)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_index is not None:
            x1, y1, x2, y2 = self.drag_original
            if self.drag_corner in (0, 2):
                x1 = ix
            else:
                x2 = ix
            if self.drag_corner in (0, 1):
                y1 = iy
            else:
                y2 = iy
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            cls, _box = self.labels[self.drag_index]
            if x2 - x1 >= 3 and y2 - y1 >= 3:
                self.labels[self.drag_index] = (cls, (x1, y1, x2, y2))
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag_index is not None:
                self.drag_index = None
                self.drag_corner = None
                self.drag_original = None
            elif self.start:
                x1, y1 = self.start
                box = (min(x1, ix), min(y1, iy), max(x1, ix), max(y1, iy))
                if box[2] - box[0] >= 3 and box[3] - box[1] >= 3:
                    self.labels.append((self.current_class_id, box))
                    self.active_index = len(self.labels) - 1
                    self.number_active = False
                    self.number_input = ""
                self.start = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i, (_cls, (x1, y1, x2, y2)) in reversed(list(enumerate(self.labels))):
                if x1 <= ix <= x2 and y1 <= iy <= y2:
                    self.labels.pop(i)
                    if self.active_index == i:
                        self.active_index = None
                    elif self.active_index is not None and self.active_index > i:
                        self.active_index -= 1
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
                consumed = True
                if self.number_active:
                    if ord("0") <= key <= ord("9"):
                        self.number_input += chr(key)
                        if len(self.number_input) > 3:
                            self.number_input = self.number_input[1:]
                    elif key in (13, 10):
                        self._commit_number()
                    elif key in (8,):
                        self.number_input = self.number_input[:-1]
                    elif key in (27,):
                        self._cancel_number()
                    else:
                        self._cancel_number()
                        consumed = False
                elif self.goto_active:
                    if ord("0") <= key <= ord("9"):
                        self.goto_input += chr(key)
                        if len(self.goto_input) > 3:
                            self.goto_input = self.goto_input[1:]
                    elif key in (13, 10):
                        self.jump_to_class()
                    elif key in (8,):
                        self.goto_input = self.goto_input[:-1]
                    elif key in (27, ord("q")):
                        self.goto_active = False
                        self.goto_input = ""
                    else:
                        consumed = False
                elif key in (ord("g"), ord("G")):
                    self.goto_active = True
                    self.goto_input = ""
                elif ord("0") <= key <= ord("9") and self.active_index is not None:
                    self.number_active = True
                    self.number_input = chr(key)
                else:
                    consumed = False
                if not consumed:
                    if key in (ord("s"),):
                        self.save()
                    elif key in (ord("n"), 32):
                        self.save()
                        if self.index >= len(self.images) - 1:
                            cv2.destroyAllWindows()
                            return
                        self.index += 1
                        break
                    elif key == ord("p"):
                        self.save()
                        self.index = max(self.index - 1, 0)
                        break
                    elif key == ord("c"):
                        if self.active_index is not None and 0 <= self.active_index < len(self.labels) and self.labels[self.active_index][0] == self.current_class_id:
                            self.active_index = None
                        self.labels = [(cls, box) for cls, box in self.labels if cls != self.current_class_id]
                    elif key in (ord("["), ord("]")):
                        current_index = self.class_ids.index(self.current_class_id) if self.current_class_id in self.class_ids else 0
                        step = -1 if key == ord("[") else 1
                        self.current_class_id = self.class_ids[(current_index + step) % len(self.class_ids)]
                        self._ensure_visible()
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
        project=str(Path(args.project).resolve()),
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
    annotate.add_argument("--player-class-id", type=int, default=None, help="defaults to class 0")
    annotate.set_defaults(func=lambda a: Annotator(a.dataset, a.split, a.player_class_id).run())
    review = sub.add_parser("review-old", help="keep and review one legacy model class")
    review.add_argument("--dataset", type=Path, required=True)
    review.add_argument("--class-id", type=int, default=9)
    review.add_argument("--split", choices=("all", "train", "val"), default="all")
    review.set_defaults(func=lambda a: OldLabelReviewer(a.dataset, a.split, a.class_id).run())
    apply_review = sub.add_parser("apply-crop-review", help="apply manually deleted crop files to labels")
    apply_review.add_argument("--dataset", type=Path, required=True)
    apply_review.add_argument("--crops", type=Path, default=Path("data/old_yolo_review/crops"))
    apply_review.add_argument(
        "--class-id", type=int, default=None, help="only apply one class; omit to apply every class"
    )
    apply_review.set_defaults(func=apply_crop_review)
    instances = sub.add_parser("export-instances", help="export one labeled image per class")
    instances.add_argument("--dataset", type=Path, required=True)
    instances.add_argument("--output", type=Path, default=None, help="defaults to <dataset>/instances")
    instances.add_argument("--margin", type=int, default=4, help="extra source pixels around each box")
    instances.add_argument("--min-size", type=int, default=96, help="nearest-neighbor upscale for small crops")
    instances.add_argument("--overwrite", action="store_true")
    instances.set_defaults(func=export_instances)
    merge = sub.add_parser("merge-dataset", help="copy a source dataset into a target dataset, reusing existing data")
    merge.add_argument("--target", type=Path, required=True, help="existing dataset to merge into")
    merge.add_argument("--source", type=Path, required=True, help="new dataset to merge from")
    merge.add_argument(
        "--overwrite",
        action="store_true",
        help="replace labels of images already present in the target instead of skipping them",
    )
    merge.set_defaults(func=merge_dataset)
    build = sub.add_parser(
        "build-73-dataset",
        help="build a fresh multi-class dataset in data/ from images, inheriting player (class 0) boxes",
    )
    build.add_argument("--output", type=Path, required=True, help="target dataset directory")
    build.add_argument("--images", type=Path, nargs="+", required=True, help="directories of images to include")
    build.add_argument("--names-yaml", type=Path, required=True, help="dataset.yaml providing the full class names")
    build.add_argument(
        "--inherit-player", type=Path, nargs="*", default=[], help="datasets to inherit class-0 (player) boxes from"
    )
    build.add_argument(
        "--inherit-monster",
        nargs="*",
        default=[],
        metavar="OLD:NEW",
        help="inherit boxes from an old class id into a new class id, e.g. 9:2 (repeatable)",
    )
    build.add_argument("--val-fraction", type=float, default=0.2)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--overwrite", action="store_true")
    build.set_defaults(func=build_73_dataset)
    prepare = sub.add_parser("prepare-player-dataset", help="reserve class 0 for players and remove background labels")
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.set_defaults(func=prepare_player_dataset)
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
