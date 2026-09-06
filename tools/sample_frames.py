# tools/sample_frames.py
r"""Content-driven frame sampling from auto_label run outputs (per-video, temporal).

Consecutive video frames carry little information gain. This tool keeps a
content-diverse subset while ALWAYS retaining the "hard" cluttered frames where
the ground is covered in dropped items (the accuracy bottleneck). Selection is
per video (temporal) and greedy:

- ``fingerprint`` = 8x8 average-hash of the downscaled frame image.
- ``clutter score`` = number of detection/待复核 boxes in that frame (proxy for
  "many dropped items on the ground": noisy low-conf detections pile up on ground
  items, so cluttered frames have many boxes).
- Keep a frame if its clutter score is in the top ``--clutter-top`` fraction of
  that video (a hard frame, never dropped), OR if its fingerprint differs from
  the last kept frame by >= ``--min-hamming`` bits (new content).

Selected frames are copied (image + label txt + review json) into ``--out`` so
the subset can be re-arbitrated (export-review → vlm-review → apply-review) and
handed to review_pack. Frame stems are unique across videos, so merging is safe.

Examples:
    python tools/sample_frames.py \
      --dir artifacts/label_combat_20260905/combat-122510 \
      --dir artifacts/label_combat_20260905/combat-125338-s001 \
      --dir artifacts/label_combat_20260905/combat-125338-s002 \
      --out data/maps/dater_shekou_src \
      --clutter-top 0.5 --min-hamming 12
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_SUFFIX = ".txt"


def _imread(path: Path) -> np.ndarray | None:
    """Read an image, tolerating non-ASCII paths (cv2.imread fails on Windows)."""
    img = cv2.imread(str(path))
    if img is not None:
        return img
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def avg_hash(img: np.ndarray) -> int:
    """8x8 average-hash as a 64-bit int (gray mean threshold per block)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    bits = (small > small.mean()).flatten().astype(np.uint8)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def hamming(a: int, b: int) -> int:
    """Number of set bits in a XOR b (perceptual distance between two hashes)."""
    return bin(a ^ b).count("1")


def clutter_score(review: Path) -> int:
    """Number of pending/待复核 boxes in a frame's review json (clutter proxy)."""
    if not review.is_file():
        return 0
    try:
        return len(json.loads(review.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return 0


def _image_stems(split: Path) -> list[str]:
    """YOLO label-image stems under an images/ dir (frame names, sorted)."""
    return sorted(p.stem for p in split.glob("*") if p.suffix.lower() in IMG_SUFFIXES)


def select_video(vdir: Path, clutter_top: float, min_hamming: int) -> list[str]:
    """Greedy content-driven select over a video's frames; returns kept stems."""
    images = vdir / "images"
    review = vdir / "review"
    stems: list[str] = []
    for sub in ("train", "val"):
        if (images / sub).is_dir():
            stems += _image_stems(images / sub)
    if not stems:
        stems = _image_stems(images)
    stems = sorted(set(stems))

    scores = {s: clutter_score(review / f"{s}.json") for s in stems}
    if scores:
        cutoff = float(np.quantile(list(scores.values()), 1.0 - clutter_top))
    else:
        cutoff = 0.0

    kept: list[str] = []
    last_hash: int | None = None
    for s in stems:
        img = _imread(images / f"{s}{_image_ext(images, s)}")
        if img is None:
            continue
        fp = avg_hash(img)
        force = cutoff and scores[s] >= cutoff  # hard (cluttered) frame: always keep
        if force or last_hash is None or hamming(fp, last_hash) >= min_hamming:
            kept.append(s)
            last_hash = fp
    return kept


def _image_ext(images: Path, stem: str) -> str:
    for ext in IMG_SUFFIXES:
        if (images / f"{stem}{ext}").is_file():
            return ext
    return ".jpg"


def main() -> None:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", action="append", type=Path, required=True,
                    help="auto_label run 输出目录（可重复，逐视频）")
    ap.add_argument("--out", type=Path, required=True, help="采样后子集目录（images/labels/review）")
    ap.add_argument("--clutter-top", type=float, default=0.4,
                    help="保留杂物得分 top 比例的难帧（0=关闭强制保留）")
    ap.add_argument("--min-hamming", type=int, default=12,
                    help="相邻帧指纹位距达到此值才保留新内容帧")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"out exists (use --overwrite): {out}")
        shutil.rmtree(out)
    for sub in ("images", "labels", "review"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    for vdir in args.dir:
        vdir = Path(vdir)
        stems = select_video(vdir, args.clutter_top, args.min_hamming)
        for s in stems:
            src_img = next((p for p in (vdir / "images").glob(f"{s}.*")), None)
            if src_img is None:
                continue
            shutil.copy2(src_img, out / "images" / src_img.name)
            src_lbl = vdir / "labels" / f"{s}{LABEL_SUFFIX}"
            if not src_lbl.is_file():
                src_lbl = next((p for p in (vdir / "labels").glob(f"{s}.*")), None)
            if src_lbl is not None and src_lbl.is_file():
                shutil.copy2(src_lbl, out / "labels" / src_lbl.name)
            rj = vdir / "review" / f"{s}.json"
            if rj.is_file():
                shutil.copy2(rj, out / "review" / rj.name)
        print(f"{vdir.name}: kept {len(stems)} frames")

    n_img = len([p for p in (out / "images").iterdir() if p.suffix.lower() in IMG_SUFFIXES]) if out.exists() else 0
    n_lbl = len([p for p in (out / "labels").iterdir() if p.suffix == ".txt"]) if (out / "labels").exists() else 0
    print(f"sampled -> {out}: images={n_img} labels={n_lbl}")


if __name__ == "__main__":
    main()
