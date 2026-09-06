# tools/merge_maps.py
r"""Merge per-map YOLO dataset folders into one unified dataset for cloud training.

Each source must use the SAME class-id table as ``--master`` (default
``data/wgc_review/dataset.yaml``). A mismatching table is REFUSED unless an
explicit ``--map-source SRC:MAP.yaml`` is given, so classes can never be silently
shifted. The output dataset.yaml carries NO ``path`` key: Ultralytics roots it at
the yaml dir, so the same copy works on an AutoDL GPU instance (see
``docs/autodl_training.md``) and in local training. Duplicate frames are
dropped by content hash; the whole pool is re-split into train/val with a
guarantee that every class appears in val.

Examples:
    python tools/merge_maps.py --out data/unified_20260905 \
        --master data/wgc_review/dataset.yaml \
        --source data/maps/射手训练场 --source data/annotated_73 --source data/new_yolo_73
    python tools/merge_maps.py --dry-run --out data/unified_20260905 ...  # inspect only
"""
from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val")


def read_names(path: Path) -> dict[int, str]:
    """Read a dataset.yaml ``names`` table as {id: name}."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = config.get("names", {})
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return dict(enumerate(names))


def image_files(path: Path) -> list[Path]:
    """Return image files under a directory in stable (sorted) order."""
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def content_hash(path: Path) -> str:
    """Return a stable content hash for de-duplication."""
    return hashlib.sha1(path.read_bytes()).hexdigest()


def label_classes(label: Path) -> set[int]:
    """Return the set of class ids referenced by a YOLO label txt (or empty set)."""
    if not label.is_file():
        return set()
    cls = set()
    for line in label.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                cls.add(int(line.split()[0]))
            except ValueError:
                continue
    return cls


def assign_split(records: list[set[int]], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Split record indices into (train, val) so every class appears in val.

    ``records[i]`` is the set of classes in image i. Deterministic for a given
    ``seed``. Val may slightly exceed ``val_fraction`` when many classes need
    coverage; that is expected and acceptable.
    """
    rng = random.Random(seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    want_val = max(1, round(len(records) * val_fraction))
    all_classes: set[int] = set()
    for cls in records:
        all_classes |= cls
    val: list[int] = []
    covered: set[int] = set()
    for i in order:
        cls = records[i]
        if (cls - covered) or len(val) < want_val:
            val.append(i)
            covered |= cls
            if len(val) >= want_val and covered == all_classes:
                break
    val.sort()
    val_set = set(val)
    train = [i for i in range(len(records)) if i not in val_set]
    return train, val


def _name_mismatch(master: dict[int, str], source: dict[int, str]) -> list[str]:
    """Return one-line descriptions of every id whose name differs between tables."""
    return [
        f"  {c}: master={master.get(c)} source={source.get(c)}"
        for c in sorted(set(master) | set(source))
        if master.get(c) != source.get(c)
    ]


def _load_mapping(path: Path) -> dict[int, int]:
    """Load an id-remap yaml {old_id: new_id}."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {int(k): int(v) for k, v in data.items()}


def _resolve_mapping(map_sources: dict, src: Path, source_names: dict[int, str], master_names: dict[int, str]) -> dict[int, int] | None:
    """Return the remap dict for ``src`` or None when names already match; raise on a real mismatch."""
    diff = _name_mismatch(master_names, source_names)
    if not diff:
        return None
    map_path = map_sources.get(str(src)) or map_sources.get(src.name)
    if map_path is None:
        raise SystemExit(
            "Class names differ between datasets; refusing to merge "
            f"{src}:\n" + "\n".join(diff) +
            "\n  → pass --map-source <src>:<map.yaml> to remap, or exclude this source."
        )
    mapping = _load_mapping(Path(map_path))
    bad = sorted(set(mapping.values()) - set(master_names))
    if bad:
        raise SystemExit(f"map for {src} targets unknown master ids: {bad}")
    return mapping


def _remapped_classes(classes: set[int], mapping: dict[int, int] | None) -> set[int]:
    if not mapping:
        return classes
    return {mapping.get(c, c) for c in classes}


def _remapped_label_text(label: Path, mapping: dict[int, int] | None) -> str | None:
    """Return remapped label text, or None when no remap needed (caller copies the file)."""
    if not mapping or not label.is_file():
        return None
    lines = label.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append(line)
            continue
        parts = line.split()
        parts[0] = str(mapping.get(int(parts[0]), int(parts[0])))
        out.append(" ".join(parts))
    return ("\n".join(out) + "\n") if out else ""


def _class_histogram(records: list[dict]) -> dict[int, int]:
    counts: Counter = Counter()
    for rec in records:
        for cls in rec["classes"]:
            counts[cls] += 1
    return dict(sorted(counts.items()))


def _print_dry_run(args, records: list[dict], train: list[int], val: list[int]) -> None:
    """Print summary/counts/coverage without writing anything."""
    all_classes: set[int] = set()
    for rec in records:
        all_classes |= rec["classes"]
    val_classes: set[int] = set()
    for i in val:
        val_classes |= records[i]["classes"]
    print(f"--dry-run-- total images={len(records)}")
    print(f"  train={len(train)} val={len(val)} (val_fraction={args.val_fraction})")
    print(f"  class histogram: {_class_histogram(records)}")
    print(f"  classes present overall: {len(all_classes)}")
    print(f"  classes missing from val: {sorted(all_classes - val_classes) if all_classes - val_classes else 'none'}")


def merge(args) -> None:
    """Merge source datasets into ``args.out`` (id-safe, deduped, re-split, no path)."""
    master_names = read_names(Path(args.master))
    map_sources = getattr(args, "map_sources", {})
    out = Path(args.out)

    records: list[dict] = []
    seen_hashes: set[str] = set()
    per_source: Counter = Counter()

    for src in args.sources:
        src = Path(src)
        source_names = read_names(src / "dataset.yaml")
        mapping = _resolve_mapping(map_sources, src, source_names, master_names)
        for split in SPLITS:
            for image in image_files(src / "images" / split):
                lbl = src / "labels" / split / f"{image.stem}.txt"
                digest = content_hash(image)
                if digest in seen_hashes:
                    continue  # duplicate frame across sources/maps
                seen_hashes.add(digest)
                classes = _remapped_classes(label_classes(lbl), mapping)
                records.append(
                    {
                        "image": image,
                        "label": lbl,
                        "classes": classes,
                        "remap": _remapped_label_text(lbl, mapping) if mapping else None,
                        "source": str(src),
                    }
                )
                per_source[str(src)] += 1

    train, val = assign_split([r["classes"] for r in records], args.val_fraction, args.seed)

    if args.dry_run:
        _print_dry_run(args, records, train, val)
        return

    if out.exists():
        shutil.rmtree(out)
    for split, idxs in (("train", train), ("val", val)):
        for i in idxs:
            rec = records[i]
            dst_img = out / "images" / split / rec["image"].name
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rec["image"], dst_img)
            dst_lbl = out / "labels" / split / f"{rec['image'].stem}.txt"
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            if rec["label"].is_file():
                if rec["remap"] is not None:
                    dst_lbl.write_text(rec["remap"], encoding="utf-8")
                else:
                    shutil.copy2(rec["label"], dst_lbl)
            else:
                dst_lbl.write_text("", encoding="utf-8")

    cfg = {"train": "images/train", "val": "images/val", "names": master_names}
    (out / "dataset.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"unified {out}: images={len(records)} train={len(train)} val={len(val)} sources={len(per_source)}")
    print(f"  sources: {dict(per_source)}")
    print(f"  class box histogram: {_class_histogram(records)}")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output unified dataset dir")
    parser.add_argument("--master", type=Path, required=True,
                        help="dataset.yaml providing the canonical 73-class names table")
    parser.add_argument("--source", action="append", type=Path, required=True,
                        help="source dataset dir (repeatable: --source A --source B)")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="print counts/histogram/coverage without writing anything")
    parser.add_argument("--map-source", action="append", default=[],
                        metavar="SRC:MAP.yaml",
                        help="optional id-remap yaml {old_id: new_id} for a source whose table differs")
    return parser


def main() -> None:
    """Parse command-line arguments."""
    args = build_parser().parse_args()
    map_sources: dict[str, Path] = {}
    for spec in args.map_source:
        src, _, map_path = spec.partition(":")
        map_sources[src] = Path(map_path)
    ns = argparse.Namespace(**vars(args))
    ns.sources = args.source  # argparse dest is "source"; merge() expects "sources"
    ns.map_sources = map_sources
    merge(ns)


if __name__ == "__main__":
    main()
