"""Prepare a local dataset folder for PAI-DLC cloud training (local only, no cloud deps).

Copies images/labels from an existing dataset directory into ``--out`` and writes a
``dataset.yaml`` **without** the Windows-absolute ``path`` key, so Ultralytics roots the
dataset at the yaml's own directory and the same copy works at any OSS mount path inside
the Linux training container (see ``ultralytics/data/utils.py`` ``check_det_dataset``).

Run from the ultralytics checkout:

    python cloud/dlc/prepare.py --dataset data/wgc_review --out artifacts/cloud_round

Then upload ``--out`` to OSS with ``cloud/dlc/upload.sh``.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_SUFFIXES = {".txt"}
SPLITS = ("train", "val")


def _copy_dir_files(src: Path, dst: Path, suffixes: set[str]) -> int:
    """Copy every file under ``src`` matching ``suffixes`` into ``dst``; returns file count."""
    if not src.is_dir():
        raise SystemExit(f"missing source dir: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(src.iterdir()):
        if f.is_file() and f.suffix.lower() in suffixes:
            shutil.copy2(f, dst / f.name)
            count += 1
    return count


def prepare(dataset: Path, out: Path, overwrite: bool) -> None:
    """Sanitize ``dataset`` into ``out`` (no absolute path in dataset.yaml) and report counts."""
    if out.exists():
        if not overwrite:
            raise SystemExit(f"out exists (use --overwrite): {out}")
        shutil.rmtree(out)

    cfg = yaml.safe_load((dataset / "dataset.yaml").read_text(encoding="utf-8"))
    for key in ("train", "val", "names"):
        if key not in cfg:
            raise SystemExit(f"dataset.yaml missing key: {key}")
    cfg.pop("path", None)  # Windows-absolute path invalid in the Linux container; yaml dir is used

    for split in SPLITS:
        imgs = _copy_dir_files(dataset / "images" / split, out / "images" / split, IMG_SUFFIXES)
        lbls = _copy_dir_files(dataset / "labels" / split, out / "labels" / split, LABEL_SUFFIXES)
        print(f"{split}: images={imgs} labels={lbls}")
        if imgs != lbls:
            print(f"  WARNING {split}: image/label count mismatch", file=sys.stderr)

    (out / "dataset.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    ids = list(cfg["names"])
    print(f"wrote {out / 'dataset.yaml'} (classes={len(ids)}):")
    print(f"  {ids[0]}: {cfg['names'][ids[0]]} ... {ids[-1]}: {cfg['names'][ids[-1]]}")


def main() -> None:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="source dataset dir with dataset.yaml")
    parser.add_argument("--out", type=Path, required=True, help="output dir to upload (cloud-safe)")
    parser.add_argument("--overwrite", action="store_true", help="recreate --out if it already exists")
    args = parser.parse_args()
    prepare(args.dataset, args.out, args.overwrite)


if __name__ == "__main__":
    main()
