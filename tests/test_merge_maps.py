from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
_spec = importlib.util.spec_from_file_location("merge_maps", _TOOLS / "merge_maps.py")
merge_maps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(merge_maps)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_read_names_dict_and_list(tmp_path):
    d = _write(tmp_path / "d.yaml", "names:\n  0: player\n  1: a\n")
    assert merge_maps.read_names(d) == {0: "player", 1: "a"}
    l = _write(tmp_path / "l.yaml", "names: [x, y]\n")
    assert merge_maps.read_names(l) == {0: "x", 1: "y"}


def test_image_files_recursive_and_sorted(tmp_path):
    for n in ("b.jpg", "a.png"):
        _write(tmp_path / "images" / "train" / n, "")
    _write(tmp_path / "images" / "train" / "skip.txt", "")
    assert [p.name for p in merge_maps.image_files(tmp_path / "images")] == ["a.png", "b.jpg"]


def test_content_hash_stable(tmp_path):
    p = _write(tmp_path / "i.jpg", "hello")
    assert merge_maps.content_hash(p) == merge_maps.content_hash(p)


def test_label_classes_parses_ids(tmp_path):
    p = _write(tmp_path / "l.txt", "0 0.5 0.5 0.1 0.1\n3 0.2 0.2 0.1 0.1\n")
    assert merge_maps.label_classes(p) == {0, 3}
    assert merge_maps.label_classes(_write(tmp_path / "n.txt", "")) == set()


def test_assign_split_covers_every_class():
    records = [{0}, {1}, {2}, {0, 1}, {2, 3}, {3}, {0}, {1, 2}]
    train, val = merge_maps.assign_split(records, 0.25, seed=42)
    all_classes = set().union(*records)
    val_classes = set().union(*(records[i] for i in val))
    assert val_classes == all_classes  # every class in val
    assert sorted(train + val) == sorted(range(len(records)))  # no dup / no missing
    assert train and val


def test_assign_split_deterministic():
    records = [{0}, {1}, {2}, {0, 1}, {2, 3}, {3}, {0}, {1, 2}]
    assert merge_maps.assign_split(records, 0.25, seed=7) == merge_maps.assign_split(records, 0.25, seed=7)


def test_cli_parser_accumulates_repeated_sources():
    p = merge_maps.build_parser()
    args = p.parse_args(["--out", "o", "--master", "m", "--source", "a", "--source", "b"])
    assert args.source == [Path("a"), Path("b")]
    args2 = p.parse_args(["--out", "o", "--master", "m", "--source", "a", "--map-source", "x:m.yaml",
                          "--source", "b", "--map-source", "y:n.yaml"])
    assert args2.source == [Path("a"), Path("b")]
    assert args2.map_source == ["x:m.yaml", "y:n.yaml"]


def _make_dataset(root: Path, names: dict[int, str], images: list[tuple[str, list[int]]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset.yaml").write_text(
        yaml.safe_dump({"train": "images/train", "val": "images/val", "names": names},
                       allow_unicode=True), encoding="utf-8")
    for split, files in (("train", images), ("val", [])):
        for name, classes in files:
            _write(root / "images" / split / name, "JPEGDATA:" + name)  # unique content per image
            lines = "".join(f"{c} 0.5 0.5 0.1 0.1\n" for c in classes)
            _write(root / "labels" / split / (name.rsplit(".", 1)[0] + ".txt"), lines)


def _args(out: Path, master_yaml: Path, sources: list[Path], **kw) -> object:
    base = {"out": out, "master": master_yaml, "sources": sources,
            "val_fraction": kw.pop("val_fraction", 0.2), "seed": kw.pop("seed", 1),
            "dry_run": kw.pop("dry_run", False), "map_sources": kw.pop("map_sources", {})}
    base.update(kw)
    return type("A", (), base)()


def test_merge_refuses_name_mismatch(tmp_path):
    master = tmp_path / "master"
    src = tmp_path / "src"
    _make_dataset(master, {0: "player", 1: "a"}, [("i.jpg", [0])])
    _make_dataset(src, {0: "player", 1: "WRONG"}, [("j.jpg", [0, 1])])
    with pytest.raises(SystemExit) as exc:
        merge_maps.merge(_args(tmp_path / "out", master / "dataset.yaml", [src]))
    assert "Class names differ" in str(exc.value)


def test_merge_end_to_end_same_table(tmp_path):
    master = tmp_path / "master"
    s1 = tmp_path / "maps" / "射手训练场"
    s2 = tmp_path / "maps" / "另一图"
    _make_dataset(master, {0: "player", 1: "a", 2: "b"}, [("m.jpg", [0])])
    _make_dataset(s1, {0: "player", 1: "a", 2: "b"}, [("s1a.jpg", [1]), ("s1b.jpg", [1, 2])])
    _make_dataset(s2, {0: "player", 1: "a", 2: "b"}, [("s2a.jpg", [2]), ("s2b.jpg", [0])])
    out = tmp_path / "unified"
    merge_maps.merge(_args(out, master / "dataset.yaml", [s1, s2], val_fraction=0.25, seed=42))
    cfg = yaml.safe_load((out / "dataset.yaml").read_text(encoding="utf-8"))
    assert "path" not in cfg                      # cloud-ready
    assert cfg["names"] == {0: "player", 1: "a", 2: "b"}
    train = list((out / "images" / "train").glob("*.jpg"))
    val = list((out / "images" / "val").glob("*.jpg"))
    assert len(train) + len(val) == 4             # no image lost
    val_classes: set[int] = set()
    for img in val:
        val_classes |= merge_maps.label_classes(out / "labels" / "val" / (img.stem + ".txt"))
    assert val_classes == {0, 1, 2}               # every class in val


def test_merge_dedups_identical_frames(tmp_path):
    master = tmp_path / "master"
    s1 = tmp_path / "s1"
    s2 = tmp_path / "s2"
    _make_dataset(master, {0: "player"}, [("m.jpg", [0])])
    # Both maps contain byte-identical image content → only first is kept.
    _make_dataset(s1, {0: "player"}, [("dup.jpg", [0])])
    _make_dataset(s2, {0: "player"}, [("dup.jpg", [0])])
    out = tmp_path / "unified"
    merge_maps.merge(_args(out, master / "dataset.yaml", [s1, s2], val_fraction=0.2, seed=1))
    total = len(list((out / "images" / "train").glob("*.jpg"))) + len(list((out / "images" / "val").glob("*.jpg")))
    assert total == 1  # duplicate frame dropped
