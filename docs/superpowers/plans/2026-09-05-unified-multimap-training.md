# Unified Multi-Map Training Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 本地按「每图一夹」存放训练数据（`data/maps/<地图>/`），新增 `tools/merge_maps.py` 把各图数据合并成一个合规（id 一致、去重、重切 train/val）的统一数据集，并让阿里云 PAI-DLC 的 `cloud/dlc/entrypoint.sh` 支持缓存与更多训练超参，最终一键统一训练。

**Architecture:** 三块——① `tools/yolo_data.py` 的 `train` 子命令加可选超参（`--cache/--patience/--close-mosaic/--cos-lr/--workers`）并转发给 `model.train`；② 新工具 `tools/merge_maps.py`（读 master 73 类表 → 逐源校验 id 一致（不一致拒绝）→ 内容哈希去重 → 按类保证 val 覆盖 → 写无 `path` 的 dataset.yaml，云端/本地都可用）；③ `cloud/dlc/entrypoint.sh` 读可选 env 拼接上述超参，并加 `PRINT_ONLY=1` 供测试打印命令。数据流：`data/maps/*/` + 既有 73 类源 → `merge_maps.py` → `data/unified_<round>/` → `prepare.py`(可选) → `upload.sh` → DLC。

**Tech Stack:** Python 3.12 + uv；Ultralytics YOLO；pytest；bash（Git Bash）；阿里云 PAI-DLC / OSS / ACR。

## Global Constraints

- **73 类主表**：`0=player` + 72 中文怪物，顺序与 `data/wgc_review/dataset.yaml` 完全一致（`data/annotated_73`、`data/new_yolo_73`、`data/hyd5_coldstart` 均一致）。任何源 `names` 与主表不一致时**必须拒绝合并**，不得静默。
- **输出 dataset.yaml 不含 `path:` 键**（云端 Linux 容器 / 本地均以 yaml 目录为根，见 `ultralytics/data/utils.py` `check_det_dataset`）。
- **训练 `--imgsz` 与 C# 运行时推理尺度一致**（当前 640）；改尺度需同步改运行时，否则召回崩（记忆实测）。
- **采用 YOLO26n**；统一大数据集改模型族/分辨率前先评估 C# 运行时成本。
- **Ruff 检查**：`uvx ruff check tools/yolo_data.py tools/merge_maps.py`。
- **CLAUDE.md 全覆盖（全局规则）**：新增/修改源码文件后，同步更新其所在目录及父目录的 `CLAUDE.md`，并随 commit 一并提交。
- **测试**：pytest，根目录有 `addopts="--doctest-modules ..."`；`tools/` 无 `__init__.py`，测试用 `importlib.util.spec_from_file_location` 按路径加载。

---

## File Structure

| 文件 | 职责 | 动作 |
| --- | --- | --- |
| `tools/yolo_data.py` | 训练 CLI：加可选超参并转发给 `model.train` | 修改 |
| `tools/merge_maps.py` | 新增：合并各图数据集 → 统一合规数据集（id 校验/去重/重切/无 path） | 新建 |
| `cloud/dlc/entrypoint.sh` | 云训练入口：读可选 env 拼超参；`PRINT_ONLY` 打印命令 | 修改 |
| `docs/cloud_dlc_training.md` | 统一训练流（prepare→upload→DLC 用 `--cache`） | 修改 |
| `docs/auto_labeling_pipeline.md` | 补「per-map 每图一夹 + merge_maps 并入」步骤 | 修改 |
| `tools/CLAUDE.md` | 新增 `merge_maps.py` 文件说明 | 修改 |
| `cloud/dlc/CLAUDE.md` | 描述 entrypoint 新增 env | 修改 |
| `tests/test_yolo_train.py` | `build_train_kwargs` 单测 | 新建 |
| `tests/test_merge_maps.py` | merge_maps 单测 | 新建 |
| `tests/test_entrypoint.py` | entrypoint `PRINT_ONLY` 验证 | 新建 |

---

### Task 1: yolo_data.py 训练 CLI 支持可选超参

**Files:**
- Modify: `tools/yolo_data.py`（`train()` 函数 ~1001-1014 行；`main()` train 子解析器 ~1089-1098 行）

**Interfaces:**
- Produces: `build_train_kwargs(args) -> dict`（只含非 None 的可选超参；供 `train()` 调 `model.train(**kwargs)`）。可选字段：`cache`(str|None)、`patience`(int|None)、`close_mosaic`(int|None)、`cos_lr`(bool)、`workers`(int|None)。

- [ ] **Step 1: 写失败测试**

Create `tests/test_yolo_train.py`:

```python
from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest

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
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_yolo_train.py -v`
Expected: `ImportError`/`AttributeError` — `build_train_kwargs` 不存在。

- [ ] **Step 3: 实现**

Replace the `train()` function body and add `build_train_kwargs`:

```python
# In tools/yolo_data.py, above train():
def build_train_kwargs(args: argparse.Namespace) -> dict:
    """Build model.train() kwargs from CLI args, forwarding only set optional hyperparams."""
    kwargs: dict = {
        "data": str(Path(args.dataset).resolve() / "dataset.yaml"),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": str(Path(args.project).resolve()),
        "name": args.name,
    }
    if getattr(args, "cos_lr", False):
        kwargs["cos_lr"] = True
    for field in ("cache", "patience", "close_mosaic", "workers"):
        value = getattr(args, field, None)
        if value is not None:
            kwargs[field] = value
    return kwargs


def train(args: argparse.Namespace) -> None:
    """Start current Ultralytics training with the generated dataset."""
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(**build_train_kwargs(args))
```

In `main()` train subparser, add after `fitting.add_argument("--name", ...)`:

```python
    fitting.add_argument("--cache", default=None,
                         help="ultralytics image cache: ram / disk / True (unset=off)")
    fitting.add_argument("--patience", type=int, default=None,
                         help="early-stop patches (unset=ultralytics default)")
    fitting.add_argument("--close-mosaic", type=int, default=None,
                         help="epoch to close mosaics (unset=ultralytics default; 0 disables)")
    fitting.add_argument("--cos-lr", action="store_true",
                         help="enable cosine learning rate (unset/False=off)")
    fitting.add_argument("--workers", type=int, default=None,
                         help="data-loader workers (unset=ultralytics default)")
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_yolo_train.py -v`
Expected: 3 passed。

- [ ] **Step 5: Ruff + 提交**

```bash
uvx ruff check tools/yolo_data.py
git add tools/yolo_data.py tests/test_yolo_train.py
git commit -m "feat(yolo): forward optional train hyperparams (cache/patience/close-mosaic/cos-lr/workers)"
```

---

### Task 2: cloud/dlc/entrypoint.sh 支持可选超参与 PRINT_ONLY

**Files:**
- Modify: `cloud/dlc/entrypoint.sh`

**Interfaces:**
- Consumes: Task 1 新增 CLI（`--cache/--patience/--close-mosaic/--cos-lr/--workers`）。
- Produces: 可选 env `CACHE/PATIENCE/CLOSE_MOSAIC/COS_LR/WORKERS`；`PRINT_ONLY=0|1`（1=只打印命令不执行）。

- [ ] **Step 1: 写失败测试**

Create `tests/test_entrypoint.py`:

```python
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ENTRY = Path(__file__).resolve().parent.parent / "cloud" / "dlc" / "entrypoint.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def _run(env_overrides: dict[str, str]) -> str:
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "DATA_DIR": "/mnt/data/d",
           "BASE_PT": "/mnt/data/models/base/best.pt", "RUN_NAME": "cloud-v1",
           "OUT_DIR": "/mnt/data/out/r1", "PRINT_ONLY": "1", **env_overrides}
    proc = subprocess.run(["bash", str(ENTRY)], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_print_only_default_has_no_optional_flags():
    out = _run({})
    assert "--cache" not in out and "--patience" not in out and "--workers" not in out


def test_print_only_includes_cache_ram():
    assert "--cache ram" in _run({"CACHE": "ram"})


def test_print_only_includes_patience_and_workers():
    out = _run({"PATIENCE": "30", "WORKERS": "8"})
    assert "--patience 30" in out and "--workers 8" in out


def test_print_only_includes_cos_lr_flag():
    assert "--cos-lr" in _run({"COS_LR": "1"})
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_entrypoint.py -v`
Expected: failure（entrypoint 尚未支持这些 env / 无 optional 拼接）。

- [ ] **Step 3: 实现 entrypoint.sh**

Insert after the existing optional-env block (line ~13 `EXPORT_ONNX`), add:

```bash
CACHE="${CACHE:-}"
PATIENCE="${PATIENCE:-}"
CLOSE_MOSAIC="${CLOSE_MOSAIC:-}"
COS_LR="${COS_LR:-0}"
WORKERS="${WORKERS:-}"
PRINT_ONLY="${PRINT_ONLY:-0}"

# Assemble optional train hyperparams (only emit flags, never bare values).
extra_args=()
[ -n "$CACHE" ] && extra_args+=(--cache "$CACHE")
[ -n "$PATIENCE" ] && extra_args+=(--patience "$PATIENCE")
[ -n "$CLOSE_MOSAIC" ] && extra_args+=(--close-mosaic "$CLOSE_MOSAIC")
[ "$COS_LR" = "1" ] && extra_args+=(--cos-lr)
[ -n "$WORKERS" ] && extra_args+=(--workers "$WORKERS")
```

Place `PRINT_ONLY` early-return immediately after building `extra_args` (before the `cd /workspace` / file checks) so tests run without real data:

```bash
if [ "$PRINT_ONLY" = "1" ]; then
  echo "PRINT "$PY" tools/yolo_data.py train --dataset \"$DATA_DIR\" --model \"$BASE_PT\" --epochs \"$EPOCHS\" --imgsz \"$IMGSZ\" --batch \"$BATCH\" --device \"$DEVICE\" --project \"$OUT_DIR/runs\" --name \"$RUN_NAME\" ${extra_args[*]}"
  exit 0
fi
```

Replace the existing `"$PY" tools/yolo_data.py train ...` invocation (lines ~29-37) with the array-spread form:

```bash
"$PY" tools/yolo_data.py train \
  --dataset "$DATA_DIR" \
  --model "$BASE_PT" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --project "$OUT_DIR/runs" \
  --name "$RUN_NAME" \
  "${extra_args[@]}"
```

- [ ] **Step 4: 验证通过**

Run: `bash -n cloud/dlc/entrypoint.sh && uv run pytest tests/test_entrypoint.py -v`
Expected: syntax OK + 4 passed。

- [ ] **Step 5: 提交**

```bash
git add cloud/dlc/entrypoint.sh tests/test_entrypoint.py
git commit -m "feat(dlc): pass optional train hyperparams via env + PRINT_ONLY debug mode"
```

---

### Task 3: merge_maps.py 辅助函数与 assign_split（TDD）

**Files:**
- Create: `tools/merge_maps.py`
- Test: `tests/test_merge_maps.py`

**Interfaces:**
- Produces: `read_names(path)->dict[int,str]`、`image_files(path)->list[Path]`、`content_hash(path)->str`、`label_classes(label)->set[int]`、`assign_split(records: list[set[int]], val_fraction: float, seed: int) -> tuple[list[int], list[int]]`（train/val 索引，deterministic）。

- [ ] **Step 1: 写失败测试**

Append to `tests/test_merge_maps.py` (import loader shared with yolo_data test — reuse pattern):

```python
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


def test_read_names_dict_and_list():
    d = _write(Path("tmp") / "d.yaml", "names:\n  0: player\n  1: a\n")
    assert merge_maps.read_names(d) == {0: "player", 1: "a"}
    l = _write(Path("tmp") / "l.yaml", "names: [x, y]\n")
    assert merge_maps.read_names(l) == {0: "x", 1: "y"}


def test_image_files_recursive_and_sorted(tmp_path):
    for n in ("b.jpg", "a.png"):
        _write(tmp_path / "images" / "train" / n, "")
    _write(tmp_path / "images" / "train" / "skip.txt", "")
    assert [p.name for p in merge_maps.image_files(tmp_path / "images")] == ["a.png", "b.jpg"]


def test_content_hash_stable():
    p = _write(Path("tmp") / "i.jpg", b"hello")
    assert merge_maps.content_hash(p) == merge_maps.content_hash(p)


def test_label_classes_parses_ids():
    p = _write(Path("tmp") / "l.txt", "0 0.5 0.5 0.1 0.1\n3 0.2 0.2 0.1 0.1\n")
    assert merge_maps.label_classes(p) == {0, 3}
    assert merge_maps.label_classes(_write(Path("tmp") / "n.txt", "")) == set()


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
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_merge_maps.py -v`
Expected: `ModuleNotFoundError`（merge_maps 不存在）。

- [ ] **Step 3: 实现辅助函数**

Create `tools/merge_maps.py` with the header docstring + these functions (from the design above):

```python
# tools/merge_maps.py
r"""Merge per-map YOLO dataset folders into one unified dataset for cloud training.

Each source must use the SAME class-id table as --master (default data/wgc_review/
dataset.yaml); a mismatching table is REFUSED unless an explicit --map-source is
given, so classes can never be silently shifted. The output dataset.yaml carries
NO ``path`` key: Ultralytics roots it at the yaml dir, so the same copy works on
the PAI-DLC OSS mount (see docs/cloud_dlc_training.md) and in local training.
Duplicate frames are dropped by content hash; the whole pool is re-split into
train/val with a guarantee that every class appears in val.

Examples:
    python tools/merge_maps.py --out data/unified_20260905 \
        --master data/wgc_review/dataset.yaml \
        --source data/maps/射手训练场 --source data/annotated_73 --source data/new_yolo_73
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
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_merge_maps.py -v`
Expected: 6 passed。

- [ ] **Step 5: Ruff + 提交**

```bash
uvx ruff check tools/merge_maps.py
git add tools/merge_maps.py tests/test_merge_maps.py
git commit -m "feat(maps): merge_maps helpers + deterministic class-aware train/val split"
```

---

### Task 4: merge_maps.py merge() 主体（id 校验 / 去重 / 重切 / 写无 path）

**Files:**
- Modify: `tools/merge_maps.py`（加 `merge()` 与 `refuse_on_name_mismatch()`）
- Test: `tests/test_merge_maps.py`

**Interfaces:**
- Consumes: Task 3 的 `read_names/image_files/content_hash/label_classes/assign_split`；Task 1 未涉及。
- Produces: `merge(args) -> None`（`args`：`out/master/sources/val_fraction/seed/dry_run/map_sources`），写合规数据集并打印摘要；`--map-source`（可选）形如 `SRC:map.yaml`。

- [ ] **Step 1: 写失败测试**

Append to `tests/test_merge_maps.py`:

```python
def _make_dataset(root: Path, names: dict[int, str], images: list[tuple[str, list[int]]]) -> None:
    (root / "dataset.yaml").write_text(
        yaml.safe_dump({"train": "images/train", "val": "images/val", "names": names},
                       allow_unicode=True), encoding="utf-8")
    for split, files in (("train", images), ("val", [])):
        for name, classes in files:
            _write(root / "images" / split / name, b"\xff\xd8")
            if classes:
                lines = "".join(f"{c} 0.5 0.5 0.1 0.1\n" for c in classes)
                _write(root / "labels" / split / (name.rsplit(".", 1)[0] + ".txt"), lines)


def test_merge_refuses_name_mismatch(tmp_path):
    master = tmp_path / "master"
    src = tmp_path / "src"
    _make_dataset(master, {0: "player", 1: "a"}, [("i.jpg", [0])])
    _make_dataset(src, {0: "player", 1: "WRONG"}, [("j.jpg", [0, 1])])
    args = type("A", (), {"out": tmp_path / "out", "master": master / "dataset.yaml",
                          "sources": [src], "val_fraction": 0.2, "seed": 1,
                          "dry_run": False, "map_sources": {}})()
    with pytest.raises(SystemExit) as exc:
        merge_maps.merge(args)
    assert "Class names differ" in str(exc.value)


def test_merge_end_to_end_same_table(tmp_path):
    master = tmp_path / "master"
    s1 = tmp_path / "maps" / "射手训练场"
    s2 = tmp_path / "maps" / "另一图"
    _make_dataset(master, {0: "player", 1: "a", 2: "b"}, [("m.jpg", [0])])
    _make_dataset(s1, {0: "player", 1: "a", 2: "b"}, [("s1a.jpg", [1]), ("s1b.jpg", [1, 2])])
    _make_dataset(s2, {0: "player", 1: "a", 2: "b"}, [("s2a.jpg", [2]), ("s2b.jpg", [0])])
    args = type("A", (), {"out": tmp_path / "unified", "master": master / "dataset.yaml",
                          "sources": [s1, s2], "val_fraction": 0.25, "seed": 42,
                          "dry_run": False, "map_sources": {}})()
    merge_maps.merge(args)
    out = tmp_path / "unified"
    cfg = yaml.safe_load((out / "dataset.yaml").read_text(encoding="utf-8"))
    assert "path" not in cfg                      # cloud-ready
    assert cfg["names"] == {0: "player", 1: "a", 2: "b"}
    train = list((out / "images" / "train").glob("*.jpg"))
    val = list((out / "images" / "val").glob("*.jpg"))
    assert len(train) + len(val) == 4              # no image lost
    # every class appears in val
    val_classes = set()
    for img in val:
        val_classes |= merge_maps.label_classes(out / "labels" / "val" / (img.stem + ".txt"))
    assert val_classes == {0, 1, 2}
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_merge_maps.py -v`
Expected: `AttributeError` — `merge` 未定义。

- [ ] **Step 3: 实现 merge()**

Append to `tools/merge_maps.py`:

```python
def _name_mismatch(master: dict[int, str], source: dict[int, str]) -> list[str]:
    return [
        f"  {c}: master={master.get(c)} source={source.get(c)}"
        for c in sorted(set(master) | set(source))
        if master.get(c) != source.get(c)
    ]


def merge(args) -> None:
    """Merge source datasets into ``args.out`` (id-safe, deduped, re-split, no path)."""
    master_names = read_names(Path(args.master))
    out = Path(args.out)

    records: list[dict] = []          # {image, label, classes, source}
    seen_hashes: set[str] = set()
    per_source: Counter = Counter()

    for src in Path(args.sources) if isinstance(args.sources, list) else [Path(args.sources)]:
        source_names = read_names(src / "dataset.yaml")
        diff = _name_mismatch(master_names, source_names)
        if diff:
            raise SystemExit("Class names differ between datasets; refusing to merge "
                             f"{src}:\n" + "\n".join(diff))
        for split in SPLITS:
            for image in image_files(src / "images" / split):
                lbl = src / "labels" / split / f"{image.stem}.txt"
                digest = content_hash(image)
                if digest in seen_hashes:
                    continue                     # dedup duplicate frame
                seen_hashes.add(digest)
                records.append({
                    "image": image, "label": lbl,
                    "classes": label_classes(lbl),
                    "source": str(src),
                })
                per_source[str(src)] += 1

    class_sets = [r["classes"] for r in records]
    train, val = assign_split(class_sets, args.val_fraction, args.seed)

    if args.dry_run:
        _print_dry_run(args, records, train, val, master_names)
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
                shutil.copy2(rec["label"], dst_lbl)
            else:
                dst_lbl.write_text("", encoding="utf-8")

    cfg = {"train": "images/train", "val": "images/val", "names": master_names}
    (out / "dataset.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                                      encoding="utf-8")
    print(f"unified {out}: images={len(records)} train={len(train)} val={len(val)} sources={len(per_source)}")
    print(f"  sources: {dict(per_source)}")
    print(f"  class box histogram: {_class_histogram(records)}")
```

Add helpers `_print_dry_run` and `_class_histogram`:

```python
def _class_histogram(records: list[dict]) -> dict[int, int]:
    counts: Counter = Counter()
    for rec in records:
        for cls in rec["classes"]:
            counts[cls] += 1
    return dict(sorted(counts.items()))


def _print_dry_run(args, records, train, val, master_names) -> None:
    print(f"--dry-run-- total images={len(records)}")
    print(f"  train={len(train)} val={len(val)} (val_fraction={args.val_fraction})")
    print(f"  class histogram: {_class_histogram(records)}")
    val_classes = set()
    for i in val:
        val_classes |= records[i]["classes"]
    missing = set(master_names) - set().union(*(r["classes"] for r in records) or set())
    uncovered = (set().union(*(r["classes"] for r in records) or set()) - val_classes)
    print(f"  types present overall: {len(set().union(*(r['classes'] for r in records) or set()))}")
    print(f"  classes missing from val: {sorted(uncovered) if uncovered else 'none'}")
    print(f"  classes absent from data entirely: {sorted(missing) if missing else 'none'}")
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_merge_maps.py -v`
Expected: 8 passed。

- [ ] **Step 5: Ruff + 提交**

```bash
uvx ruff check tools/merge_maps.py
git add tools/merge_maps.py tests/test_merge_maps.py
git commit -m "feat(maps): merge_maps merge() with id safety gate, dedup, re-split, pathless yaml"
```

- [ ] **Step 6: 手工 dry-run 校验（真实数据）**

Run (from repo root):
```bash
uv run python tools/merge_maps.py --dry-run \
  --out data/unified_20260905 \
  --master data/wgc_review/dataset.yaml \
  --source data/wgc_review --source data/annotated_73 \
  --source data/new_yolo_73 --source data/hyd5_coldstart
```
Expected: 打印 total images、train/val 数、类别直方图；**若无 `data/maps/*` 目录，先建 `data/maps/` 再把新增 combat 标注并入**（见 Task 6 文档）。确认无误后去掉 `--dry-run` 写盘。

---

### Task 5: merge_maps.py 接入 main() CLI

**Files:**
- Modify: `tools/merge_maps.py`（加 `main()` + `if __name__ == "__main__":`）

**Interfaces:**
- Consumes: Task 4 的 `merge()`。

- [ ] **Step 1: 实现 main()**

Append to `tools/merge_maps.py`:

```python
def main() -> None:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output unified dataset dir")
    parser.add_argument("--master", type=Path, required=True,
                        help="dataset.yaml providing the canonical 73-class names table")
    parser.add_argument("--source", type=Path, nargs="+", required=True,
                        help="source dataset dirs (each with its own dataset.yaml)")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="print counts/histogram/coverage without writing anything")
    parser.add_argument("--map-source", nargs="*", default=[],
                        metavar="SRC:MAP.yaml",
                        help="optional id remap yaml {old_id: new_id} for a source whose table differs")
    args = parser.parse_args()
    map_sources: dict[str, Path] = {}
    for spec in args.map_source:
        src, _, map_path = spec.partition(":")
        map_sources[src] = Path(map_path)
    merge(type("A", (), vars(args))(map_sources=map_sources))


if __name__ == "__main__":
    main()
```

> 说明：`merge()` 的 `map_sources` 是备用能力（18 类拼音源默认不合并）。当前 `merge()` 未消费 `map_sources`；要让 `merge_maps` 真正支持重映射，需在 `merge()` 里对匹配 `map_sources` 的源加载映射并改写 label 行——**本计划默认不启用**（18 类源排除），如需启用再补 Task 5a。

- [ ] **Step 2: 验证 CLI 可用（dry-run）**

Run: `uv run python tools/merge_maps.py --help`
Expected: 显示帮助、无报错。

- [ ] **Step 3: Ruff + 提交**

```bash
uvx ruff check tools/merge_maps.py
git add tools/merge_maps.py
git commit -m "feat(maps): wire merge_maps CLI (--out/--master/--source/--dry-run/--map-source)"
```

---

### Task 6: 文档与 CLAUDE.md 更新（全局规则）

**Files:**
- Modify: `docs/cloud_dlc_training.md`、`docs/auto_labeling_pipeline.md`、`tools/CLAUDE.md`、`cloud/dlc/CLAUDE.md`

**Interfaces:** -（纯文档，承接上面的工具行为）

- [ ] **Step 1: tools/CLAUDE.md — 新增 merge_maps 说明**

In the file list table, add a row:

```
| `merge_maps.py` | 把各图（data/maps/*）达标数据集合并成统一训练集：逐源校验 names==master（不一致拒绝，防类别错位）、内容哈希去重、按类保证 val 覆盖稀有类、写不含 path 的 dataset.yaml（本地/云通用） |
```

- [ ] **Step 2: cloud/dlc/CLAUDE.md — 描述 entrypoint 新 env**

In "关键规则 / 注意事项" or file list `entrypoint.sh` row, append:

```
| `entrypoint.sh` | 入口读 env ...（EPOCHS/IMGSZ/BATCH/DEVICE/EXPORT_ONNX，另可选 `CACHE/PATIENCE/CLOSE_MOSAIC/COS_LR/WORKERS` → `tools/yolo_data.py train` 对应超参；`PRINT_ONLY=1` 仅打印命令不执行） |
```

- [ ] **Step 3: docs/cloud_dlc_training.md — 补统一训练流**

Add a §4.5「统一大数据集训练」subsection: 三个要点——① 合并 `data/maps/*/` + 既有 73 类源 → `merge_maps.py` → `data/unified_<round>`（无 path，直接上传）；② 提交 DLC 时给环境变量 `CACHE=ram`（大数据集从 OSS 逐 epoch 全量读图是主要瓶颈，`--cache ram` 首轮后驻留内存）；③ 机型 A10/抢占式，`EPOCHS/PATIENCE` 按数据量调。并把 step ③ 环境变量表的 `CACHE` 行加进去，标注默认 `-`（不设则不缓存）。

- [ ] **Step 4: docs/auto_labeling_pipeline.md — 补 per-map + merge 步骤**

在第 8 步（人工启动训练）前，加一句：「新域/新图数据先落 `data/maps/<地图>/`（每图一夹），后用 `merge_maps.py` 并入统一训练集」。并把 §9.5 的云化一行改成指向 `merge_maps.py` 先合并再上云。

- [ ] **Step 5: Ruff 无关 + 提交**

```bash
git add docs/cloud_dlc_training.md docs/auto_labeling_pipeline.md tools/CLAUDE.md cloud/dlc/CLAUDE.md
git commit -m "docs: unified multi-map flow + merge_maps tool + entrypoint hyperparam env"
```

---

### Task 7: 端到端验证（真实数据 + 云端命令）

**Files:** -（验证与交接，无源码改动）

- [ ] **Step 1: 全量单测 + Ruff**

Run:
```bash
uv run pytest tests/test_merge_maps.py tests/test_yolo_train.py tests/test_entrypoint.py -v
uvx ruff check tools/yolo_data.py tools/merge_maps.py
```
Expected: 全部 passed，ruff 无告警。

- [ ] **Step 2: 本地合并真实数据（dry-run → 实写）**

Run:
```bash
uv run python tools/merge_maps.py --dry-run --out data/unified_20260905 \
  --master data/wgc_review/dataset.yaml \
  --source data/wgc_review --source data/annotated_73 \
  --source data/new_yolo_73 --source data/hyd5_coldstart
# 确认直方图/覆盖无误后（去掉 --dry-run 实写）：
uv run python tools/merge_maps.py --out data/unified_20260905 \
  --master data/wgc_review/dataset.yaml \
  --source data/wgc_review --source data/annotated_73 \
  --source data/new_yolo_73 --source data/hyd5_coldstart
```
Expected: `train=N val=M`，val 覆盖所有出现过的类（uncovered none）。

- [ ] **Step 3: 云端一次性验证（用户执行）**

```bash
BUCKET=mxdzlk-yolo-train ./cloud/dlc/upload.sh 20260905-unified data/unified_20260905 weights/20260902/best.pt
# DLC job 额外 env: CACHE=ram  (PRINT_ONLY=1 可在本地先打印命令确认)
```
Expected: OSS/`datasets/20260905-unified` 与 `models/base/` 就位；DLC 跑通，输出回 `out/20260905-unified`。

- [ ] **Step 4: 汇总记录**

在 memory（`cloud-training-fullv1.md` 或新建）记录本轮：`merge_maps.py` 上线、统一训练集规模、云端 `--cache` 生效后的耗时。并更新 `weights/<日期>/` 归档。

---

## Self-Review

- **Spec coverage**：每图一夹（Task 6 文档 + Task 4/5 合并）✅；id 一致性安全门（Task 4 `refuse_on_name_mismatch`）✅；去重（Task 4 content_hash）✅；重切 train/val 且保证稀有类进 val（Task 3 `assign_split`）✅；无 path dataset.yaml（Task 4）✅；云上 `--cache` + 超参透传（Task 1/2）✅；CLAUDE.md 更新（Task 6）✅。
- **Placeholder scan**：无 TBD/TODO；每步含具体代码与测试。
- **Type consistency**：`build_train_kwargs`、`assign_split`、`merge(args)`、`read_names` 等在后续任务引用一致；`merge()` 的 `map_sources` 明确标注为备用（本计划不启用），无跨任务悬空引用。
- **已知限制（明示）**：18 类拼音源（`new_yolo_dataset`/`new_yolo_review`）与 `datasets/mxdzlk_cmsc` 默认**不并入**（表不一致），需单独人工映射后才可进统一集，避免拼音误映射历史前科。`merge()` 暂不消费 `--map-source` 重映射。
