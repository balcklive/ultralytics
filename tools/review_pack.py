"""生成人工复核材料包：标准数据集结构 + 每类 crop 文件夹/拼图板 + 整帧标注可视化。

用法（对 auto_label.py run 的输出目录）:
  uv run python tools/review_pack.py --src artifacts/auto_label_record --out data/wgc_review

产出:
  <out>/images/train/ + labels/train/ + dataset.yaml   # 可直接用 yolo_data.py annotate 编辑
  <out>/crops/<拼音类>/...jpg                          # 每类一个 crop 文件夹（全部 1250 框）
  <out>/crops_vis/<拼音类>.png + index.json            # 每类拼图板（格子编号可追溯）
  <out>/labels_vis/<帧>.jpg                            # 整帧全类别画框（颜色区分类别）

人工复核后再跑一遍本脚本即可用最新 labels 重新生成 crops/crops_vis/labels_vis。
labels 直接改动用: uv run python tools/yolo_data.py annotate --dataset <out> --split train
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
FONT_S = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 13)

# 复核可视化配色（BGR），覆盖 73 类中常见的；未列出的用哈希色
PALETTE = {
    0: (255, 255, 255),   # player 白
    2: (255, 128, 0),     # 蜗牛 蓝绿
    3: (255, 160, 0),     # 蓝蜗牛 蓝
    4: (0, 140, 255),     # 蘑菇仔 橙
    5: (0, 80, 120),      # 木妖 棕
    6: (0, 0, 255),       # 红蜗牛 红
    7: (0, 255, 255),     # 花蘑菇 黄
    8: (80, 220, 80),     # 绿水灵 绿
    15: (60, 180, 60),    # 绿蘑菇 深绿
}


def imread_u(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def py(name):
    from pypinyin import lazy_pinyin
    return "".join(lazy_pinyin(name)) or name


def color_of(cls):
    if cls in PALETTE:
        return PALETTE[cls]
    rng = np.random.RandomState(cls)
    return tuple(int(v) for v in rng.randint(60, 256, 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="auto_label.py run 的输出目录（含 images/ labels/）")
    ap.add_argument("--out", required=True, help="复核数据集目录")
    ap.add_argument("--classes-json", default="data/templates/classes.json",
                    help="id->中文名 映射")
    ap.add_argument("--cell", type=int, default=84, help="拼图板格子边长 px")
    ap.add_argument("--cols", type=int, default=10, help="拼图板每行格数")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    names = {int(k): v for k, v in json.loads(
        Path(args.classes_json).read_text(encoding="utf-8")).items()}
    for cid in (0,):
        names.setdefault(cid, "player")

    # 1) 标准数据集结构（images/train, labels/train, dataset.yaml）
    im_train, lb_train = out / "images" / "train", out / "labels" / "train"
    im_train.mkdir(parents=True, exist_ok=True)
    lb_train.mkdir(parents=True, exist_ok=True)
    val_stems = {f.stem for f in (out / "images" / "val").glob("*.jpg")}         if (out / "images" / "val").is_dir() else set()
    for f in sorted((src / "images").glob("*.jpg")):
        dst = (out / "images" / "val" / f.name) if f.stem in val_stems else im_train / f.name
        shutil.copy2(f, dst)
    for f in sorted((src / "labels").glob("*.txt")):
        dst = (out / "labels" / "val" / f.name) if f.stem in val_stems else lb_train / f.name
        shutil.copy2(f, dst)
    (out / "dataset.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\n"
        f"val: images/{'val' if val_stems else 'train'}\n"
        + "names:\n" + "".join(f"  {k}: {v}\n" for k, v in sorted(names.items())),
        encoding="utf-8")

    # 2) 全量 crop 导出（每类一个文件夹）+ 3) 每类拼图板
    crop_root = out / "crops"
    vis_root = out / "crops_vis"
    crop_root.mkdir(parents=True, exist_ok=True)
    vis_root.mkdir(parents=True, exist_ok=True)
    for d in crop_root.iterdir():
        if d.is_dir():
            shutil.rmtree(d)
    index = {}
    n_crops = 0
    label_files = sorted(lb_train.glob("*.txt")) + \
        sorted((out / "labels" / "val").glob("*.txt"))
    for lf in label_files:
        stem = lf.stem
        img = imread_u(im_train / f"{stem}.jpg")
        if img is None:
            img = imread_u(out / "images" / "val" / f"{stem}.jpg")
        if img is None:
            continue
        H, W = img.shape[:2]
        for i, line in enumerate(lf.read_text(encoding="utf-8").splitlines()):
            p = line.split()
            if len(p) < 5:
                continue
            cls = int(p[0])
            cx, cy, w, h = (float(x) for x in p[1:5])
            x1, y1 = max(0, int((cx - w / 2) * W)), max(0, int((cy - h / 2) * H))
            x2, y2 = min(W, int((cx + w / 2) * W)), min(H, int((cy + h / 2) * H))
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            if max(crop.shape[:2]) < 96:
                s = 96 / max(crop.shape[:2])
                crop = cv2.resize(crop, None, fx=s, fy=s,
                                  interpolation=cv2.INTER_NEAREST)
            cname = py(names.get(cls, str(cls)))
            d = crop_root / cname
            d.mkdir(parents=True, exist_ok=True)
            fn = f"{stem}__b{i:02d}_{x1}_{y1}.jpg"
            cv2.imwrite(str(d / fn), crop)
            index.setdefault(cname, []).append(
                {"file": fn, "frame": stem, "cls": cls, "cls_name": names.get(cls),
                 "box": [x1, y1, x2, y2]})
            n_crops += 1

    (vis_root / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")

    CELL, COLS, LAB = args.cell, args.cols, 22
    for cname, items in sorted(index.items()):
        rows = (len(items) + COLS - 1) // COLS
        canvas = np.full((rows * (CELL + LAB) + 6, COLS * (CELL + 4) + 6, 3),
                         255, np.uint8)
        for i, it in enumerate(items):
            c = imread_u(crop_root / cname / it["file"])
            if c is None:
                continue
            r, col = divmod(i, COLS)
            y, x = r * (CELL + LAB) + 4, col * (CELL + 4) + 4
            canvas[y:y + CELL, x:x + CELL] = cv2.resize(
                c, (CELL, CELL), interpolation=cv2.INTER_NEAREST)
            pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            ImageDraw.Draw(pil).text((x + 2, y + CELL + 1), str(i + 1),
                                     font=FONT_S, fill=(200, 0, 0))
            canvas = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(pil).text((canvas.shape[1] - 130, 4),
                                 f"{cname} {len(items)}", font=FONT, fill=(150, 0, 150))
        canvas = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(vis_root / f"{cname}.png"), canvas)

    # 4) 整帧全类别画框
    lvis = out / "labels_vis"
    lvis.mkdir(parents=True, exist_ok=True)
    n_boxes = 0
    for lf in label_files:
        stem = lf.stem
        img = imread_u(im_train / f"{stem}.jpg")
        if img is None:
            img = imread_u(out / "images" / "val" / f"{stem}.jpg")
        if img is None:
            continue
        H, W = img.shape[:2]
        for line in lf.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cls = int(p[0])
            cx, cy, w, h = (float(x) for x in p[1:5])
            x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
            x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
            col = color_of(cls)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, f"{cls}:{py(names.get(cls, '?'))}",
                        (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            n_boxes += 1
        cv2.imwrite(str(lvis / f"{stem}.jpg"), img)

    print(f"数据集: {len(list(im_train.glob('*.jpg')))} 帧 -> {out}")
    print(f"crops: {n_crops} 张 -> {crop_root}")
    print(f"crops_vis: {len(index)} 类拼图板 -> {vis_root}")
    print(f"labels_vis: {len(list(lvis.glob('*.jpg')))} 帧（{n_boxes} 框）-> {lvis}")


if __name__ == "__main__":
    main()
