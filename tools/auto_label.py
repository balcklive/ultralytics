"""自动标注管线：图片/录像 → YOLO 候选 → 模板分类 → VLM 仲裁 → YOLO 标签。

分层架构（依据 2026-08-30 实测：中心裁剪模板分类 top-1 97.1%、
YOLO conf=0.05 类别无关召回 79.3%、VLM 73 类直判仅 52.5%）：
1. 抽帧（视频按 fps 采样）+ 近重复帧剔除（dHash）；
2. YOLO 低置信度出候选框（只当"哪里可能有怪"，不带类别）；
3. 中心裁剪模板匹配给候选框打类别分（多模板取最优）；
4. 分数分层：>=high 阈值直接采信；[low, high) 交给 VLM 对照 monster_instances
   实例图鉴做"以图搜图"仲裁（Codex 后端，禁止纯文字描述推理）；<low 丢弃；
5. 输出 YOLO txt 标签 + review 清单（低置信 VLM 结果落到 review 目录）。

高置信误判加固（2026-09-01，方案1+3）：
- 决策级联的直采路径只采信"非易混类且非静态"的框。
- --vlm-recheck-classes：易混类（换色家族/绿色背景易误类）即使 YOLO 高置信
  也降级送 VLM 复核——实测高置信直采的绿水灵框 97% 实为静态花草（见
  docs/auto_labeling_pipeline.md 结论 14）。
- --static-thresh：静态背景剔除，帧间运动量低于阈值的框（花草/UI/装饰）
  不直接采信，仅视频源生效。

用法:
  # 图片目录
  uv run python tools/auto_label.py run --source data/new_images --model weights/20260823/best.pt
  # 录像（易混类降级 + 静态剔除）
  uv run python tools/auto_label.py run --source game.mp4 --fps 2 --model ... \
      --vlm-recheck-classes "蓝蜗牛,红蜗牛,绿水灵" --static-thresh 3
  # 构建模板库
  uv run python tools/auto_label.py build-templates
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT / "data" / "templates"


def imread_u(path):
    """兼容中文路径的 imread（Windows 下 cv2.imread 不支持非 ASCII 路径）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def py(name):
    """中文类名转拼音（cv2.putText 与文件名只支持 ASCII；中文仅存于 json 数据中）。"""
    from pypinyin import lazy_pinyin
    return "".join(lazy_pinyin(name)) or name

# 置信度分层（模板 NCC 分数）
HIGH_THRESH = 0.93   # 直接采信
LOW_THRESH = 0.85    # 低于此丢弃，介于两者之间交 VLM
CENTER = 0.5         # 中心裁剪比例
SLIDE = 6            # 分类时允许的框偏移余量 px

PROMPT = (
    "第1张图是怪物图鉴，每格上方有黄色编号（1-%d）。第2张图是若干待分类的"
    "游戏怪物裁剪图，每格上方有黄色 T 编号。请对每个 T 编号的裁剪图，在图鉴中"
    "找到外观最匹配的怪物格子（只看怪物本体：颜色、壳型、五官、体型；忽略背景"
    "和拍摄角度差异）。如果整格是背景/杂物/半截无主体的图，或没有可信匹配，"
    "cell 填 -1。只输出 JSON："
    '{"T1":{"cell":编号,"conf":0-1},...}，覆盖所有 T 编号，不要其他文字。'
)


# ---------- 模板库 ----------

def load_classes(dataset_yaml):
    with open(dataset_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {int(k): v for k, v in data["names"].items()}


def build_template_library(datasets, out_dir=TEMPLATE_DIR, max_per_class=8):
    """从多个已标注数据集聚合裁剪模板，按内容去重后存盘。

    datasets: [(images_dir, labels_dir, dataset_yaml, class_id_offset), ...]。
    各数据集类别 id 空间可能不同（如 mxdzlk_cmsc 无 player、怪物 id 比
    annotated_73 小 1），用 offset 统一到模型 id 空间。存盘文件名
    f"{cls:03d}_{idx}.png"，同时写 classes.json 记录 id->名称。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_cls = defaultdict(list)
    seen = {}
    names = {}
    for images_dir, labels_dir, dataset_yaml, offset in datasets:
        names.update(load_classes(dataset_yaml))
        d = Path(images_dir)
        for img_path in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
            label_path = Path(labels_dir) / (img_path.stem + ".txt")
            img = imread_u(img_path)
            if img is None or not label_path.exists():
                continue
            H, W = img.shape[:2]
            for line in label_path.read_text(encoding="utf-8").splitlines():
                p = line.strip().split()
                if len(p) < 5:
                    continue
                cls = int(p[0]) + offset
                cx, cy, w, h = (float(x) for x in p[1:5])
                x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
                x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
                tpl = img[y1:y2, x1:x2]
                if tpl.size == 0:
                    continue
                key = (cls, cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY).tobytes())
                if key in seen:
                    continue
                seen[key] = True
                if len(by_cls[cls]) < max_per_class:
                    by_cls[cls].append(tpl)
    for f in out_dir.glob("*.png"):
        f.unlink()
    counts = {}
    for cls, tpls in sorted(by_cls.items()):
        for i, t in enumerate(tpls):
            cv2.imwrite(str(out_dir / f"{cls:03d}_{i:02d}.png"), t)
        counts[names.get(cls, cls)] = len(tpls)
    (out_dir / "classes.json").write_text(
        json.dumps(names, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"模板库: {len(by_cls)} 类 {sum(len(v) for v in by_cls.values())} 个 -> {out_dir}")
    return by_cls


def load_template_library(out_dir=TEMPLATE_DIR):
    """加载模板库，返回 (names, {cls: [灰度中心裁剪模板]})，模板已预裁剪缓存。"""
    names = json.loads((Path(out_dir) / "classes.json").read_text(encoding="utf-8"))
    names = {int(k): v for k, v in names.items()}
    by_cls = defaultdict(list)
    for f in sorted(Path(out_dir).glob("*.png")):
        head = f.stem.split("_")[0]
        if not head.isdigit():
            continue  # 跳过临时文件等非模板文件
        img = imread_u(f)
        if img is None:
            continue
        by_cls[int(head)].append(_cc(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
    return names, dict(by_cls)


# ---------- 核心分类 ----------

def _cc(gray, ratio=CENTER):
    h, w = gray.shape
    y0, x0 = int(h * (1 - ratio) / 2), int(w * (1 - ratio) / 2)
    return gray[y0:h - y0, x0:w - x0]


def classify_crop(img, box, templates, scales=(1.0, 0.8, 0.65)):
    """对候选框做多尺度中心裁剪模板分类。

    候选框（YOLO 紧贴框，精灵占满）与模板（如 42x42 标记框中心裁剪，
    精灵只占约 60~80%）中精灵占比不同，把测试裁剪按 scales 缩放对齐后
    与各模板比对取最优。同尺寸模板共享一次缩放（resize 结果按目标尺寸缓存）。
    返回 [(score, cls), ...] 按分数降序（只含有模板的类）。
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    H, W = img.shape[:2]
    crop = img[max(0, y1 - SLIDE):min(H, y2 + SLIDE),
               max(0, x1 - SLIDE):min(W, x2 + SLIDE)]
    if crop.size == 0:
        return []
    cg = _cc(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    resized = {}  # (tW,tH) -> 缩放+补边后的测试裁剪，避免同尺寸模板重复缩放
    scores = []
    for cls, tpls in templates.items():
        best = -1.0
        for g in tpls:
            gh, gw = g.shape
            for k in scales:
                tW, tH = max(int(gw * k), 8), max(int(gh * k), 8)
                if (tW, tH) not in resized:
                    c = cv2.resize(cg, (tW, tH), interpolation=cv2.INTER_AREA
                                   if k < 1 else cv2.INTER_CUBIC)
                    pad_h, pad_w = max(0, gh - tH) + 3, max(0, gw - tW) + 3
                    resized[(tW, tH)] = cv2.copyMakeBorder(
                        c, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
                c = resized[(tW, tH)]
                if c.shape[0] < gh or c.shape[1] < gw:
                    continue
                best = max(best, float(cv2.matchTemplate(c, g, cv2.TM_CCOEFF_NORMED).max()))
        if best > 0:
            scores.append((best, cls))
    return sorted(scores, reverse=True)


# ---------- VLM 仲裁（以图搜图，Codex 后端，对照 monster_instances 图鉴） ----------

def load_atlas(args):
    """加载/构建 monster_instances 实例图鉴。返回 (图鉴jpeg b64, 格号->类id)。"""
    import base64

    atlas_path = TEMPLATE_DIR / "atlas.png"
    if not atlas_path.exists():
        build_atlas(args.instances, args.player, atlas_path)
    names, _ = load_template_library()
    mapping = {int(k): v for k, v in json.loads(
        (TEMPLATE_DIR / "atlas.json").read_text(encoding="utf-8")).items()}
    # 图鉴目录名是下划线拼音（monster_instances 约定），归一到类 id
    from pypinyin import lazy_pinyin
    us2id = {"_".join(lazy_pinyin(nm)): cid for cid, nm in names.items()}
    cell2id = {no: us2id.get(name) for no, name in mapping.items()}
    atlas_b64 = base64.b64encode(
        cv2.imencode(".jpg", imread_u(atlas_path))[1].tobytes()).decode()
    return atlas_b64, cell2id


def _crop_img(img, box, margin=10, min_side=96):
    """按框裁剪（带余量），短边不足 min_side 时放大，供图鉴匹配。"""
    x1, y1, x2, y2 = (int(v) for v in box)
    crop = img[max(0, y1 - margin):y2 + margin, max(0, x1 - margin):x2 + margin]
    if crop.size and max(crop.shape[:2]) < min_side:
        s = min_side / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    return crop


def _sheet_b64(imgs, cell=112, lab=24, cols=3):
    """把若干裁剪图拼成带 T 编号的网格图并转 jpeg base64。"""
    import base64

    rows = (len(imgs) + cols - 1) // cols
    sheet = np.full((rows * (cell + lab) + 4, cols * cell + 4, 3), 30, np.uint8)
    for i, img in enumerate(imgs):
        if img is None:
            continue
        s = min(cell / img.shape[1], cell / img.shape[0])
        im = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)),
                        interpolation=cv2.INTER_NEAREST)
        r, c = divmod(i, cols)
        y0, x0 = r * (cell + lab) + 2, c * cell + 2
        sheet[y0 + lab:y0 + lab + im.shape[0], x0:x0 + im.shape[1]] = im
        cv2.putText(sheet, f"T{i+1}", (x0 + 2, y0 + lab - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    _, buf = cv2.imencode(".jpg", sheet)
    return base64.b64encode(buf.tobytes()).decode()


def vlm_arbitrate_batch(crops, atlas_b64, cell2id, model="gpt-5.6-luna"):
    """一批裁剪图对照实例图鉴做"以图搜图"仲裁（Codex 后端，一次调用判整批）。

    crops 与 T 编号一一对应。返回 [(cid|None, conf), ...]；
    无可信匹配(cell=-1)或解析失败为 (None, 0.0)。
    不传文字候选/描述——实测文字推理会自信地错（73 类直判仅 52.5%），
    必须以实例图鉴为唯一类别依据。
    """
    out = _codex_vlm([atlas_b64, _sheet_b64(crops)], PROMPT % max(cell2id), model=model)
    res = [(None, 0.0)] * len(crops)
    m = re.search(r"\{.*\}", out or "", re.DOTALL)
    if not m:
        return res
    try:
        verdicts = json.loads(m.group(0))
    except json.JSONDecodeError:
        return res
    for j in range(len(crops)):
        v = verdicts.get(f"T{j + 1}", {})
        try:
            cid = cell2id.get(int(v.get("cell", -1)))
        except (TypeError, ValueError):
            continue
        if cid is not None:
            res[j] = (cid, float(v.get("conf", 0) or 0))
    return res


# ---------- 帧提取 ----------

def iter_frames(source, fps=None):
    """产出 (name, BGR)。目录则遍历图片，视频按 fps 抽帧并剔除近重复帧。"""
    src = Path(source)
    if src.is_dir():
        for p in sorted(src.glob("*.png")) + sorted(src.glob("*.jpg")):
            img = imread_u(p)
            if img is not None:
                yield p.stem, img
        return
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {src}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(src_fps / fps)) if fps else 1
    prev_hash = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            small = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (8, 8),
                interpolation=cv2.INTER_AREA)
            h = (small > small.mean()).tobytes()  # 8x8 均值阈值哈希，剔除近重复帧
            if h != prev_hash:
                prev_hash = h
                yield f"{src.stem}_f{idx:06d}", frame
        idx += 1
    cap.release()


# ---------- 主流程 ----------

def _parse_classes(names, spec, arg_name):
    """把 '类名/id,类名/id' 字符串解析为类别 id 集合（空串返回 None）。"""
    allow = set()
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if tok.isdigit():
            allow.add(int(tok))
        elif tok:
            allow.update(cid for cid, nm in names.items() if nm == tok)
    return allow if allow else None


class _BgModel:
    """滚动背景中值模型（静态背景剔除用，复用 frame_review 的运动热图思路）。

    维护最近 window 帧的 1/4 分辨率灰度图，取像素级中值作为背景。
    怪物在动画/移动，会偏离中值；静态花草/UI 与中值几乎一致 → 运动量≈0。
    只对视频源启用（图片目录各帧无时间连续性，中值无意义）。
    """

    def __init__(self, window=8, scale=4):
        self.window = window
        self.scale = scale
        self._buf = []
        self.bg = None

    def add(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = g.shape[:2]
        self._buf.append(cv2.resize(g, (w // self.scale, h // self.scale)))
        if len(self._buf) > self.window:
            self._buf.pop(0)
        if len(self._buf) >= 3:  # 足够帧才出背景，前几帧不启用剔除
            self.bg = np.median(np.stack(self._buf), axis=0).astype(np.uint8)
        return self.bg

    def motion(self, box):
        """框内 |当前帧-背景| 灰度均值（在 1/4 分辨率上算，返回 0~255）。"""
        if self.bg is None:
            return None
        g = self._buf[-1]
        bh, bw = self.bg.shape[:2]
        x1, y1 = max(0, int(box[0]) // self.scale), max(0, int(box[1]) // self.scale)
        x2, y2 = min(bw, int(box[2]) // self.scale), min(bh, int(box[3]) // self.scale)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        return float(cv2.absdiff(g[y1:y2, x1:x2], self.bg[y1:y2, x1:x2]).mean())


def run_pipeline(args):
    from ultralytics import YOLO

    names, templates = load_template_library()
    atlas_b64, cell2id = load_atlas(args)
    model = YOLO(args.model)
    out_img = Path(args.out) / "images"
    out_lbl = Path(args.out) / "labels"
    out_review = Path(args.out) / "review"
    for d in (out_img, out_lbl, out_review):
        d.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)

    # 地图怪物白名单：来自 bot-cs data/maps/names.json 的"地图→出没表"。
    # 不在清单里的框（含高置信）不再直接采信，降级送 VLM——
    # 实测新模型会把底部 UI 按钮 0.5+ 置信度判成"冰龙"，白名单是硬约束。
    allow = _parse_classes(names, getattr(args, "map_classes", ""), "map-classes")
    if allow:
        print(f"地图白名单: {sorted(names[c] for c in allow)}")

    # 方案1：易混类白名单。换色家族（蓝/红蜗牛、蘑菇仔/花/绿蘑菇）与绿色背景
    # 易误类（绿水灵）即使 YOLO 高置信也会自信地错（实测直采 97% 绿水灵框实为
    # 静态花草），这些类高置信不再直接采信，降级送 VLM 迷你图鉴复核。
    vlm_recheck = _parse_classes(names, getattr(args, "vlm_recheck_classes", ""),
                                 "vlm-recheck-classes")
    if vlm_recheck:
        print(f"VLM 必审类(高置信降级): {sorted(names[c] for c in vlm_recheck)}")

    # 方案3：静态背景剔除。仅视频源启用（图片目录无时间连续性）。
    is_video = not Path(args.source).is_dir()
    bg = _BgModel(window=getattr(args, "bg_window", 8)) if is_video else None
    static_thresh = getattr(args, "static_thresh", 0.0)
    if bg and static_thresh > 0:
        print(f"静态背景剔除: 帧间运动量 < {static_thresh} 的框不直接采信（送 VLM 复核）")

    for name, img in iter_frames(args.source, args.fps):
        if bg is not None:
            bg.add(img)
        H, W = img.shape[:2]
        r = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        labels, review, pending = [], [], []
        for bi, (box, ycls, yconf) in enumerate(zip(
                r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                r.boxes.conf.tolist())):
            scores = classify_crop(img, box, templates)
            tpl_top = scores[0] if scores else None
            in_recheck = vlm_recheck is not None and int(ycls) in vlm_recheck
            # 静态框判定：帧间几乎不动的框（花草/UI/装饰）不做自动采信。
            # 怪物有动画/移动会偏离中值背景；静态花草与中值一致。
            is_static = bool(bg and static_thresh > 0) and \
                (m := bg.motion(box)) is not None and m < static_thresh
            if not in_recheck and not is_static and \
                    yconf >= args.high_conf and (allow is None or int(ycls) in allow):
                # YOLO 高置信且在地图白名单内：直接采信（实测类准确率 96.8%）
                labels.append((int(ycls), box, yconf))
                stats["auto_yolo_high"] += 1
            elif not in_recheck and not is_static and tpl_top and \
                    tpl_top[0] >= HIGH_THRESH and (allow is None or tpl_top[1] in allow):
                # 模板高分：直接采信（合成域/干净精灵的强信号）
                labels.append((tpl_top[1], box, tpl_top[0]))
                stats["auto_template"] += 1
            elif not in_recheck and not is_static and tpl_top and \
                    int(ycls) == tpl_top[1] and yconf >= args.agree_conf \
                    and (allow is None or int(ycls) in allow):
                # 双源一致
                labels.append((int(ycls), box, yconf))
                stats["auto_agree"] += 1
            elif yconf >= args.vlm_conf:
                # 疑难（含被降级的易混类/静态框）：攒一帧的量做一次批量图鉴仲裁
                pending.append((bi, box, scores, ycls, yconf))
                if in_recheck:
                    stats["downgraded_recheck"] += 1
                elif is_static:
                    stats["downgraded_static"] += 1
            else:
                stats["rejected"] += 1

        if pending:
            crops = [_crop_img(img, box) for _, box, _, _, _ in pending]
            verdicts = vlm_arbitrate_batch(crops, atlas_b64, cell2id)
            for (_, box, scores, ycls, yconf), (cid, vconf) in zip(pending, verdicts):
                if cid is not None:
                    if args.write_vlm:
                        # 实测 VLM 仲裁有错判，默认不直接写入标签（只落 review/）
                        labels.append((cid, box, yconf))
                    review.append({"box": [round(v) for v in box],
                                   "yolo_cls": names.get(int(ycls)),
                                   "yolo_conf": round(yconf, 3),
                                   "tpl_top3": [names[c] for _, c in scores[:3]],
                                   "vlm_cls": names.get(cid), "vlm_conf": vconf})
                    stats["vlm_arbitrated"] += 1
                else:
                    # VLM 图鉴也认不出：仍落 review（vlm_cls 空），由人工/export 处理
                    review.append({"box": [round(v) for v in box],
                                   "yolo_cls": names.get(int(ycls)),
                                   "yolo_conf": round(yconf, 3),
                                   "tpl_top3": [names[c] for _, c in scores[:3]],
                                   "vlm_cls": None, "vlm_conf": 0.0})
                    stats["vlm_unmatched"] += 1
        # 写出
        cv2.imwrite(str(out_img / f"{name}.jpg"), img)
        lines = []
        for cid, (bx1, by1, bx2, by2), _ in labels:
            cx, cy = (bx1 + bx2) / 2 / W, (by1 + by2) / 2 / H
            bw, bh = (bx2 - bx1) / W, (by2 - by1) / H
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out_lbl / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        if review:
            (out_review / f"{name}.json").write_text(
                json.dumps(review, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{name}: {len(labels)} 标签, {len(review)} 待复核")

    print("\n== 汇总 ==")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


def export_review(args):
    """把 run 产生的 review/*.json 转成便于人工浏览的结构。

    输出（写到 --out 的 review_vis/ 与 crops/）：
    - review_vis/<帧>.jpg：整帧可视化，绿=已入库自动标签，橙=待复核（含 VLM 建议）；
    - crops/<拼音类名>/xxx.jpg：按 VLM 建议类别归类的待复核裁剪图（放大到 96px），
      文件名含帧号与 YOLO 原判断（拼音），方便按类集中核对（如蜗牛家族）。
    图上文字与目录/文件名一律用拼音：cv2.putText 与文件系统展示环节
    均不支持中文（中文仍保留在 review/*.json 数据里）。
    """
    names, _ = load_template_library()
    root = Path(args.out)
    vis_dir = root / "review_vis"
    crop_dir = root / "crops"
    vis_dir.mkdir(parents=True, exist_ok=True)
    n_vis = n_crop = 0
    for rj in sorted((root / "review").glob("*.json")):
        stem = rj.stem
        img_path = next((root / "images").glob(f"{stem}.*"), None)
        if img_path is None:
            continue
        img = imread_u(img_path)
        entries = json.loads(rj.read_text(encoding="utf-8"))
        # 叠加已入库标签（绿）
        lbl = root / "labels" / f"{stem}.txt"
        if lbl.exists() and img is not None:
            H, W = img.shape[:2]
            for line in lbl.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                cls = int(p[0])
                cx, cy, w, h = (float(x) for x in p[1:5])
                x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
                x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(img, f"{cls}:{py(names.get(cls, '?'))}",
                            (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 220, 0), 2)
        for e in entries:
            x1, y1, x2, y2 = e["box"]
            vlm_cls = e.get("vlm_cls") or "未知"
            yolo_cls = e.get("yolo_cls") or "?"
            if img is not None:
                # 先裁剪（不带框线），再在整帧图上画框
                crop = img[max(0, y1 - 6):y2 + 6, max(0, x1 - 6):x2 + 6]
                if crop.size:
                    if max(crop.shape[:2]) < 96:
                        s = 96 / max(crop.shape[:2])
                        crop = cv2.resize(crop, None, fx=s, fy=s,
                                          interpolation=cv2.INTER_NEAREST)
                    d = crop_dir / py(vlm_cls)
                    d.mkdir(parents=True, exist_ok=True)
                    fn = f"{stem}_yolo-{py(yolo_cls)}_{x1}_{y1}.jpg"
                    cv2.imwrite(str(d / fn), crop)
                    n_crop += 1
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(img, f"{py(vlm_cls)}? (yolo:{py(yolo_cls)})",
                            (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 165, 255), 1)
        if img is not None:
            cv2.imwrite(str(vis_dir / f"{stem}.jpg"), img)
            n_vis += 1
    print(f"可视化 {n_vis} 帧 -> {vis_dir}")
    print(f"待复核裁剪 {n_crop} 张（按 VLM 建议类归类）-> {crop_dir}")


# ---------- VLM 对照实例图自动复核 ----------

def build_atlas(exemplar_root, player_img=None, out=TEMPLATE_DIR / "atlas.png",
                only=None):
    """把每类实例图拼成带编号索引的图鉴（供 VLM 以图搜类）。

    每格上方标注全局索引号（1 起）。返回 (索引->拼音目录名) 映射并写
    atlas_index.json；player 图（如提供）占最后一格，映射名为 'player'。
    only: 目录名（下划线拼音）白名单；给定则只含这些类（+player）。
    干扰格越少 VLM 越准——全图鉴下红蜗牛曾被匹配到火野猪格（conf 0.86），
    地图白名单 8 格迷你图鉴实测 85% 准确。
    """
    import os
    dirs = sorted(d for d in os.listdir(exemplar_root)
                  if os.path.isdir(os.path.join(exemplar_root, d))
                  and (only is None or d in only))
    cells = []  # (索引, 目录名, 图片路径)
    idx = 1
    for d in dirs:
        for f in sorted(os.listdir(os.path.join(exemplar_root, d))):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                cells.append((idx, d, os.path.join(exemplar_root, d, f)))
                idx += 1
                break  # 每类取一张
    if player_img:
        cells.append((idx, "player", str(player_img)))
        idx += 1

    CELL_W, CELL_H, LABEL_H = 96, 96, 26
    COLS = 8
    rows = (len(cells) + COLS - 1) // COLS
    atlas = np.full((rows * (CELL_H + LABEL_H) + 8, COLS * CELL_W + 8, 3),
                    30, dtype=np.uint8)
    mapping = {}
    for i, (no, name, path) in enumerate(cells):
        img = imread_u(path)
        if img is None:
            continue
        s = min(CELL_W / img.shape[1], CELL_H / img.shape[0])
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)),
                         interpolation=cv2.INTER_NEAREST)
        r, c = divmod(i, COLS)
        y0, x0 = r * (CELL_H + LABEL_H) + 4, c * CELL_W + 4
        atlas[y0 + LABEL_H:y0 + LABEL_H + img.shape[0],
              x0:x0 + img.shape[1]] = img
        cv2.putText(atlas, str(no), (x0 + 2, y0 + LABEL_H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        mapping[no] = name
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), atlas)
    out.with_suffix(".json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"图鉴: {len(mapping)} 格 -> {out}")
    return mapping


def _codex_vlm(images_b64, prompt, model="gpt-5.6-luna"):
    """经 Codex 后端调 GPT 视觉模型（复用 ~/.codex/auth.json 登录态）。

    参考 artifacts/test_openai_vision.py。返回文本或 None。
    """
    import os
    import uuid

    os.environ.setdefault("http_proxy", "http://127.0.0.1:7890")
    os.environ.setdefault("https_proxy", "http://127.0.0.1:7890")
    from openai import OpenAI

    auth = json.load(open(os.path.expanduser("~/.codex/auth.json")))
    client = OpenAI(
        base_url="https://chatgpt.com/backend-api/codex",
        api_key=auth["tokens"]["access_token"],
        default_headers={
            "chatgpt-account-id": auth["tokens"].get("account_id", ""),
            "originator": "codex_cli_rs",
            "OpenAI-Beta": "responses=experimental",
            "session_id": str(uuid.uuid4()),
        },
    )
    content = [{"type": "input_text", "text": prompt}]
    for b64 in images_b64:
        content.append({"type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{b64}"})
    try:
        for attempt in range(3):  # 代理链路偶发中断（incomplete chunked read），重试
            try:
                parts = []
                with client.responses.stream(
                        input=[{"role": "user", "content": content}],
                        model=model, store=False) as stream:
                    for event in stream:
                        if event.type == "response.output_text.delta":
                            parts.append(event.delta)
                return "".join(parts)
            except Exception as e:
                if attempt == 2:
                    print(f"  [VLM 调用失败] {e}")
                    return None
    except Exception as e:
        print(f"  [VLM 调用失败] {e}")
        return None


def vlm_review(args):
    """用 VLM 对照实例图鉴自动裁决 crops/ 里的待复核裁剪。

    流程：build_atlas 生成图鉴 -> 每批 --batch 张目标裁剪拼成带 T 编号的
    目标图 -> VLM 输出每格匹配的图鉴索引 -> 索引有效且 conf>=阈值时把文件
    移到对应类目录（改判），否则移入 crops_rejected/（不物理删除，留审计）。
    全部判定写入 vlm_review_log.jsonl。之后照常运行 apply-review 回填。
    """
    import shutil

    names, _ = load_template_library()
    root = Path(args.out)
    crop_dir = root / "crops"
    if not crop_dir.is_dir():
        raise SystemExit(f"未找到裁剪目录: {crop_dir}，先运行 export-review")

    atlas_b64, cell2id = load_atlas(args)
    if getattr(args, "classes", ""):
        # 地图白名单迷你图鉴：只含指定类（+player），干扰格越少越准
        import base64
        from pypinyin import lazy_pinyin
        only = {"_".join(lazy_pinyin(t.strip())) for t in args.classes.split(",")
                if t.strip()}
        mini = TEMPLATE_DIR / "atlas_mini.png"
        build_atlas(args.instances, args.player, mini, only=only)
        mapping = {int(k): v for k, v in json.loads(
            mini.with_suffix(".json").read_text(encoding="utf-8")).items()}
        names, _ = load_template_library()
        us2id = {"_".join(lazy_pinyin(nm)): cid for cid, nm in names.items()}
        cell2id = {no: us2id.get(nm) for no, nm in mapping.items()}
        atlas_b64 = base64.b64encode(
            cv2.imencode(".jpg", imread_u(mini))[1].tobytes()).decode()
        print(f"迷你图鉴: {len(cell2id)} 格 {sorted(mapping.values())}")

    # 收集所有待复核裁剪
    targets = [f for cls_dir in sorted(crop_dir.iterdir()) if cls_dir.is_dir()
               for f in sorted(cls_dir.glob("*.jpg"))]
    print(f"待复核裁剪 {len(targets)} 张，批大小 {args.batch}")

    rejected_dir = root / "crops_rejected"
    log_path = root / "vlm_review_log.jsonl"
    stats = defaultdict(int)
    with log_path.open("a", encoding="utf-8") as log:
        for i in range(0, len(targets), args.batch):
            batch = targets[i:i + args.batch]
            out = _codex_vlm([atlas_b64, _sheet_b64([imread_u(f) for f in batch])],
                             PROMPT % max(cell2id), model=args.model)
            m = re.search(r"\{.*\}", out or "", re.DOTALL)
            if not m:
                stats["no_parse"] += len(batch)
                continue
            try:
                verdicts = json.loads(m.group(0))
            except json.JSONDecodeError:
                stats["no_parse"] += len(batch)
                continue
            for j, f in enumerate(batch, 1):
                v = verdicts.get(f"T{j}", {})
                try:
                    cell = int(v.get("cell", -1))
                except (TypeError, ValueError):
                    cell = -1
                conf = float(v.get("conf", 0) or 0)
                cid = cell2id.get(cell)
                if cid is not None and conf >= args.conf:
                    dest = crop_dir / py(names[cid]) / f.name
                    moved = "keep" if dest.parent == f.parent else "move"
                    if dest.parent != f.parent:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(f), str(dest))
                    stats[moved] += 1
                else:
                    rejected_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(rejected_dir / f.name))
                    stats["reject"] += 1
                log.write(json.dumps(
                    {"file": f.name, "from": f.parent.name,
                     "cell": cell, "conf": conf,
                     "to": py(names[cid]) if cid is not None else "REJECT"},
                    ensure_ascii=False) + "\n")
            log.flush()
            print(f"  {min(i + args.batch, len(targets))}/{len(targets)} "
                  f"(keep={stats['keep']} move={stats['move']} "
                  f"reject={stats['reject']})")
    print("== 汇总 ==", dict(stats))


def _ncc_scores(crop, templates, scales=(1.0, 0.9, 0.8, 0.7, 0.6)):
    """裁剪图 vs 模板集的多尺度 NCC 打分。templates: {cls: [灰度图]}。

    与 classify_crop 同思路（测试图缩放到模板尺寸再匹配），但输入是已裁好的
    裁剪图（不带框），模板也不再预做中心裁剪。返回 [(score, cls), ...] 降序。
    """
    cg = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = {}
    scores = []
    for cls, tpls in templates.items():
        best = -1.0
        for g in tpls:
            gh, gw = g.shape
            for k in scales:
                tW, tH = max(int(gw * k), 8), max(int(gh * k), 8)
                if (tW, tH) not in resized:
                    c = cv2.resize(cg, (tW, tH), interpolation=cv2.INTER_AREA
                                   if k < 1 else cv2.INTER_CUBIC)
                    pad_h, pad_w = max(0, gh - tH) + 3, max(0, gw - tW) + 3
                    resized[(tW, tH)] = cv2.copyMakeBorder(
                        c, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
                c = resized[(tW, tH)]
                if c.shape[0] < gh or c.shape[1] < gw:
                    continue
                best = max(best, float(cv2.matchTemplate(c, g, cv2.TM_CCOEFF_NORMED).max()))
        if best > 0:
            scores.append((best, cls))
    return sorted(scores, reverse=True)


def refine_crops(args):
    """域内模板精校：用可信类目录的种子裁剪做模板，NCC 重分类全部 crops/。

    背景（2026-08-31）：VLM 图鉴匹配有系统性错法（红蜗牛→火野猪 conf0.86、
    绿水灵→青螃蟹），且同一卷录像里精灵像素级一致——同域 NCC 一致性远比
    VLM 可靠。做法：
    1. 种子：--allow 白名单类（拼音目录）各取至多 --seed 张裁剪做模板；
       白名单外目录（如 huoyezhu/qingpangxie/weizhi）目录名不可信，不做种子；
    2. 全部 crops（含白名单外目录与"未知"）对种子集打分；
    3. >= --thresh 移动到最高分类目录（文件名不变，apply-review 可解析）；
       否则移入 crops_rejected/。全程写 refine_log.jsonl。
    """
    import shutil

    names, _ = load_template_library()
    root = Path(args.out)
    crop_dir = root / "crops"
    if not crop_dir.is_dir():
        raise SystemExit(f"未找到裁剪目录: {crop_dir}，先运行 export-review")

    allow = None
    if getattr(args, "allow", ""):
        allow = set()
        for tok in args.allow.split(","):
            tok = tok.strip()
            if tok.isdigit():
                allow.add(py(names[int(tok)]))
            elif tok:
                allow.add(py(tok))
    print(f"种子白名单: {sorted(allow) if allow else '(全部目录)'}")

    # 1) 种子模板
    templates = defaultdict(list)
    for cls_dir in sorted(crop_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        if allow is not None and cls_dir.name not in allow:
            continue
        files = sorted(cls_dir.glob("*.jpg"))
        if not files:
            continue
        step = max(1, len(files) // args.seed)
        for f in files[::step][:args.seed]:
            img = imread_u(f)
            if img is not None:
                templates[cls_dir.name].append(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    print(f"种子: {len(templates)} 类 {sum(len(v) for v in templates.values())} 张")
    if not templates:
        raise SystemExit("没有可用种子目录")

    # 2)+3) 全量重分类
    rejected_dir = root / "crops_rejected"
    log_path = root / "refine_log.jsonl"
    stats = defaultdict(int)
    with log_path.open("a", encoding="utf-8") as log:
        for cls_dir in sorted(crop_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            for f in sorted(cls_dir.glob("*.jpg")):
                img = imread_u(f)
                if img is None:
                    continue
                scores = _ncc_scores(img, templates)
                top = scores[0] if scores else None
                dest = None
                if top and top[0] >= args.thresh:
                    dest = crop_dir / top[1] / f.name
                if dest is not None and dest.parent != cls_dir \
                        and dest.exists():
                    dest = None  # 同名说明该框已有同类裁决，按重复处理
                if dest is not None and dest.parent != cls_dir:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(dest))
                    stats["move"] += 1
                elif dest is not None:
                    stats["keep"] += 1
                else:
                    rejected_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(rejected_dir / f.name))
                    stats["reject"] += 1
                log.write(json.dumps(
                    {"file": f.name, "from": cls_dir.name,
                     "to": top[1] if top else "-",
                     "score": round(top[0], 4) if top else 0.0,
                     "top3": [(c, round(s, 3)) for s, c in scores[:3]]},
                    ensure_ascii=False) + "\n")
        # 汇报各目录余量
    print("== 各目录剩余 ==")
    for d in sorted(crop_dir.iterdir()):
        if d.is_dir():
            n = len(list(d.glob("*.jpg")))
            if n:
                print(f"  {d.name}: {n}")
    print(f"== 汇总 == keep={stats['keep']} move={stats['move']} "
          f"reject={stats['reject']} (日志 {log_path.name})")


def apply_review(args):
    """把人工或 vlm-review 整理后的 crops/ 裁决回填到 labels/。

    约定（export-review 生成、人工/VLM 编辑后）：
    - crops/<拼音类>/<帧>_yolo-<拼音>_<x1>_<y1>.jpg 存在 -> 采纳该框，
      类别以所在文件夹为准（移动到别的类目录 = 改判）；
    - review 里对应框的裁剪图不在 crops/（已被移入 crops_rejected/ 或删除）
      -> 拒绝（误检，不入库）。
    采纳的框合并进 labels/<帧>.txt（自动标签保留），review/*.json 归档到
    review_applied/。执行前自动备份 labels/ 到 labels_backup-<时间戳>/。
    """
    import shutil
    from datetime import datetime

    names, _ = load_template_library()
    py2id = {py(nm): cid for cid, nm in names.items()}
    root = Path(args.out)
    crop_dir = root / "crops"
    if not crop_dir.is_dir():
        raise SystemExit(f"未找到裁剪目录: {crop_dir}，先运行 export-review")

    # 收集人工裁决: (帧stem, x1, y1) -> 类别id
    verdict = {}
    for cls_dir in crop_dir.iterdir():
        if not cls_dir.is_dir() or cls_dir.name not in py2id:
            continue
        cid = py2id[cls_dir.name]
        for f in cls_dir.glob("*.jpg"):
            parts = f.stem.rsplit("_", 3)
            if len(parts) != 4 or not parts[0] or not parts[2].lstrip("-").isdigit():
                print(f"  [跳过] 文件名不符合约定: {cls_dir.name}/{f.name}")
                continue
            stem, _, x1, y1 = parts
            verdict[(stem, int(x1), int(y1))] = cid

    accepted = rejected = 0
    backup = root / f"labels_backup-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    applied_dir = root / "review_applied"
    for rj in sorted((root / "review").glob("*.json")):
        stem = rj.stem
        entries = json.loads(rj.read_text(encoding="utf-8"))
        lbl = root / "labels" / f"{stem}.txt"
        if not backup.exists() and lbl.exists():
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(lbl, backup / lbl.name)
        kept = [e for e in entries if (stem, e["box"][0], e["box"][1]) in verdict]
        accepted += len(kept)
        rejected += len(entries) - len(kept)
        # 采纳框追加到标签
        if kept:
            img_path = next((root / "images").glob(f"{stem}.*"), None)
            if img_path is None:
                print(f"  [警告] 找不到帧图片，跳过 {stem}")
                continue
            img = imread_u(img_path)
            H, W = img.shape[:2]
            lines = lbl.read_text(encoding="utf-8").splitlines() if lbl.exists() else []
            for e in kept:
                cid = verdict[(stem, e["box"][0], e["box"][1])]
                x1, y1, x2, y2 = e["box"]
                cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                bw, bh = (x2 - x1) / W, (y2 - y1) / H
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            lbl.write_text("\n".join(lines), encoding="utf-8")
        # 归档 review json（无论有无采纳，均已处理完）
        applied_dir.mkdir(parents=True, exist_ok=True)
        rj.rename(applied_dir / rj.name)

    print(f"采纳 {accepted} 框（已并入 labels/），拒绝 {rejected} 框")
    print(f"labels 备份 -> {backup}，review 归档 -> {applied_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-templates", help="从已标注数据构建模板库")
    b.add_argument("--datasets", nargs="+", default=[
        "datasets/mxdzlk_cmsc,1",
        "data/annotated_73/images/train,data/annotated_73/labels/train,0",
    ], help="每项为 数据集根,offset 或 images_dir,labels_dir,offset；"
            "offset 把该数据集的类别 id 平移到模型 id 空间")
    b.add_argument("--out", default=str(TEMPLATE_DIR))
    b.add_argument("--max-per-class", type=int, default=8)

    r = sub.add_parser("run", help="自动标注图片目录或视频")
    r.add_argument("--source", required=True)
    r.add_argument("--model", default="weights/20260823/best.pt")
    r.add_argument("--out", default="auto_labeled")
    r.add_argument("--conf", type=float, default=0.05, help="YOLO 候选置信度下限")
    r.add_argument("--high-conf", type=float, default=0.7,
                   help="YOLO 直接采信的置信度下限")
    r.add_argument("--vlm-conf", type=float, default=0.25,
                   help="低于此置信度且模板不认时才值得送 VLM")
    r.add_argument("--agree-conf", type=float, default=0.35,
                   help="双源一致时的 YOLO 置信度下限")
    r.add_argument("--write-vlm", action="store_true",
                   help="把 VLM 仲裁结果也写入标签（默认仅落 review/ 待人工确认）")
    r.add_argument("--map-classes", default="",
                   help="当前地图怪物白名单（逗号分隔中文名或 id，来自 bot-cs "
                        "data/maps/names.json 的出没表）；不在清单内的框"
                        "（含 YOLO 高置信）降级送 VLM 复核，防 UI/杂框误入库")
    r.add_argument("--vlm-recheck-classes", default="",
                   help="VLM 必审类（逗号分隔中文名或 id）：这些类即使 YOLO 高置信"
                        "也不直接采信，降级送 VLM 迷你图鉴复核。用于换色家族"
                        "（蓝/红蜗牛、蘑菇仔/花/绿蘑菇）与绿色背景易误类（绿水灵）——"
                        "实测 YOLO 高置信直采的绿水灵框 97%% 实为静态花草")
    r.add_argument("--static-thresh", type=float, default=0.0,
                   help="静态背景剔除阈值（帧间运动量，0~255，默认 0=关闭）：框内"
                        "|当前帧-背景中值| 灰度均值低于此值的框视为静态花草/UI/装饰，"
                        "不直接采信而送 VLM 复核。仅视频源生效。推荐从 3~5 起步")
    r.add_argument("--bg-window", type=int, default=8,
                   help="滚动背景中值窗口帧数（静态剔除用），默认 8")
    r.add_argument("--instances", default="datasets/mxdzlk_cmsc/monster_instances",
                   help="VLM 图鉴实例图根目录（拼音子目录）")
    r.add_argument("--player", default="datasets/player/magic.png",
                   help="玩家示例图（作为图鉴最后一格）")
    r.add_argument("--imgsz", type=int, default=1344)
    r.add_argument("--fps", type=float, default=5, help="视频抽帧帧率")
    r.add_argument("--no-vlm", action="store_true")

    e = sub.add_parser("export-review", help="review/*.json 转可视化图+按类裁剪图")
    e.add_argument("--out", required=True, help="run 的输出目录（含 review/ images/ labels/）")

    rf = sub.add_parser("refine-crops",
                        help="域内模板精校：种子 NCC 重分类全部 crops/")
    rf.add_argument("--out", required=True, help="run 的输出目录（含 crops/）")
    rf.add_argument("--allow", default="",
                    help="种子白名单（逗号分隔类名或 id，目录名用拼音）；"
                         "空=全部目录做种子（仅当目录名全部可信时）")
    rf.add_argument("--seed", type=int, default=3, help="每类种子数")
    rf.add_argument("--thresh", type=float, default=0.9,
                    help="NCC 采纳阈值，低于此移入 crops_rejected/")

    ar = sub.add_parser("apply-review", help="人工整理后的 crops/ 裁决回填 labels/")
    ar.add_argument("--out", required=True, help="run 的输出目录（含 crops/ review/ labels/）")

    v = sub.add_parser("vlm-review", help="VLM 对照实例图鉴自动裁决 crops/")
    v.add_argument("--out", required=True, help="run 的输出目录（含 crops/）")
    v.add_argument("--instances", default="datasets/mxdzlk_cmsc/monster_instances",
                   help="每类实例图根目录（拼音子目录）")
    v.add_argument("--player", default="datasets/player/magic.png",
                   help="玩家示例图（作为图鉴最后一格）")
    v.add_argument("--batch", type=int, default=6, help="每次 VLM 调用打包的目标数")
    v.add_argument("--classes", default="",
                   help="迷你图鉴白名单（逗号分隔类名）；给定则图鉴只含这些类"
                        "+player，VLM 匹配更准（实测全图鉴红蜗牛→火野猪，"
                        "迷你图鉴 85%%）；不传=全 73 类图鉴")
    v.add_argument("--conf", type=float, default=0.6, help="低于此置信度视为拒绝")
    v.add_argument("--model", default="gpt-5.6-luna")

    a = ap.parse_args()
    if a.cmd == "build-templates":
        ds = []
        for item in a.datasets:
            parts = item.split(",")
            if len(parts) == 2:  # 数据集根,offset
                root = Path(parts[0])
                ds.append((root / "images", root / "labels",
                           root / "dataset.yaml", int(parts[1])))
            else:  # images_dir,labels_dir,offset
                im, lb, off = parts
                ds.append((im, lb, "data/annotated_73/dataset.yaml", int(off)))
        build_template_library(ds, a.out, a.max_per_class)
    elif a.cmd == "export-review":
        export_review(a)
    elif a.cmd == "refine-crops":
        refine_crops(a)
    elif a.cmd == "apply-review":
        apply_review(a)
    elif a.cmd == "vlm-review":
        vlm_review(a)
    else:
        run_pipeline(a)


if __name__ == "__main__":
    main()
