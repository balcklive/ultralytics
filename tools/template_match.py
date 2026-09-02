"""基于 OpenCV 模板匹配的怪物检测/标注工具。

从已有 YOLO 标注数据中提取各类怪物的精灵模板（去重），在目标图片上做
多模板、多尺度归一化互相关匹配，输出候选框。用途：
1. 自动标注管线的第一检测来源（与 YOLO 模型互为印证）；
2. 评测模板匹配相对人工标注的 IoU / 召回 / 精确率（eval 模式）。

用法:
  uv run python tools/template_match.py eval \
      --data datasets/mxdzlk_cmsc/dataset.yaml --tpl-ratio 0.3 \
      --thresh 0.75 --max-test 40 --out artifacts/template_eval.json
  uv run python tools/template_match.py match --data ... --source <图片或目录> --out <标签输出目录>
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_classes(dataset_yaml):
    """读取 dataset.yaml，返回 {class_id: name}。"""
    with open(dataset_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {int(k): v for k, v in data["names"].items()}


def load_labels(label_path, img_w, img_h):
    """读取 YOLO txt（容忍 CRLF），返回 [(cls, x1, y1, x2, y2)] 像素坐标。"""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = (float(x) for x in parts[1:5])
        boxes.append(
            (
                cls,
                (cx - w / 2) * img_w,
                (cy - h / 2) * img_h,
                (cx + w / 2) * img_w,
                (cy + h / 2) * img_h,
            )
        )
    return boxes


def crop_template(img, box, pad=0.0):
    """按 GT 框裁剪模板，可选外扩比例（吸纳动画帧/描边差异）。"""
    _, x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx, dy = w * pad / 2, h * pad / 2
    H, W = img.shape[:2]
    return img[
        max(0, int(y1 - dy)) : min(H, int(y2 + dy)),
        max(0, int(x1 - dx)) : min(W, int(x2 + dx)),
    ].copy()


def build_templates(img_paths, labels_dir, classes, max_per_class=8, pad=0.0):
    """从已标注图片裁剪每类模板并按像素内容去重。

    img_paths 可传目录或路径列表。返回 {cls: [template(BGR), ...]}，
    模板尺寸保留原始像素（用于同分辨率匹配）。
    """
    if isinstance(img_paths, (str, Path)):
        d = Path(img_paths)
        img_paths = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
    by_cls = defaultdict(list)
    seen = defaultdict(set)
    for img_path in img_paths:
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        for box in load_labels(label_path, W, H):
            cls = box[0]
            if cls not in classes or len(by_cls[cls]) >= max_per_class:
                continue
            tpl = crop_template(img, box, pad)
            key = tpl.tobytes()
            if key in seen[cls]:
                continue
            seen[cls].add(key)
            by_cls[cls].append(tpl)
    return dict(by_cls)


def nms(boxes, scores, iou_thr=0.4):
    """按分数的贪心 NMS，boxes 为 [x1,y1,x2,y2]。"""
    idxs = np.argsort(scores)[::-1]
    keep = []
    while idxs.size:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        a1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a2 = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (a1 + a2 - inter + 1e-9)
        idxs = rest[iou <= iou_thr]
    return keep


def match_image(img, templates, thresh=0.75, scales=(1.0,), pad=0.0):
    """对单图做多模板多尺度匹配，返回 [(cls, x1, y1, x2, y2, score), ...]。

    灰度归一化互相关；同模板多个命中像素经 NMS 去重；跨类重叠由调用方
    按分数再做一次 NMS。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    cand_boxes, cand_scores, cand_cls = [], [], []
    for cls, tpls in templates.items():
        best = None  # 每类取所有模板/尺度中的最佳响应图
        for tpl in tpls:
            g = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
            for s in scales:
                t = (
                    cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                    if s != 1.0
                    else g
                )
                th, tw = t.shape
                if th < 8 or tw < 8 or th > H or tw > W:
                    continue
                res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
                if best is None or res.max() > best[0].max():
                    best = (res, tw, th)
        if best is None:
            continue
        res, tw, th = best
        ys, xs = np.where(res >= thresh)
        for y, x in zip(ys, xs):
            gx, gy = x + tw / 2, y + th / 2
            gw, gh = tw / (1 + pad), th / (1 + pad)
            cand_boxes.append(
                [gx - gw / 2, gy - gh / 2, gx + gw / 2, gy + gh / 2]
            )
            cand_scores.append(float(res[y, x]))
            cand_cls.append(cls)
    if not cand_boxes:
        return []
    boxes = np.array(cand_boxes)
    scores = np.array(cand_scores)
    cls_arr = np.array(cand_cls)
    keep = nms(boxes, scores, iou_thr=0.4)
    return [
        (int(cls_arr[i]), *boxes[i].tolist(), float(scores[i])) for i in keep
    ]


def iou_matrix(a, b):
    """a: [N,4], b: [M,4] -> IoU 矩阵 [N,M]。"""
    N, M = len(a), len(b)
    out = np.zeros((N, M))
    for i in range(N):
        xx1 = np.maximum(a[i, 0], b[:, 0])
        yy1 = np.maximum(a[i, 1], b[:, 1])
        xx2 = np.minimum(a[i, 2], b[:, 2])
        yy2 = np.minimum(a[i, 3], b[:, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_a = (a[i, 2] - a[i, 0]) * (a[i, 3] - a[i, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        out[i] = inter / (area_a + area_b - inter + 1e-9)
    return out


def evaluate(det, gt, iou_thr=0.5):
    """单图评测：返回 (tp, fp, fn, matched_gt_idx)。匹配取 IoU 最大且类别一致。"""
    if not det:
        return 0, 0, len(gt), set()
    d = np.array([x[1:5] for x in det])
    g = np.array([x[1:5] for x in gt]) if gt else np.zeros((0, 4))
    ious = iou_matrix(d, g)
    matched_gt, tp, fp = set(), 0, 0
    order = np.argsort([x[5] for x in det])[::-1]
    for di in order:
        best_g, best_i = -1, iou_thr
        for gi in range(len(gt)):
            if gi in matched_gt:
                continue
            if ious[di, gi] >= best_i and det[di][0] == gt[gi][0]:
                best_i, best_g = ious[di, gi], gi
        if best_g >= 0:
            tp += 1
            matched_gt.add(best_g)
        else:
            fp += 1
    return tp, fp, len(gt) - len(matched_gt), matched_gt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["eval", "match"])
    ap.add_argument("--data", required=True, help="dataset.yaml 路径")
    ap.add_argument("--source", help="match 模式: 图片或目录")
    ap.add_argument("--out", default="artifacts/template_eval.json")
    ap.add_argument("--tpl-ratio", type=float, default=0.3,
                    help="eval 模式: 用作模板池的图片比例（其余作测试集）")
    ap.add_argument("--max-per-class", type=int, default=8)
    ap.add_argument("--max-test", type=int, default=40)
    ap.add_argument("--thresh", type=float, default=0.75)
    ap.add_argument("--pad", type=float, default=0.1,
                    help="模板外扩比例，吸纳动画帧差异")
    ap.add_argument("--scales", default="1.0", help="逗号分隔的缩放列表")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.data).parent
    classes = load_classes(args.data)
    images_dir = root / "images"
    labels_dir = root / "labels"
    img_paths = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    random.Random(args.seed).shuffle(img_paths)

    n_tpl = max(1, int(len(img_paths) * args.tpl_ratio))
    scales = tuple(float(s) for s in args.scales.split(","))

    print(f"[1/3] 从 {n_tpl} 张图构建模板 (max {args.max_per_class}/类)...")
    templates = build_templates(
        [p for p in img_paths[:n_tpl]], labels_dir, classes,
        args.max_per_class, args.pad,
    )
    print(f"      覆盖 {len(templates)}/{len(classes)} 类，"
          f"共 {sum(len(v) for v in templates.values())} 个模板")

    test_paths = img_paths[n_tpl : n_tpl + args.max_test]

    if args.mode == "match":
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in test_paths or img_paths[: args.max_test]:
            img = cv2.imread(str(p))
            dets = match_image(img, templates, args.thresh, scales, args.pad)
            H, W = img.shape[:2]
            lines = [
                f"{c} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}"
                for c, x1, y1, x2, y2, _ in dets
            ]
            (out_dir / (p.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
            print(f"  {p.name}: {len(dets)} boxes -> {out_dir/(p.stem+'.txt')}")
        return

    print(f"[2/3] 在 {len(test_paths)} 张测试图上匹配 (thresh={args.thresh})...")
    tp = fp = fn = 0
    ious_all, det_scores, per_img = [], [], []
    vis_dir = Path("artifacts/template_eval_vis")
    vis_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(test_paths):
        img = cv2.imread(str(p))
        H, W = img.shape[:2]
        gt = load_labels(labels_dir / (p.stem + ".txt"), W, H)
        dets = match_image(img, templates, args.thresh, scales, args.pad)
        t, f, n, matched = evaluate(dets, gt)
        tp += t; fp += f; fn += n
        for gi in matched:
            best = max(
                iou_matrix(np.array([d[1:5] for d in dets]),
                           np.array([gt[gi][1:5]]))[j][0]
                for j in range(len(dets))
            )
            ious_all.append(best)
        det_scores += [d[5] for d in dets]
        per_img.append((p.name, len(gt), len(dets), t, f, n))
        if i < 6:  # 可视化前几张
            vis = img.copy()
            for c, x1, y1, x2, y2, s in dets:
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                cv2.putText(vis, f"{c} {s:.2f}", (int(x1), int(y1) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(str(vis_dir / f"{p.stem}.png"), vis)

    print(f"[3/3] 结果: TP={tp} FP={fp} FN={fn}")
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    report = {
        "thresh": args.thresh, "pad": args.pad, "scales": scales,
        "n_templates": {classes[c]: len(v) for c, v in sorted(templates.items())},
        "n_test_images": len(test_paths), "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4), "recall": round(rec, 4),
        "mean_iou_matched": round(float(np.mean(ious_all)), 4) if ious_all else None,
        "score_min": min(det_scores) if det_scores else None,
        "per_image": per_img,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"precision={prec:.3f} recall={rec:.3f} "
          f"meanIoU={report['mean_iou_matched']} -> {out}")


if __name__ == "__main__":
    main()
