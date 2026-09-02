"""整帧人工查漏复核器：逐帧检查标注，补画漏检怪物框、改类、删框，实时回写 labels。

用法:
  uv run python tools/frame_review.py --root data/wgc_review
（root 由 tools/review_pack.py 生成：images/train/ labels/train/ dataset.yaml）

操作:
  左键拖拽空白处   画新框（类别=右侧面板当前选中类，画完立即写盘）
  左键点已有框     弹出菜单：改判类别 / 删除
  滚轮             以光标为中心缩放（0.5x~8x）
  中键拖拽 / 方向键 平移
  m                开/关 运动热图（|当前帧-背景|，游戏背景静态，
                   动过的区域=怪物活动区，用于定位漏检）
  n / 空格         下一帧（自动落盘）；p 上一帧
  [ ]              切换当前画框类别
  u                撤销上一次操作
  q                退出
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

PANEL_W = 300
WIN_W, WIN_H = 1256, 840
FONT = cv2.FONT_HERSHEY_SIMPLEX


def imread_u(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def py(name):
    from pypinyin import lazy_pinyin
    return "".join(lazy_pinyin(name)) or name


class FrameReview:
    def __init__(self, root: Path, classes: str = ""):
        self.root = root
        self.im_dir = root / "images" / "train"
        self.lbl_dir = root / "labels" / "train"
        cfg = yaml.safe_load((root / "dataset.yaml").read_text(encoding="utf-8"))
        self.names = {int(k): v for k, v in cfg["names"].items()}
        self.id2py = {cid: py(nm) for cid, nm in self.names.items()}
        self.py2id = {v: k for k, v in self.id2py.items()}
        # 面板类列表：默认全部；传 --classes（逗号分隔中文名或 id）则只显示指定类 + player，
        # 73 类里本地图外的怪物不显示，避免逐类翻找（如火焰之地Ⅴ只需 猴子/火野猪/黑斧木妖）。
        self.panel_ids = sorted(self.names)
        if classes:
            keep = {0}
            for tok in classes.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                if tok.isdigit():
                    keep.add(int(tok))
                else:
                    keep.update(cid for cid, nm in self.names.items() if nm == tok)
            self.panel_ids = sorted(keep & set(self.names))
        self.cur_cls = self.panel_ids[0] if self.panel_ids else 0
        self.frames = sorted(self.im_dir.glob("*.jpg"))
        self.fi = 0
        self.img = None
        self.H = self.W = 0
        self.lines = []          # [(cls, cx, cy, w, h)] 归一化
        self.dirty = False
        # 视图状态
        self.scale = 1.0
        self.ox = self.oy = 0.0
        self.fit_scale = 1.0
        self._pan = None
        self._draw = None        # 画框进行中 (x1,y1) 屏幕坐标
        self.menu = None
        self.sel = None          # label 行号
        self.show_motion = False
        self.undo_stack = []
        self.msg = "drag=new box  wheel=zoom  middle-drag/arrows=pan  m=motion  n/p=frame  q=quit"
        self.window = "frame review | drag=new box  wheel=zoom  middle-drag/arrows=pan  m=motion  n/p  q"
        # 右侧类别面板滚动偏移（73 类超出可视高度，滚轮在面板内滚动列表）
        self.panel_scroll = 0
        # 背景中值（1/4 分辨率，灰度）——运动热图用
        self.bg = None
        self._build_bg()

    # ---------- 数据 ----------

    def _build_bg(self):
        smalls = []
        for f in self.frames[::2]:
            img = imread_u(f)
            if img is None:
                continue
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            smalls.append(cv2.resize(g, (g.shape[1] // 4, g.shape[0] // 4)))
        if smalls:
            self.bg = np.median(np.stack(smalls), axis=0).astype(np.uint8)

    def load(self):
        f = self.frames[self.fi]
        self.img = imread_u(f)
        self.H, self.W = self.img.shape[:2]
        lf = self.lbl_dir / (f.stem + ".txt")
        self.lines = []
        if lf.exists():
            for l in lf.read_text(encoding="utf-8").splitlines():
                p = l.split()
                if len(p) >= 5:
                    self.lines.append([int(p[0]), *map(float, p[1:5])])
        if not hasattr(self, "_fit_done") or self._fit_for != (self.W, self.H):
            self.fit_scale = min((WIN_W - PANEL_W) / self.W, WIN_H / self.H)
            self.scale = self.fit_scale
            self.ox = self.oy = 0.0
            self._fit_for = (self.W, self.H)
        self.menu = None
        self.sel = None
        self.undo_stack = []
        self.dirty = False

    def save(self):
        f = self.frames[self.fi]
        out = "\n".join(
            f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in self.lines)
        (self.lbl_dir / (f.stem + ".txt")).write_text(out + ("\n" if out else ""),
                                                      encoding="utf-8")
        self.dirty = False

    def line_box(self, ln):
        _, cx, cy, w, h = ln
        return (int((cx - w / 2) * self.W), int((cy - h / 2) * self.H),
                int((cx + w / 2) * self.W), int((cy + h / 2) * self.H))

    # ---------- 坐标变换 ----------

    def to_screen(self, x, y):
        return ((x - self.ox) * self.scale, (y - self.oy) * self.scale)

    def to_img(self, sx, sy):
        return (sx / self.scale + self.ox, sy / self.scale + self.oy)

    # ---------- 操作 ----------

    def add_box(self, x1, y1, x2, y2, cls):
        x1, x2 = sorted((max(0, min(self.W, x1)), max(0, min(self.W, x2))))
        y1, y2 = sorted((max(0, min(self.H, y1)), max(0, min(self.H, y2))))
        if x2 - x1 < 6 or y2 - y1 < 6:
            return
        ln = [cls, (x1 + x2) / 2 / self.W, (y1 + y2) / 2 / self.H,
              (x2 - x1) / self.W, (y2 - y1) / self.H]
        self.lines.append(ln)
        self.undo_stack.append(("add", len(self.lines) - 1))
        self.dirty = True
        self.save()
        self.msg = f"added {self.id2py.get(cls)} box ({x2 - x1}x{y2 - y1})"

    def set_cls(self, idx, cls):
        old = self.lines[idx][0]
        self.lines[idx][0] = cls
        self.undo_stack.append(("cls", idx, old))
        self.dirty = True
        self.save()
        self.msg = f"line {idx + 1}: {self.id2py.get(old)} -> {self.id2py.get(cls)}"

    def del_box(self, idx):
        removed = self.lines.pop(idx)
        self.undo_stack.append(("del", idx, removed))
        self.dirty = True
        self.save()
        self.msg = f"deleted line {idx + 1}"

    def undo(self):
        if not self.undo_stack:
            self.msg = "nothing to undo"
            return
        op = self.undo_stack.pop()
        if op[0] == "add":
            self.lines.pop(op[1])
        elif op[0] == "cls":
            self.lines[op[1]][0] = op[2]
        else:
            self.lines.insert(min(op[1], len(self.lines)), op[2])
        self.dirty = True
        self.save()
        self.msg = "undone"

    # ---------- 绘制 ----------

    def color_of(self, cls):
        if cls == 3:
            return (255, 160, 0)
        if cls == 6:
            return (0, 0, 255)
        if cls == 8:
            return (80, 220, 80)
        if cls == 4:
            return (0, 140, 255)
        if cls == 0:
            return (255, 255, 255)
        if cls == 5:
            return (0, 80, 120)
        if cls in (7,):
            return (0, 255, 255)
        if cls == 15:
            return (60, 180, 60)
        return (200, 0, 200)

    def draw(self):
        canvas = np.full((WIN_H, WIN_W, 3), (245, 245, 245), np.uint8)
        vw = int(self.W * self.scale)
        vh = int(self.H * self.scale)
        # 视野裁剪（视口=窗口去掉右面板）
        w_v, h_v = WIN_W - PANEL_W, WIN_H
        ox = max(0.0, min(self.ox, max(0.0, self.W - w_v / self.scale)))
        oy = max(0.0, min(self.oy, max(0.0, self.H - h_v / self.scale)))
        self.ox, self.oy = ox, oy
        x0, y0 = int(ox), int(oy)
        x1 = min(self.W, int(ox + w_v / self.scale) + 1)
        y1 = min(self.H, int(oy + h_v / self.scale) + 1)
        if x1 > x0 and y1 > y0:
            region = self.img[y0:y1, x0:x1]
            rw, rh = int((x1 - x0) * self.scale), int((y1 - y0) * self.scale)
            rw, rh = min(rw, w_v), min(rh, h_v)  # 防缩放取整越界
            region = cv2.resize(region, (rw, rh),
                                interpolation=cv2.INTER_NEAREST if self.scale > 1.5
                                else cv2.INTER_AREA)
            canvas[0:rh, 0:rw] = region
        # 运动热图叠加
        if self.show_motion and self.bg is not None:
            g = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(g, cv2.resize(self.bg, (g.shape[1], g.shape[0])))
            mask = (diff > 25).astype(np.uint8) * 255
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
            mask = cv2.resize(mask, (rw, rh))
            overlay = canvas[0:rh, 0:rw].copy()
            overlay[mask > 0] = (0, 0, 255)
            cv2.addWeighted(overlay, 0.45, canvas[0:rh, 0:rw], 0.55, 0, canvas[0:rh, 0:rw])
        # 标注框
        for i, ln in enumerate(self.lines):
            bx = self.line_box(ln)
            sx1, sy1 = self.to_screen(bx[0], bx[1])
            sx2, sy2 = self.to_screen(bx[2], bx[3])
            if sx2 < 0 or sy2 < 0 or sx1 > vw or sy1 > vh:
                continue
            col = self.color_of(ln[0])
            th = 2 if self.scale < 2 else 3
            cv2.rectangle(canvas, (int(sx1), int(sy1)), (int(sx2), int(sy2)),
                          col, th)
            cv2.putText(canvas, f"{ln[0]}:{self.id2py.get(ln[0], '?')}",
                        (int(sx1), max(14, int(sy1) - 4)), FONT,
                        max(0.35, min(0.7, 0.5 * self.scale / self.fit_scale)), col, 1)
            if i == self.sel:
                cv2.rectangle(canvas, (int(sx1) - 2, int(sy1) - 2),
                              (int(sx2) + 2, int(sy2) + 2), (0, 200, 255), 1)
        # 正在画的框
        if self._draw:
            cv2.rectangle(canvas, self._draw[0], self._draw[1], (180, 0, 255), 2)
        # 右侧面板
        px0 = WIN_W - PANEL_W
        cv2.rectangle(canvas, (px0, 0), (WIN_W, WIN_H), (235, 235, 235), -1)
        cv2.putText(canvas, f"frame {self.fi + 1}/{len(self.frames)}", (px0 + 10, 26),
                    FONT, 0.6, (0, 0, 0), 2)
        cv2.putText(canvas, self.frames[self.fi].stem[-14:], (px0 + 10, 48),
                    FONT, 0.45, (90, 90, 90), 1)
        cv2.putText(canvas, f"zoom {self.scale:.2f}x  boxes {len(self.lines)}",
                    (px0 + 10, 70), FONT, 0.5, (90, 90, 90), 1)
        cv2.putText(canvas, f"draw class: {self.id2py.get(self.cur_cls)}",
                    (px0 + 10, 96), FONT, 0.55, (0, 60, 0), 2)
        y0 = 116
        ROW_H = 26
        visible = (WIN_H - 60 - y0) // ROW_H
        # 限制滚动范围：最后一类对齐到可视区底部
        cids = self.panel_ids
        self.panel_scroll = max(0, min(self.panel_scroll, len(cids) - visible))
        for i in range(visible):
            idx = i + self.panel_scroll
            if idx >= len(cids):
                break
            cid = cids[idx]
            yy = y0 + i * ROW_H
            cur = cid == self.cur_cls
            cv2.rectangle(canvas, (px0 + 6, yy - 15), (WIN_W - 10, yy + 5),
                          (180, 255, 180) if cur else (240, 240, 240), -1)
            cv2.putText(canvas, f"{cid:2d} {self.id2py[cid]}", (px0 + 14, yy),
                        FONT, 0.5, (0, 0, 0) if cur else (60, 60, 60), 1)
        # 滚动提示（还有更多类）
        if self.panel_scroll > 0:
            cv2.putText(canvas, "^ more", (px0 + 10, y0 - 4), FONT, 0.45, (0, 0, 200), 1)
        if self.panel_scroll < len(cids) - visible:
            cv2.putText(canvas, "v more", (px0 + 10, WIN_H - 62), FONT, 0.45, (0, 0, 200), 1)
        hy = WIN_H - 46
        cv2.putText(canvas, "m=motion " + ("ON" if self.show_motion else "off"),
                    (px0 + 10, hy), FONT, 0.5, (0, 0, 200) if self.show_motion else (90, 90, 90), 1)
        cv2.putText(canvas, "u=undo  n/p=frame  q=quit", (px0 + 10, hy + 22),
                    FONT, 0.45, (90, 90, 90), 1)
        cv2.putText(canvas, self.msg[:78], (6, WIN_H - 8), FONT, 0.5, (0, 0, 200), 1)
        # 菜单
        if self.menu:
            m = self.menu
            cv2.rectangle(canvas, (m["x"], m["y"]), (m["x"] + m["w"], m["y"] + m["h"]),
                          (50, 50, 50), -1)
            cv2.rectangle(canvas, (m["x"], m["y"]), (m["x"] + m["w"], m["y"] + m["h"]),
                          (255, 255, 255), 1)
            for j, (label, _) in enumerate(m["items"]):
                yy = m["y"] + 6 + j * 26
                if m.get("hover") == j:
                    cv2.rectangle(canvas, (m["x"] + 3, yy - 4),
                                  (m["x"] + m["w"] - 3, yy + 18), (90, 90, 90), -1)
                col = (0, 200, 120) if label.startswith("X ") else (255, 255, 255)
                cv2.putText(canvas, label, (m["x"] + 10, yy + 10), FONT, 0.5, col, 1)
        return canvas

    # ---------- 交互 ----------

    def _hit(self, sx, sy):
        ix, iy = self.to_img(sx, sy)
        tol = 6 / self.scale
        for i in range(len(self.lines) - 1, -1, -1):
            x1, y1, x2, y2 = self.line_box(self.lines[i])
            near_x = x1 - tol <= ix <= x2 + tol
            near_y = y1 - tol <= iy <= y2 + tol
            if near_x and near_y:
                return i
        return None

    def _open_menu(self, sx, sy, idx):
        items = []
        for cid in self.panel_ids:
            items.append((f"{cid:2d} {self.id2py[cid]}"
                          + ("  <-now" if cid == self.lines[idx][0] else ""),
                          ("to", cid)))
        items.append(("X  delete box", ("del",)))
        items.append(("cancel", ("cancel",)))
        w, ih = 230, 26
        h = 12 + len(items) * ih
        self.menu = {"x": int(min(sx + 6, WIN_W - PANEL_W - w - 4)),
                     "y": int(min(max(4, sy - 10), WIN_H - h - 4)),
                     "w": w, "h": h, "items": items, "hover": None, "idx": idx}

    def _menu_hit(self, x, y):
        m = self.menu
        if not (m["x"] <= x <= m["x"] + m["w"] and m["y"] <= y <= m["y"] + m["h"]):
            return None
        j = (y - m["y"] - 6) // 26
        return j if 0 <= j < len(m["items"]) else None

    def on_mouse(self, event, x, y, flags, *args):
        in_panel = x >= WIN_W - PANEL_W
        if event == cv2.EVENT_MOUSEWHEEL:
            try:
                delta = cv2.getMouseWheelDelta(flags)
            except AttributeError:
                delta = 1 if flags > 0 else -1
            if in_panel:
                # 面板内滚轮 = 类别列表滚动（73 类超出可视高度）
                self.panel_scroll += (-1 if delta > 0 else 1) * 3
                return
            ix, iy = self.to_img(x, y)
            factor = 1.25 if delta > 0 else 1 / 1.25
            ns = max(self.fit_scale * 0.9, min(8.0, self.scale * factor))
            self.ox = ix - x / ns
            self.oy = iy - y / ns
            self.scale = ns
            return
        if event == cv2.EVENT_MOUSEMOVE:
            if self.menu:
                self.menu["hover"] = self._menu_hit(x, y)
            elif self._pan:
                (ox0, oy0), (mx0, my0) = self._pan
                self.ox = ox0 - (x - mx0) / self.scale
                self.oy = oy0 - (y - my0) / self.scale
            elif self._draw:
                self._draw[1] = (x, y)
            return
        if event in (cv2.EVENT_MBUTTONDOWN,) and not in_panel:
            self._pan = [(self.ox, self.oy), (x, y)]
            return
        if event == cv2.EVENT_MBUTTONUP:
            self._pan = None
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.menu:
                j = self._menu_hit(x, y)
                if j is None:
                    self.menu = None
                    return
                _, action = self.menu["items"][j]
                idx = self.menu["idx"]
                self.menu = None
                if action[0] == "to":
                    self.set_cls(idx, action[1])
                elif action[0] == "del":
                    self.del_box(idx)
                return
            if in_panel:
                # 类别选择（含滚动偏移，仅面板类）
                y0, step = 116, 26
                ids = self.panel_ids
                k = (y - y0 + 15) // step + self.panel_scroll
                if 0 <= k < len(ids) and y >= y0 - 15:
                    self.cur_cls = ids[k]
                return
            hit = self._hit(x, y)
            if hit is None:
                self._draw = [(x, y), (x, y)]
            else:
                self.sel = hit
                self._open_menu(x, y, hit)
            return
        if event == cv2.EVENT_LBUTTONUP:
            if self._draw:
                (ax, ay), (bx, by) = self._draw
                self._draw = None
                ix1, iy1 = self.to_img(ax, ay)
                ix2, iy2 = self.to_img(bx, by)
                self.add_box(ix1, iy1, ix2, iy2, self.cur_cls)
            return

    def run(self):
        self.load()
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.on_mouse)
        while True:
            cv2.imshow(self.window, self.draw())
            raw = cv2.waitKey(30)
            if raw < 0:
                continue
            key = raw & 0xFF
            ext = (raw >> 16) & 0x1FF
            if key == ord("q"):
                self.save()
                break
            if key == 27:
                if self.menu:
                    self.menu = None
                else:
                    self.save()
                    break
            elif key in (ord("n"), ord(" "), 13):
                self.save()
                if self.fi < len(self.frames) - 1:
                    self.fi += 1
                    self.load()
            elif key == ord("p"):
                self.save()
                if self.fi > 0:
                    self.fi -= 1
                    self.load()
            elif key == ord("m"):
                self.show_motion = not self.show_motion
            elif key == ord("u"):
                self.undo()
            elif key == ord("["):
                ids = self.panel_ids
                self.cur_cls = ids[(ids.index(self.cur_cls) - 1) % len(ids)]
            elif key == ord("]"):
                ids = self.panel_ids
                self.cur_cls = ids[(ids.index(self.cur_cls) + 1) % len(ids)]
            elif key == ord("s"):
                self.save()
                self.msg = "saved"
            # 方向键平移
            pan = 80 / self.scale
            if ext == 0x25:  # Left
                self.ox -= pan
            elif ext == 0x27:  # Right
                self.ox += pan
            elif ext == 0x26:  # Up
                self.oy -= pan
            elif ext == 0x28:  # Down
                self.oy += pan
            if self.dirty:
                self.save()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="review_pack.py 生成的复核数据集目录")
    ap.add_argument("--classes", default="",
                    help="右侧面板只显示指定类（逗号分隔中文名或 id，另含 player）；"
                         "不传则显示全部 73 类。如火焰之地Ⅴ：--classes \"猴子,火野猪,黑斧木妖\"")
    a = ap.parse_args()
    FrameReview(Path(a.root), a.classes).run()


if __name__ == "__main__":
    main()
