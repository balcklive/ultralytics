"""交互式 crop 类别复核器：点格子弹菜单，一键改判/标废，实时回写 labels。

用法:
  uv run python tools/crop_review.py --root data/wgc_review
（root 由 tools/review_pack.py 生成，含 crops/<拼音类>/、labels/train/、dataset.yaml）

操作:
  左键点格子      弹出菜单：点正确类别=改判；点 "X 非怪物"=标废；Esc/点空白=取消
  右键点格子      快捷标废（不经菜单）
  点右侧类名      切换到该类视图
  滚轮 / 拖滚动条  滚动；PgUp/PgDn/Home/End/方向键 同样可用
  u               撤销上一次操作
  q               退出

每次操作立即写 labels/train/<帧>.txt 并移动 crop 文件（labels 是唯一事实源；
改完后重跑 review_pack.py 可刷新全部派生材料）。"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

CELL, LAB, COLS = 84, 22, 10
GRID_W = COLS * (CELL + 4) + 6
PANEL_W = 360
WIN_W, WIN_H = GRID_W + PANEL_W + 10, 840
ROW_H = CELL + LAB + 4
GRID_Y0 = 32   # 顶部类别标签页高度，网格从这行开始
FONT = cv2.FONT_HERSHEY_SIMPLEX


def imread_u(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def py(name):
    from pypinyin import lazy_pinyin
    return "".join(lazy_pinyin(name)) or name


class CropReview:
    def __init__(self, root: Path):
        self.root = root
        self.crop_root = root / "crops"
        self.bad_dir = root / "crops_bad"
        self.lbl_dir = root / "labels" / "train"
        cfg = yaml.safe_load((root / "dataset.yaml").read_text(encoding="utf-8"))
        self.names = {int(k): v for k, v in cfg["names"].items()}
        self.id2py = {cid: py(nm) for cid, nm in self.names.items()}
        self.py2id = {v: k for k, v in self.id2py.items()}
        idx_path = root / "crops_vis" / "index.json"
        idx = json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else {}
        # items: {拼音类: [ {file, frame, cls, box}, ... ]}；index.json 缺失则现场扫描
        self.items = {}
        for cname in sorted(self.py2id):
            d = self.crop_root / cname
            if idx.get(cname):
                self.items[cname] = [it for it in idx[cname]
                                     if (self.crop_root / cname / it["file"]).exists()]
            elif d.is_dir():
                self.items[cname] = [{"file": f.name, "frame": f.name.split("__")[0],
                                      "cls": self.py2id[cname], "box": None}
                                     for f in sorted(d.glob("*.jpg"))]
        self.classes = [c for c in sorted(self.items) if self.items[c]]
        self.view = self.classes[0] if self.classes else None
        self.scroll = 0
        self.max_scroll = 0
        self.sel = None          # (类名, 序号)
        self.menu = None         # {x,y,w,h,items:[(label,action)],hover}
        self.undo_op = None
        self.thumb_cache = {}
        self.msg = "click cell -> menu: pick correct class / X=bad"
        self.window = "crop review | click cell=menu  right-click=bad  u=undo  q=quit"
        self._drag_sb = False

    # ---------- 标签读写 ----------

    def _label_path(self, frame):
        return self.lbl_dir / f"{frame}.txt"

    def _find_line(self, frame, cls, box):
        """按类别+像素框(容差3px)定位 label 行号。"""
        img = imread_u(self.root / "images" / "train" / f"{frame}.jpg")
        H, W = img.shape[:2]
        lines = self._label_path(frame).read_text(encoding="utf-8").splitlines()
        for i, l in enumerate(lines):
            p = l.split()
            if len(p) < 5 or int(p[0]) != cls:
                continue
            cx, cy, w, h = (float(x) for x in p[1:5])
            x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
            x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
            if all(abs(a - b) <= 3 for a, b in zip((x1, y1, x2, y2), box)):
                return i
        return None

    def _rewrite(self, frame, lines):
        self._label_path(frame).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _set_cls(self, frame, line_no, new_cls):
        p = self._label_path(frame)
        lines = p.read_text(encoding="utf-8").splitlines()
        parts = lines[line_no].split()
        parts[0] = str(new_cls)
        lines[line_no] = " ".join(parts)
        self._rewrite(frame, lines)

    def _del_line(self, frame, line_no):
        p = self._label_path(frame)
        lines = p.read_text(encoding="utf-8").splitlines()
        removed = lines.pop(line_no)
        self._rewrite(frame, lines)
        return removed

    # ---------- 操作 ----------

    def reassign(self, to_cls):
        """把选中格子改判到 to_cls（拼音类名）。"""
        frm, i = self.sel
        it = self.items[frm][i]
        line_no = self._find_line(it["frame"], it["cls"], it["box"]) if it["box"] else None
        if line_no is None and it["box"] is None:
            self.msg = "[skip] no box info"
            return
        src = self.crop_root / frm / it["file"]
        dst = self.crop_root / to_cls / it["file"]
        if dst.exists():
            self.msg = f"[skip] exists: {to_cls}/{it['file']}"
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        new_cls = self.py2id[to_cls]
        if line_no is not None:
            self._set_cls(it["frame"], line_no, new_cls)
        self.undo_op = ("reassign", to_cls, frm, it, line_no, i)
        self.items.setdefault(to_cls, []).append({**it, "cls": new_cls})
        del self.items[frm][i]
        self.sel = None
        self.msg = f"moved -> {to_cls}  ({it['file']})"

    def mark_bad(self):
        """选中格子标记为误检：移入 crops_bad/ 并删 label 行。"""
        frm, i = self.sel
        it = self.items[frm][i]
        src = self.crop_root / frm / it["file"]
        self.bad_dir.mkdir(parents=True, exist_ok=True)
        line_no = self._find_line(it["frame"], it["cls"], it["box"]) if it["box"] else None
        line_text = None
        if line_no is not None:
            line_text = self._del_line(it["frame"], line_no)
        shutil.move(str(src), str(self.bad_dir / it["file"]))
        self.undo_op = ("bad", frm, it, line_no, line_text, i)
        del self.items[frm][i]
        self.sel = None
        self.msg = f"marked BAD  ({it['file']})"

    def undo(self):
        if not self.undo_op:
            self.msg = "nothing to undo"
            return
        op = self.undo_op
        if op[0] == "reassign":
            _, to_cls, frm, it, line_no, i = op
            # 1) 从目标类内存列表移除（undo 不生效的根源就是漏了这步）
            lst = self.items.get(to_cls, [])
            for k, x in enumerate(lst):
                if x["file"] == it["file"]:
                    del lst[k]
                    break
            # 2) 文件移回原目录
            src = self.crop_root / to_cls / it["file"]
            if src.exists():
                shutil.move(str(src), str(self.crop_root / frm / it["file"]))
            # 3) label 行还原类别
            if line_no is not None:
                self._set_cls(it["frame"], line_no, it["cls"])
            # 4) 回原类列表原位置
            self.items[frm].insert(min(i, len(self.items[frm])), it)
            self.thumb_cache.pop((to_cls, it["file"]), None)
        else:  # bad
            _, frm, it, line_no, line_text, i = op
            src = self.bad_dir / it["file"]
            if src.exists():
                shutil.move(str(src), str(self.crop_root / frm / it["file"]))
            if line_no is not None and line_text:
                p = self._label_path(it["frame"])
                lines = p.read_text(encoding="utf-8").splitlines()
                lines.insert(min(line_no, len(lines)), line_text)
                self._rewrite(it["frame"], lines)
            self.items[frm].insert(min(i, len(self.items[frm])), it)
            self.thumb_cache.pop((frm, it["file"]), None)
        self.undo_op = None
        self.msg = "undone"

    # ---------- 绘制 ----------

    def thumb(self, cname, it):
        key = (cname, it["file"])
        if key not in self.thumb_cache:
            img = imread_u(self.crop_root / cname / it["file"])
            if img is None:
                img = np.full((CELL, CELL, 3), 128, np.uint8)
            elif max(img.shape[:2]) != CELL:
                img = cv2.resize(img, (CELL, CELL), interpolation=cv2.INTER_NEAREST)
            self.thumb_cache[key] = img
        return self.thumb_cache[key]

    def draw(self):
        canvas = np.full((WIN_H, WIN_W, 3), 245, np.uint8)
        cv2.rectangle(canvas, (0, 0), (GRID_W, WIN_H), (255, 255, 255), -1)
        # 顶部类别标签页（点击切换视图）
        if self.classes:
            tw = (GRID_W - 8) // len(self.classes)
            for k, cname in enumerate(self.classes):
                tx = 4 + k * tw
                cur = cname == self.view
                cv2.rectangle(canvas, (tx, 2), (tx + tw - 2, GRID_Y0 - 2),
                              (180, 255, 180) if cur else (225, 225, 225), -1)
                cv2.putText(canvas, cname[:9], (tx + 4, GRID_Y0 - 9), FONT, 0.42,
                            (0, 60, 0) if cur else (90, 90, 90), 1)
        items = self.items.get(self.view, [])
        rows = (len(items) + COLS - 1) // COLS
        self.max_scroll = max(0, rows * ROW_H + 8 - (WIN_H - GRID_Y0))
        self.scroll = min(self.scroll, self.max_scroll)
        first_row = self.scroll // ROW_H
        last_row = min(rows, first_row + (WIN_H - GRID_Y0) // ROW_H + 1)
        for r in range(first_row, last_row):
            for c in range(COLS):
                i = r * COLS + c
                if i >= len(items):
                    break
                y = GRID_Y0 + r * ROW_H - self.scroll
                x = 4 + c * (CELL + 4)
                if y + CELL <= 0 or y >= WIN_H:
                    continue
                ty, by = max(GRID_Y0, y), min(WIN_H, y + CELL)  # 视口边缘部分粘贴
                if (self.view, i) == self.sel:
                    cv2.rectangle(canvas, (x - 2, max(GRID_Y0, y) - 2),
                                  (x + CELL + 2, min(WIN_H, y + CELL + 2)),
                                  (0, 200, 255), 3)
                canvas[ty:by, x:x + CELL] = self.thumb(self.view, items[i])[ty - y:by - y]
                cv2.putText(canvas, str(i + 1), (x + 2, y + CELL + 14),
                            FONT, 0.45, (0, 0, 220), 1)
        # 滚动条（可点击/拖动）
        if self.max_scroll > 0:
            th = max(30, int((WIN_H - GRID_Y0) * WIN_H / (rows * ROW_H + 8)))
            ty = GRID_Y0 + int(self.scroll / self.max_scroll * (WIN_H - GRID_Y0 - th))
            cv2.rectangle(canvas, (GRID_W - 8, GRID_Y0), (GRID_W - 2, WIN_H), (200, 200, 200), -1)
            cv2.rectangle(canvas, (GRID_W - 8, ty), (GRID_W - 2, ty + th), (120, 120, 255), -1)
        # 右侧面板
        px0 = GRID_W + 6
        cv2.rectangle(canvas, (px0, 0), (WIN_W, WIN_H), (235, 235, 235), -1)
        cv2.putText(canvas, f"view: {self.view} ({len(items)})", (px0 + 10, 28),
                    FONT, 0.6, (30, 30, 30), 2)
        y0 = 50
        for k, cname in enumerate(self.classes):
            yy = y0 + k * 30
            cv2.rectangle(canvas, (px0 + 6, yy - 18), (WIN_W - 10, yy + 6),
                          (210, 255, 210) if self.view == cname else (240, 240, 240), -1)
            cv2.putText(canvas, f"{cname} ({len(self.items[cname])})",
                        (px0 + 14, yy), FONT, 0.55,
                        (0, 0, 0) if self.view == cname else (60, 60, 60), 1)
        yy = y0 + len(self.classes) * 30 + 24
        cv2.putText(canvas, "ops:", (px0 + 10, yy), FONT, 0.5, (0, 0, 0), 1)
        cv2.putText(canvas, "click cell -> menu", (px0 + 10, yy + 22), FONT, 0.45, (80, 80, 80), 1)
        cv2.putText(canvas, "right-click = BAD", (px0 + 10, yy + 42), FONT, 0.45, (80, 80, 80), 1)
        cv2.putText(canvas, "u=undo  q=quit", (px0 + 10, yy + 62), FONT, 0.45, (80, 80, 80), 1)
        if self.sel:
            it = self.items[self.sel[0]][self.sel[1]]
            cv2.putText(canvas, f"sel: {it['file']}"[:70], (6, WIN_H - 10),
                        FONT, 0.5, (0, 60, 0), 1)
        cv2.putText(canvas, self.msg[:80], (6, WIN_H - 30), FONT, 0.5, (0, 0, 200), 1)
        # 弹出菜单（最上层）
        if self.menu:
            m = self.menu
            cv2.rectangle(canvas, (m["x"], m["y"]), (m["x"] + m["w"], m["y"] + m["h"]),
                          (50, 50, 50), -1)
            cv2.rectangle(canvas, (m["x"], m["y"]), (m["x"] + m["w"], m["y"] + m["h"]),
                          (255, 255, 255), 1)
            for j, (label, _) in enumerate(m["items"]):
                yy = m["y"] + 6 + j * 26
                hover = m.get("hover") == j
                if hover:
                    cv2.rectangle(canvas, (m["x"] + 3, yy - 4),
                                  (m["x"] + m["w"] - 3, yy + 18), (90, 90, 90), -1)
                col = (0, 200, 120) if label.startswith("X ") else (255, 255, 255)
                cv2.putText(canvas, label, (m["x"] + 10, yy + 10), FONT, 0.5, col, 1)
        return canvas

    # ---------- 交互 ----------

    def _open_menu(self, x, y):
        """在格子处弹出类别菜单（含改判目标类、非怪物、取消）。"""
        items = []
        for cname in self.classes:
            mark = "  <-now" if cname == self.view else ""
            items.append((f"{self.py2id[cname]:2d} {cname}{mark}", ("to", cname)))
        items.append(("X  BAD (not a monster)", ("bad",)))
        items.append(("cancel", ("cancel",)))
        w, ih = 240, 26
        h = 12 + len(items) * ih
        mx = min(x + 6, GRID_W - w - 4)
        my = min(max(4, y - 10), WIN_H - h - 4)
        self.menu = {"x": int(mx), "y": int(my), "w": w, "h": h, "items": items,
                     "hover": None}

    def _menu_hit(self, x, y):
        m = self.menu
        if not (m["x"] <= x <= m["x"] + m["w"] and m["y"] <= y <= m["y"] + m["h"]):
            return None
        j = (y - m["y"] - 6) // 26
        return j if 0 <= j < len(m["items"]) else None

    def _scroll_to(self, frac):
        self.scroll = int(max(0.0, min(1.0, frac)) * self.max_scroll)

    def grid_hit(self, mx, my):
        """返回命中格子的 (类名, 序号)；未命中 None。"""
        if mx >= GRID_W - 10 or my < GRID_Y0:  # 滚动条/标签页区域
            return None
        row = (my - GRID_Y0 + self.scroll) // ROW_H
        col = (mx - 4) // (CELL + 4)
        if col >= COLS or row < 0:
            return None
        i = row * COLS + col
        items = self.items.get(self.view, [])
        return (self.view, i) if 0 <= i < len(items) else None

    def on_mouse(self, event, x, y, flags, *args):
        if event == cv2.EVENT_MOUSEWHEEL:
            # 滚轮增量在 flags（cv2.getMouseWheelDelta），y 是光标坐标
            try:
                delta = cv2.getMouseWheelDelta(flags)
            except AttributeError:
                delta = 1 if flags > 0 else -1
            self.scroll = max(0, self.scroll - 3 * ROW_H * (1 if delta > 0 else -1))
            return
        if event == cv2.EVENT_MOUSEMOVE:
            if self.menu:
                self.menu["hover"] = self._menu_hit(x, y)
            elif self._drag_sb:
                self._scroll_to(max(0.0, (y - GRID_Y0)) / max(1, WIN_H - GRID_Y0))
            return
        if event == cv2.EVENT_LBUTTONUP:
            self._drag_sb = False
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.menu:
                j = self._menu_hit(x, y)
                if j is None:
                    self.menu = None
                    return
                _, action = self.menu["items"][j]
                self.menu = None
                if action[0] == "to":
                    self.reassign(action[1])
                elif action[0] == "bad":
                    self.mark_bad()
                return
            if x > GRID_W - 10 and x < GRID_W:      # 滚动条
                self._drag_sb = True
                self._scroll_to(max(0.0, (y - GRID_Y0)) / max(1, WIN_H - GRID_Y0))
                return
            if y < GRID_Y0:                          # 顶部类别标签页
                if self.classes:
                    tw = (GRID_W - 8) // len(self.classes)
                    k = (x - 4) // tw
                    if 0 <= k < len(self.classes):
                        self.view = self.classes[k]
                        self.scroll = 0
                        self.sel = None
                return
            hit = self.grid_hit(x, y)
            if hit:
                self.sel = hit
                self._open_menu(x, y)
                return
            px0 = GRID_W + 6
            if x >= px0:                             # 右侧类名 = 切换视图
                k = (y - 50 + 18) // 30
                if 0 <= k < len(self.classes):
                    self.view = self.classes[k]
                    self.scroll = 0
                    self.sel = None
            return
        if event == cv2.EVENT_RBUTTONDOWN:
            hit = self.grid_hit(x, y)
            if hit:
                self.sel = hit
                self.menu = None
                self.mark_bad()

    def run(self):
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.on_mouse)
        while True:
            cv2.imshow(self.window, self.draw())
            raw = cv2.waitKey(30)
            if raw < 0:
                continue
            key = raw & 0xFF
            ext = (raw >> 16) & 0x1FF  # Windows 扩展键（PgUp=0x21 等）
            if key in (ord("q"),):
                break
            if key == 27:  # Esc：关菜单，菜单没开才退出
                if self.menu:
                    self.menu = None
                else:
                    break
            elif key == ord("d") and self.sel and not self.menu:
                self.mark_bad()
            elif key == ord("u"):
                self.undo()
            elif key in (ord("["),):
                k = self.classes.index(self.view)
                self.view = self.classes[(k - 1) % len(self.classes)]
                self.scroll = 0
            elif key in (ord("]"),):
                k = self.classes.index(self.view)
                self.view = self.classes[(k + 1) % len(self.classes)]
                self.scroll = 0
            elif key in (ord(","), ord(".")):
                self.scroll += (-1 if key == ord(",") else 1) * (WIN_H - ROW_H)
            elif ext == 0x21:   # PgUp
                self.scroll -= WIN_H
            elif ext == 0x22:   # PgDn
                self.scroll += WIN_H
            elif ext == 0x24:   # Home
                self.scroll = 0
            elif ext == 0x23:   # End
                self.scroll = self.max_scroll
            elif ext == 0x26:   # Up
                self.scroll -= ROW_H
            elif ext == 0x28:   # Down
                self.scroll += ROW_H
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="review_pack.py 生成的复核数据集目录")
    a = ap.parse_args()
    CropReview(Path(a.root)).run()


if __name__ == "__main__":
    main()
