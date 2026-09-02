"""批量调用豆包 VL 对怪物裁剪图做类别判断，统计 top-1 准确率。

输入目录内的图片文件名约定为 `<cls>_<类名>_<序号>.png`（cls 为真值类别 id）。
调用 C:\\Users\\admin\\.claude\\skills\\vlm-image\\scripts\\vlm_image.py（标准库实现，
凭据自动解析），用线程池并行。结果逐行追加到 JSONL，支持断点续跑。

用法:
  uv run python tools/vlm_classify.py --crops artifacts/vlm_test_sample \
      --classes artifacts/vlm_classes.txt --out artifacts/vlm_eval.jsonl --workers 4
"""

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

VLM_SCRIPT = Path(r"C:\Users\admin\.claude\skills\vlm-image\scripts\vlm_image.py")

PROMPT_TMPL = (
    "这是2D横版游戏画面中一个小怪物的放大截图。候选怪物类别列表：{classes}。"
    "请判断这个怪物最可能是哪一类。注意：颜色、体型、五官是关键区分特征。"
    "如果都不像，回答'未知'。"
)

SCHEMA = {
    "type": "object",
    "properties": {
        "class": {"type": "string", "description": "最可能的类别名，必须从候选列表中选或填未知"},
        "confidence": {"type": "number", "description": "0-1"},
    },
    "required": ["class", "confidence"],
}


def classify_one(img_path, classes_str, model=None):
    """调用 skill 脚本分类单张图，返回 (pred, conf)。失败返回 (None, None)。"""
    cmd = [
        sys.executable, str(VLM_SCRIPT), str(img_path),
        "-p", PROMPT_TMPL.format(classes=classes_str),
        "--schema", json.dumps(SCHEMA, ensure_ascii=False), "--json", "-q",
    ]
    if model:
        cmd += ["--model", model]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return None, None
    m = re.search(r"\{.*\}", r.stdout, re.DOTALL)
    if not m:
        return None, None
    try:
        d = json.loads(m.group(0))
        return d.get("class"), d.get("confidence")
    except json.JSONDecodeError:
        return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crops", required=True)
    ap.add_argument("--classes", required=True, help="逗号分隔类名列表的文件")
    ap.add_argument("--out", default="artifacts/vlm_eval.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    classes_str = Path(args.classes).read_text(encoding="utf-8").strip()

    crops = sorted(Path(args.crops).glob("*.png"))
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["file"])
            except Exception:
                pass
    todo = [c for c in crops if c.name not in done]
    print(f"total {len(crops)}, done {len(done)}, todo {len(todo)}")

    with out_path.open("a", encoding="utf-8") as fout, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(classify_one, c, classes_str, args.model): c for c in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            crop = futs[fut]
            pred, conf = fut.result()
            gt_id = int(crop.name.split("_")[0])
            rec = {"file": crop.name, "gt_id": gt_id,
                   "pred": pred, "conf": conf}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if n % 20 == 0:
                print(f"  {n}/{len(todo)}")

    # 统计
    recs = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()]
    ok = sum(1 for r in recs if r["pred"] is not None
             and classes_str.split(",")[r["gt_id"]] == r["pred"])
    unknown = sum(1 for r in recs if r["pred"] in (None, "未知"))
    fail = sum(1 for r in recs if r["pred"] is not None and r["pred"] != "未知"
               and r["pred"] not in classes_str.split(","))
    print(f"\n样本 {len(recs)}  top-1 正确 {ok}  准确率 {ok/len(recs):.1%}  "
          f"未知/失败 {unknown}  非法输出 {fail}")


if __name__ == "__main__":
    main()
