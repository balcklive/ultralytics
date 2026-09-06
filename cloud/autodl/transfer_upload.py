# cloud/autodl/transfer_upload.py
r"""SFTP 上传训练数据 + 基础权重到 AutoDL 实例（本机无 sshpass，用 paramiko）。

凭据从环境变量取（**不硬编码**）：AUTODL_HOST / AUTODL_PORT / AUTODL_USER / AUTODL_PASS。
上传内容（见 `docs/autodl_training.md` ②）：
  --data-dir  (unified 数据包, 通常为 artifacts/cloud_round_<round> 或 data/unified_<round>)
  --weight    基础权重 best.pt (如 weights/20260902/best.pt)
上传到远端：
  数据 → <REMOTE_BASE>/datasets/<round>/         REMOTE_BASE 默认 /root/ultralytics-YOLO26
  权重 → <REMOTE_BASE>/best.pt
排除 *.cache / __pycache__（内嵌本机绝对路径，远端无用且会干扰）。

用法:
  AUTODL_HOST=connect.bjb1.seetacloud.com AUTODL_PORT=40777 AUTODL_USER=root AUTODL_PASS=xxx \
  uv run --with paramiko python cloud/autodl/transfer_upload.py \
    --data-dir artifacts/cloud_round_20260906 --weight weights/20260902/best.pt --round 20260906
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import paramiko


def _req(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise SystemExit(f"need env {name} (AutoDL SSH credential)")
    return val


def _mkdirs(sftp: paramiko.SFTPClient, path: str) -> None:
    cur = ""
    for part in path.strip("/").split("/"):
        cur += "/" + part
        try:
            sftp.stat(cur)
        except OSError:
            sftp.mkdir(cur)


def upload_dir(sftp: paramiko.SFTPClient, local: Path, remote: str) -> int:
    _mkdirs(sftp, remote)
    n = 0
    for p in sorted(local.rglob("*")):
        if any(x in p.parts for x in ("__pycache__",)) or p.suffix in (".cache", ".pyc"):
            continue
        dst = remote + "/" + p.relative_to(local).as_posix()
        if p.is_dir():
            _mkdirs(sftp, dst)
        else:
            sftp.put(str(p), dst)
            n += 1
    return n


def main() -> None:
    """Parse command-line arguments and upload."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True, help="本地数据包目录（含 images/labels/dataset.yaml）")
    ap.add_argument("--weight", type=Path, required=True, help="本地基础权重 best.pt")
    ap.add_argument("--round", required=True, help="OSS/远端 round 目录名，如 20260906-unified")
    ap.add_argument("--remote-base", default="/root/ultralytics-YOLO26", help="远端 base 目录")
    args = ap.parse_args()

    host = _req("AUTODL_HOST")
    port = int(_req("AUTODL_PORT"))
    user = _req("AUTODL_USER")
    pw = _req("AUTODL_PASS")

    if not args.data_dir.is_dir() or not (args.data_dir / "dataset.yaml").is_file():
        raise SystemExit(f"data-dir not a dataset: {args.data_dir}")
    if not args.weight.is_file():
        raise SystemExit(f"weight not found: {args.weight}")

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=port, username=user, password=pw, timeout=30)
    s = c.open_sftp()
    data_remote = f"{args.remote_base}/datasets/{args.round}"
    n = upload_dir(s, args.data_dir, data_remote)
    print(f"uploaded {n} files -> {data_remote}")
    s.put(str(args.weight), f"{args.remote_base}/best.pt")
    print(f"uploaded weight -> {args.remote_base}/best.pt")
    print(f"verifying: train={len(list((args.data_dir / 'images' / 'train').iterdir())) if (args.data_dir / 'images' / 'train').is_dir() else 'n/a'}")
    c.close()
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
