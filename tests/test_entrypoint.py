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
    proc = subprocess.run(["bash", str(ENTRY)], env=env, capture_output=True, text=True, check=False)
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


def test_print_only_includes_close_mosaic():
    assert "--close-mosaic 10" in _run({"CLOSE_MOSAIC": "10"})
