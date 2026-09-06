# cloud/autodl/

AutoDL GPU 实例训练方法（标准训练路径，替代已删除的阿里云 PAI-DLC）。本机仅 CPU，训练统一走 AutoDL。

> 完整方法论见 `docs/autodl_training.md`；脚本凭据走环境变量（不硬编码密码）。

## 文件清单

| 文件 | 说明 |
| --- | --- |
| `transfer_upload.py` | SFTP 上传数据包 + 基础权重到 AutoDL（paramiko，本机无 sshpass）。凭据 env：`AUTODL_HOST/PORT/USER/PASS`；`--data-dir/--weight/--round`；排除 `*.cache`/`__pycache__`。数据→`<base>/datasets/<round>/`、权重→`<base>/best.pt` |
| `train_remote.py` | **远端执行的 py3.8 训练模板**：首行 `sys.path.insert(0,'/root/ultralytics-YOLO26')` 才能 import ultralytics（该项目是源码包，非 pip）。env `TRAIN_BASE/DATA/NAME/EPOCHS`；`YOLO(base).train(batch=-1, device=0, cache=True, patience=30, workers=8)` |

## 调用链

- 本地 `python tools/merge_maps.py` 合并数据集（`data/unified_<round>/`，无 `path`）→ `transfer_upload.py` 上传 → 用户把 `train_remote.py` 上传远端并 `nohup` 跑 → SFTP 拉回 `best.pt` → 本地 `tools/export_onnx.py` 导出 `best.onnx` + 生成 `best.names` → 归档 `weights/<日期>/`。

## 关键规则 / 注意事项

- **仓库代码不上云**：本地工具是 py3.12+（`int | None` 等），远端 base 是 py3.8，跑不了。只传数据 + 权重。
- **远端 ultralytics 非 pip 包**：必须 `sys.path.insert(0,'/root/ultralytics-YOLO26')` 或 `cd 项目目录`，否则 `ModuleNotFoundError`。
- **远端无 `onnx`**：导出 onnx 一律在本地用 `tools/export_onnx.py`（`--imgsz 640` 与运行时一致）。
- 上传文件多时（>几十个）SFTP 慢 → 打包 `.tar.gz` 单文件、远端 `tar xzf` 解压。
- `batch=-1` 走 GPU 自动；`cache=True` 免逐 epoch 读盘（数据盘 `/root/autodl-tmp` 大，可放数据）。
- 实例用完可关闭省费用；数据在本地有份，远端无需保留。
- 运行时：RTX 4080 上 304 帧/100ep ≈ 几分钟；`patience 30` 早停。
