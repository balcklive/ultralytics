# AutoDL GPU 训练方法论（工作文档）

> **目标**：以后所有训练统一走 **AutoDL GPU 实例**，作为标准方法论（本机仅 CPU 22 核，完整训练跑不动）。
> 本机角色 = **数据整备 + 权重归档**；AutoDL 角色 = **GPU 训练**。数据/权重通过 SFTP 传递，不走阿里云 PAI/OSS。
> 实现文件在 `cloud/autodl/`；本文是完整流程与踩坑记录（2026-09-06 首次全流程跑通）。

## 0. 何时用 / 前置

- 本机无 GPU（`nvidia-smi` 无、torch 为 `+cpu`）；训练一律上 AutoDL。
- AutoDL 实例：`connect.bjb1.seetacloud.com:端口`，root + 密码（由用户提供）。典型配置 **RTX 4080 32GB**。
- 实例 env：`/root/miniconda3/bin/python`（**py3.8.10**）含 torch(CUDA 13)+cv2；**ultralytics 只在项目目录里**（`/root/ultralytics-YOLO26/`，v8.4.7），**不是 pip 装的**。
- 数据盘 `/root/autodl-tmp`（默认 50G，远大于系统盘 `/`）；数据集/权重/日志都放这里。

## 1. 方法论总览

```
本地数据整备(merge_maps) → SFTP 传数据+基础权重到 Autodl → 写 py3.8 训练脚本 → GPU 跑
→ SFTP 拉回 best.pt → 本地 export best.onnx + best.names → 归档 weights/<日期>/
```

- **只有数据 + 基础权重上云**（仓库代码**不上云**——本地工具是 py3.12+ 语法，远端 py3.8 跑不了）。
- 远端**不装 onnx**（导出 onnx 在本地用 `tools/export_onnx.py` 做）。

## 2. 步骤 ① 本地数据整备

用 `tools/merge_maps.py` 把各图/各源合并成统一训练集（`data/unified_<round>/`，73 类，**无 `path` 键**，`--dry-run` 先看规模）：

```bash
uv run python tools/merge_maps.py --out data/unified_20260906 \
  --master data/wgc_review/dataset.yaml \
  --source data/wgc_review --source data/annotated_73 \
  --source data/maps/sheshou_xunlianchang
```

> 基础权重用每轮归档版（如 `weights/20260902/best.pt` = 上一轮最优），从最强模型续训。

## 3. 步骤 ② SFTP 上传数据 + 权重（本机无 sshpass，用 paramiko）

`cloud/autodl/transfer_upload.py`（paramiko，递归上传，剔除 `*.cache`）：

```bash
uv run --with paramiko python cloud/autodl/transfer_upload.py prod
```
脚本把 `artifacts/data_pack/unified_<round>/` 传到 `远端/root/ultralytics-YOLO26/datasets/unified_<round>/`、基础权重传 `远端/root/ultralytics-YOLO26/best.pt`。

- 文件多时（>几十个）SFTP 逐文件传慢 → 可 `.tar.gz` 单文件传、远端解压（见 CLAUDE.md）。
- 远端 dataset.yaml 无 `path` 键即通用（ultralytics 以 yaml 目录为根），故直接能用。

## 4. 步骤 ③ 写 py3.8 训练脚本并跑

远端跑训练前**必须**让 ultralytics 可导入：项目里的 ultralytics 不是 pip 包。两种方式：
- 脚本首行 `sys.path.insert(0, '/root/ultralytics-YOLO26')`（**推荐**，脚本放任意目录）；或
- `cd /root/ultralytics-YOLO26 && python`（此时 cwd 在 path 上）。

`cloud/autodl/train_remote.py` 模板（上传到远端 `:40777` 后执行）：

```python
import sys; sys.path.insert(0, '/root/ultralytics-YOLO26')
from ultralytics import YOLO
YOLO('/root/ultralytics-YOLO26/best.pt').train(
    data='/root/ultralytics-YOLO26/datasets/unified_20260906/dataset.yaml',
    epochs=100, imgsz=640, batch=-1, device=0,
    project='/root/ultralytics-YOLO26/runs', name='unified-v1',
    cache=True, patience=30, workers=8)
```

后台跑（`nohup ... &`，日志落 `/root/autodl-tmp/train.log`），完成后日志出现 `TRAIN_DONE`。
- `batch=-1` GPU 自动（4080 上 batch≈40，243 训练图约 6 步/epoch）；`cache=True` 免逐 epoch 读盘。
- RTX 4080 上 304 帧/100ep ≈ **7 分钟**；`patience 30` 早停。
- 结果在 `runs/unified-v1/`（`best.pt`/`last.pt`/`results.csv`/`results.png`）。

## 5. 步骤 ④ 导出 ONNX + names + 归档（本地做）

ONNX 远端没装 `onnx`，**在本地导出**（C# 运行时同款 `tools/export_onnx.py`）：

```bash
# 先 SFTP 拉回 best.pt 到 weights/<日期>/
uv run python tools/export_onnx.py --model weights/<日期>/best.pt --output weights/<日期>/best.onnx --imgsz 640
# 生成 best.names（每行=类 id 的中文名）
uv run python -c "import yaml,pathlib;n=yaml.safe_load(pathlib.Path('data/unified_<round>/dataset.yaml').read_text(encoding='utf-8'))['names'];pathlib.Path('weights/<日期>/best.names').write_text(''.join(f'{n[i]}\n' for i in range(len(n))), encoding='utf-8')"
```
归档 `weights/<日期>/best.{pt,onnx,names}`。

## 6. 结果口径（重要）

- 本轮 val 是**更大更难**的切分（如 61 帧 vs full-v1 的 23 帧），**直接和旧模型 mAP 比没意义**。
- 看涨没涨，用**同一口径**：把新 best 在旧 val 集上跑（或反之），或相信**逐类 mAP50**（尤其被新数据补强的类）。
- 每类意义：`recall` 低=漏检；`mAP50` 低=框得不准或类混（换色家族）。样本 <10 帧的类噪声大，勿当结论。

## 7. 已踩坑（照抄避免）

| 坑 | 现象 | 规避 |
|---|---|---|
| 远端无 pip 版 ultralytics | `ModuleNotFoundError: No module named 'ultralytics'` | `sys.path.insert(0,'/root/ultralytics-YOLO26')` 或 `cd 项目目录` |
| 远端 base 是 py3.8、本地工具是 py3.12+ | 本地 merge_maps/export 传上去 `int \| None` 语法崩 | **仓库代码不上云**，只传数据+权重 |
| 远端没装 `onnx` | `ModelNotFound: onnx` | **在本地**用 `tools/export_onnx.py` 导出 onnx |
| 本机无 sshpass | 交互密码 ssh 连不上 | 用 paramiko 脚本（`uv run --with paramiko`）或 `uvx --from paramiko` |
| 文件多散传慢 | SFTP 逐文件超时/慢 | 打 `.tar.gz` 单文件传、远端解压 |
| `nohup ... &` 命令的 exec 通道超时 | paramiko `TimeoutError`（后台进程占着通道） | 正常；py 已在跑，另行 `cat` 日志验证，别卡在收尾 |
| `python -c` 多引号嵌套 | 引号被剥成 `YOLO(/path)` 语法错 | 用**脚本文件**（SFTP 上传再执行），别写 `-c` 长串 |

## 8. 与数据/飞轮的衔接

对应 `docs/auto_labeling_pipeline.md` 第 8 步「人工启动训练」的 AutoDL 替代：新批复核集并入 `data/maps/<地图>/` → `merge_maps.py` 合并 → 上面 ②③④ 即完成一轮新训练，权重归位 `weights/<日期>/`。

相关实现：`cloud/autodl/`（文件职责见其 `CLAUDE.md`）。
