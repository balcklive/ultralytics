# YOLO 怪物与玩家训练工作流

本文档记录当前项目已经搭建完成的训练基础设施，以及从旧版 YOLO 生成数据、人工审核、标注玩家、训练新版模型和视频验证的完整用法。

## 当前方案

项目使用 `maoxiandao` 中的旧版模型识别怪物，再使用当前 Ultralytics YOLO 训练新版模型。

最终类别规划为：

```text
0: player
1: lanmugu
2: lvshuijing
3: hongwoniu
...
17: muyao
```

旧模型中的 `0: __background__` 不是标准 YOLO 目标类别，已从最终训练类别中移除。背景使用空标签文件表示。

## 已完成的代码产物

### 数据处理和标注工具

[tools/yolo_data.py](../tools/yolo_data.py) 提供以下命令：

| 命令                     | 用途                                     |
| ------------------------ | ---------------------------------------- |
| `auto-label`             | 使用旧版 YOLO 自动生成怪物检测框和裁剪图 |
| `apply-crop-review`      | 根据人工删除的裁剪图同步删除 YOLO 标签框 |
| `prepare-player-dataset` | 移除旧背景类别并设置 `0: player`         |
| `annotate`               | 标注、删除和调整玩家/怪物框              |
| `review-old`             | 可选的窗口式裁剪图审核工具               |
| `train`                  | 启动新版 YOLO 训练                       |

### 视频验证工具

[tools/predict_video.py](../tools/predict_video.py) 可以对视频逐帧推理，保存带框视频，并输出模型加载、单帧推理和端到端处理耗时。

### Python 环境

项目使用 `uv` 管理环境，Python 版本固定为 3.12：

```powershell
uv sync
uv sync --extra dev --extra solutions
```

环境解释器：

```text
.venv\Scripts\python.exe
```

## 数据目录

本地数据和模型已加入 `.gitignore`，不会提交到 Git：

```text
data/
models/
```

推荐目录结构：

```text
data/
├── images/                         # 原始游戏截图
├── old_yolo_dataset/               # 训练数据集
│   ├── dataset.yaml
│   ├── images/train/
│   ├── images/val/
│   └── labels/
└── old_yolo_review/
    └── crops/                      # 旧模型检测框裁剪图
        ├── manifest.json
        ├── 9_lvwoniu/
        └── ...

models/
└── yolo26s.pt                      # 训练起始模型
```

旧模型默认路径：

```text
..\maoxiandao\bot\resource\bundles\models\best.pt
```

## 1. 使用旧模型生成数据

将截图放入 `data/images`，然后执行：

```powershell
.venv\Scripts\python.exe tools/yolo_data.py auto-label `
  --images data/images `
  --old-model ..\maoxiandao\bot\resource\bundles\models\best.pt `
  --output data/old_yolo_dataset `
  --review-output data/old_yolo_review `
  --overwrite
```

该命令会：

- 将图片随机拆分为 `train` 和 `val`
- 保留旧模型的原始怪物类别 ID
- 生成标准 YOLO 标签
- 生成每个检测框的裁剪图
- 在 `data/old_yolo_review/crops/manifest.json` 中记录裁剪图与原标签框的对应关系

如果重新运行 `auto-label --overwrite`，会重新生成数据和标签，并覆盖人工修改结果。重新运行前应确认确实需要重建。

## 2. 审核旧模型检测结果

推荐使用文件夹审核方式，不需要 GUI。

直接检查并删除错误裁剪图：

```text
data/old_yolo_review/crops/
```

可以进入任意类别目录，例如：

```text
data/old_yolo_review/crops/9_lvwoniu/
```

删除错误的 `.jpg` 后，执行：

```powershell
.venv\Scripts\python.exe tools/yolo_data.py apply-crop-review `
  --dataset data/old_yolo_dataset `
  --crops data/old_yolo_review/crops
```

该命令会递归处理所有类别：

- 仍存在的裁剪图：保留对应标签框
- 被删除的裁剪图：删除对应标签框
- 原始图片不会删除
- 如果一张图片的所有框都被删除，其标签文件为空，可作为负样本

完成人工删除后不要再次运行 `review-old`，否则它可能重新生成审核裁剪图。

## 3. 设置最终类别映射

旧模型审核完成后执行一次：

```powershell
.venv\Scripts\python.exe tools/yolo_data.py prepare-player-dataset `
  --dataset data/old_yolo_dataset
```

该命令将数据集类别设置为：

```text
0: player
1–17: 怪物
```

重要：玩家标注完成后不要再次运行旧版背景迁移。执行任何会修改标签的命令前，建议先备份：

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item data/old_yolo_dataset/labels data/backups/labels_$stamp -Recurse
```

## 4. 标注和修正玩家、怪物框

标注训练集：

```powershell
.venv\Scripts\python.exe tools/yolo_data.py annotate `
  --dataset data/old_yolo_dataset `
  --split train
```

标注验证集：

```powershell
.venv\Scripts\python.exe tools/yolo_data.py annotate `
  --dataset data/old_yolo_dataset `
  --split val
```

标注器功能：

- `[` / `]`：切换当前类别
- 左键拖动空白区域：新增当前类别框
- 拖动已有框的四个角：调整框尺寸
- 右键点击框：删除错误框
- `c`：清除当前类别的所有框
- `s`：保存当前图片
- `n` / 空格：保存并进入下一张
- `p`：保存并返回上一张
- `q`：保存并退出

最后一张图片按 `n` 或空格会自动保存并退出。

玩家默认使用类别 `0`，怪物框可以通过 `[` / `]` 切换到对应旧模型类别后新增或修改。

## 5. 训练新版模型

使用当前项目中的起始模型训练：

```powershell
.venv\Scripts\python.exe tools/yolo_data.py train `
  --dataset data/old_yolo_dataset `
  --model models/yolo26s.pt `
  --epochs 100 `
  --imgsz 640 `
  --batch -1 `
  --device cpu
```

如果有 NVIDIA GPU，可以使用：

```powershell
--device 0
```

训练输出默认保存到当前项目：

```text
runs/train/monster-player/weights/best.pt
runs/train/monster-player/weights/last.pt
```

`best.pt` 是验证集效果最好的权重，`last.pt` 是最近一个训练周期的权重。

## 6. 验证模型

验证集评估：

```powershell
.venv\Scripts\python.exe yolo val `
  model=runs/train/monster-player/weights/best.pt `
  data=data/old_yolo_dataset/dataset.yaml
```

视频推理：

```powershell
.venv\Scripts\python.exe tools/predict_video.py `
  --model weights/best.pt `
  --source weights/test_video/tmp_jump_template_player.mp4 `
  --output weights/test_video/player_test_pred.mp4 `
  --device cpu `
  --conf 0.25
```

如果模型位于训练输出目录，可以使用绝对路径：

```powershell
.venv\Scripts\python.exe tools/predict_video.py `
  --model D:\02-code\04-officials\yoloe\runs\detect\runs\train\monster-player-2\weights\best.pt `
  --source weights/test_video/tmp_jump_template_player.mp4 `
  --output weights/test_video/player_test_pred.mp4 `
  --device cpu `
  --conf 0.25
```

视频脚本会输出：

- 模型加载耗时
- 首帧推理耗时
- 平均单帧推理耗时和 FPS
- 平均端到端处理耗时和 FPS
- 总耗时

此前一次 CPU 测试结果为：平均推理 `0.074s/frame`，端到端处理 `0.098s/frame`，约 `10.22 FPS`。

## 7. 评估结果解读

- `Precision`：检测结果中有多少是真目标
- `Recall`：真实目标中有多少被检测到
- `mAP50`：IoU 阈值为 0.50 时的平均精度
- `mAP50-95`：IoU 从 0.50 到 0.95、步长 0.05 的平均精度，更严格地衡量框的位置和尺寸

验证集只有少量图片时，指标可能偏乐观。最终应使用不同地图、不同视角和不同场景的图片进行独立测试。

## 8. 数据安全规则

以下命令会修改或重建数据，执行前应确认目标路径：

- `auto-label --overwrite`
- `apply-crop-review`
- `prepare-player-dataset`
- `annotate`

建议始终保留一份标签备份，并把原始截图、审核后的标签、最终训练集分开保存。Git 只保存脚本和配置，`data/`、`models/`、训练权重和视频输出均为本地产物。
