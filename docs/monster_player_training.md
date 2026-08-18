# 怪物与玩家检测训练

本仓库提供 `tools/yolo_data.py`，用于把 `maoxiandao` 的旧版怪物模型转成新版 YOLO 的两类数据集。

## 环境

本地训练环境固定使用 Python 3.12。首次安装：

```powershell
uv sync
uv sync --extra dev --extra solutions
```

之后可使用 `.venv\Scripts\python.exe` 运行下面的命令。若机器没有 Python 3.12，先执行 `uv python install 3.12`。

## 1. 自动生成怪物标注

将游戏截图或视频抽帧放入一个目录：

```powershell
python tools/yolo_data.py auto-label `
  --images D:\path\to\frames `
  --old-model ..\maoxiandao\bot\resource\bundles\models\best.pt `
  --output datasets\monster_player
```

旧模型识别到的所有类别都会统一写成类别 `0: monster`。输出数据集会按固定随机种子拆分为 `train` 和 `val`，原图片会复制到数据集目录。

## 2. 标注玩家

```powershell
python tools/yolo_data.py annotate --dataset datasets\monster_player --split train
python tools/yolo_data.py annotate --dataset datasets\monster_player --split val
```

在窗口中左键拖动新增玩家框，右键点击删除框，`s` 保存，`n`/空格下一张，`p` 上一张，`c` 清除当前图片的所有玩家框，`q` 退出。绿色是旧模型生成的怪物框，橙色是人工添加的玩家框。

## 3. 训练新版模型

```powershell
python tools/yolo_data.py train `
  --dataset datasets\monster_player `
  --model yolo26n.pt `
  --epochs 100 `
  --imgsz 640
```

也可以直接使用 `configs/monster_player.yaml` 作为数据配置。建议先人工抽查自动生成的怪物框，再开始训练；旧模型漏检的怪物必须手工补框，否则会成为训练中的漏标负样本。
