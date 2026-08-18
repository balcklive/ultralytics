# 怪物与玩家检测训练

本仓库提供 `tools/yolo_data.py`，用于复用 `maoxiandao` 的旧版怪物模型生成新版 YOLO 数据集。旧模型的类别 ID 和类别名称会原样保留，不会把所有怪物合并成一个类别。

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
  --images data\images `
  --old-model ..\maoxiandao\bot\resource\bundles\models\best.pt `
  --output data\old_yolo_dataset `
  --review-output data\old_yolo_review
```

输出数据集会按固定随机种子拆分为 `train` 和 `val`，原图片会复制到数据集目录；标签是标准 YOLO 格式，类别 ID 与旧模型完全一致。复核目录包含带检测框的完整图片，以及按类别分目录保存的检测框裁剪图。

如果某个地图只有 `9: lvwoniu`，可以快速清理其他错误类别并审核类别 9：

```powershell
python tools/yolo_data.py review-old `
  --dataset data\old_yolo_dataset `
  --class-id 9
```

该命令会先自动删除所有非类别 9 标签，然后从每个检测框生成单独的裁剪图并逐张显示。按 `k`/`n`/空格表示正确并保留，按 `d` 表示错误：程序会删除该裁剪图，并同步删除原标签中的对应框；`p` 上一张，`q` 退出。需要补画漏检框时，再使用 `annotate` 工具处理原图。

也可以完全不用审核窗口：直接在 `data\old_yolo_review\crops\9_lvwoniu_review` 中删除你认为错误的 `.jpg`，保留正确图片，然后执行：

```powershell
python tools/yolo_data.py apply-crop-review `
  --dataset data\old_yolo_dataset `
  --crops data\old_yolo_review\crops
```

程序会递归遍历所有类别文件夹，根据各目录中的 `manifest.json` 将被删除的裁剪图对应检测框从 YOLO 标签中删除，原始游戏图片不会删除。也可以增加 `--class-id 9` 只处理一个类别。

审核完成后，先将类别转换为“玩家 0、怪物 1–17”：

```powershell
python tools/yolo_data.py prepare-player-dataset --dataset data\old_yolo_dataset
```

## 2. 标注玩家

```powershell
python tools/yolo_data.py annotate --dataset datasets\monster_player --split train
python tools/yolo_data.py annotate --dataset datasets\monster_player --split val
```

准备完成后，玩家类别为 `0: player`，旧模型真实怪物为 `1–17`：

```powershell
python tools/yolo_data.py annotate --dataset data\old_yolo_dataset --split train
python tools/yolo_data.py annotate --dataset data\old_yolo_dataset --split val
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
