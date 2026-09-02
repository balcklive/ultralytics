# 新增图片标注校核与训练工作流

本文档记录"新增一批图片 → 旧模型自动识别 → 目视复核删错框 → 回填 → 手动补框 → 合并 → 重新训练"的完整流程，以及其中使用的命令和标注器操作。全部命令通过 `tools/yolo_data.py` 执行。

## 类别体系

- 主数据集使用 **73 类**：`0: player` + `1–72` 怪物（中文名，如 `2 蜗牛`、`8 绿水灵`、`57 火野猪`）。
- 类别名单来源：`datasets/mxdzlk_cmsc_player_mirror/dataset.yaml`（仅作为类别清单引用，不参与训练）。
- 训练只用 `data/` 下的数据集，`datasets/` 不纳入训练。

## 流程总览

```text
① 准备 73 类数据集（首次）        build-73-dataset
② 旧模型自动识别新图片            auto-label
     └─ 结果图 + 目标裁剪图 + manifest.json 存入 review 目录
③ 目视复核，删除错误截图          手动删 crops 里的 .jpg
④ 回填删除到标签                  apply-crop-review
⑤ 手动补填缺失框 / 修改框         annotate
⑥ 合并精修结果到主数据集          merge-dataset --overwrite
⑦ 用原权重重新训练                train
```

---

## ① 准备 73 类数据集（首次 / 重建）

从图片目录构建带 73 类名单的数据集，并可继承旧数据集的标签：

```powershell
.venv\Scripts\python.exe tools/yolo_data.py build-73-dataset `
  --output data/annotated_73 `
  --images data/images data/new_images `
  --names-yaml datasets/mxdzlk_cmsc_player_mirror/dataset.yaml `
  --inherit-player data/old_yolo_dataset data/new_yolo_dataset `
  --inherit-monster 9:2 `
  --overwrite
```

参数：
- `--images`：要纳入的图片目录（可多个）。
- `--names-yaml`：提供完整类别名单的 `dataset.yaml`。
- `--inherit-player`：从这些数据集继承类别 `0`（player）框。
- `--inherit-monster OLD:NEW`：把旧数据集里旧类别 `OLD` 的框按新类别 `NEW` 继承（可重复，如 `9:2` 表示旧 `lvwoniu` → 新 `蜗牛`）。
- `--overwrite`：重建前清空输出目录（会丢掉已有手动标注，谨慎使用）。

> 首次构建不需要 `--overwrite`（输出目录为空）；后续重建旧数据会清掉该目录里的手动标注。

---

## ② 旧模型自动识别新图片

```powershell
.venv\Scripts\python.exe tools/yolo_data.py auto-label `
  --images data/new_images `
  --old-model weights/20260823/best.pt `
  --output data/new_yolo_73 `
  --review-output data/new_yolo_73_review `
  --conf 0.25
```

产出：
- 数据集 `data/new_yolo_73`：图片 + YOLO 标签 + `dataset.yaml`。
- 复核目录 `data/new_yolo_73_review/`：
  - 带框的完整结果图；
  - 每个目标按类别分目录的**裁剪图**（如 `crops/8_绿水灵/xxx.jpg`）；
  - `crops/manifest.json`，记录每个裁剪图对应的标签文件、类别、框坐标。

> 用旧模型识别时类别 ID 会原样保留，所以建议直接用与目标类别体系一致的模型（如 73 类模型）。

---

## ③ 目视复核，删除错误截图

打开 `data/new_yolo_73_review/crops/<类别目录>/`，**只删除**识别错误的 `.jpg` 裁剪图。

注意：
- 只删图片，**不要删** `manifest.json` 和类别文件夹。
- 删除仅表示"这个框不对"，标签尚未改动，需执行下一步回填。

---

## ④ 回填删除到标签

```powershell
.venv\Scripts\python.exe tools/yolo_data.py apply-crop-review `
  --dataset data/new_yolo_73 `
  --crops data/new_yolo_73_review/crops
```

读取 `manifest.json`：凡裁剪图已被删除的，对应框从标签中同步删除；保留的框保留。结束时打印 `kept X boxes, removed Y boxes`。

- 某张图所有框都被删光时，标签文件变空，训练时作为背景/负样本。
- 可选 `--class-id N` 只处理某一类别。

---

## ⑤ 手动补填缺失框 / 修改框

```powershell
.venv\Scripts\python.exe tools/yolo_data.py annotate --dataset data/new_yolo_73 --split train
.venv\Scripts\python.exe tools/yolo_data.py annotate --dataset data/new_yolo_73 --split val
```

### 标注器操作

界面：左侧图片（自动缩放），右侧**常驻类别面板**（`编号 类别名`，当前类别蓝色高亮，可点击/滚轮滚动）。

| 操作 | 说明 |
| --- | --- |
| 左键空白拖拽 | 画新框（画完直接输数字即可定类别） |
| **画框后直接输数字** | 给刚画的框输入怪物编号，`Enter` 确认定类（`Esc` 取消，`Backspace` 退格） |
| 点击面板某行 | 切换到该类别 |
| 点击已有框内部 | 选中该框，再输数字可改其类别（黄色高亮） |
| 拖框四角 | 实时缩放（放大/缩小） |
| 右键点击框 | 删除该框 |
| `g` + 数字 + `Enter` | 跳转当前类别 |
| `[` / `]` | 循环切换类别 |
| `c` | 清空当前类别的所有框 |
| `s` | 保存当前图 |
| `n` / 空格 | 保存并下一张 |
| `p` | 保存并上一张 |
| `q` | 保存并退出 |

- 中文类别名正常显示（PIL + 系统中文字体），中文文件名图片也能正常读写。
- 状态栏显示当前图、当前类别、框数、数字输入预览和按键提示。

---

## ⑥ 合并精修结果到主数据集

精修完成后，把新图的标签合并进主训练集：

```powershell
.venv\Scripts\python.exe tools/yolo_data.py merge-dataset `
  --target data/annotated_73 `
  --source data/new_yolo_73 `
  --overwrite
```

- `--overwrite`：**覆盖**目标中已存在图片的标签（新图标签被精修结果替换）；不加则跳过已存在图片。
- 类别名不一致会拒绝合并并列出差异。
- 目标中不存在的图片会被复制进来（图片 + 标签）。

> 合并前建议备份标签：
> ```powershell
> $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
> Copy-Item data/annotated_73/labels data/backups/labels_$stamp -Recurse
> ```

---

## ⑦ 用原权重重新训练

```powershell
.venv\Scripts\python.exe tools/yolo_data.py train `
  --dataset data/annotated_73 `
  --model models/yolo26s.pt `
  --epochs 100 `
  --imgsz 640 `
  --device cpu
```

- 73 类训练需从基础模型（`models/yolo26s.pt`）或已训练的 73 类权重起步；旧的 18 类 `weights/best.pt` 不适用。
- 训练输出默认在 `runs/train/monster-player/weights/`。

---

## 数据安全

- `auto-label --overwrite`、`apply-crop-review`、`build-73-dataset --overwrite`、`merge-dataset --overwrite`、`annotate` 都会改动标签；执行前确认路径。
- 旧数据集（`data/old_yolo_dataset`、`data/new_yolo_dataset`）与 `datasets/` 是参考/历史数据，不会在本流程中被修改或删除。
- 建议保留标签备份（见步骤 ⑥）。
