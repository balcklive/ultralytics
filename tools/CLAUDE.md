# tools/

本目录存放项目专用的数据准备、标注、训练辅助脚本，供本地数据工作流使用，不参与 Ultralytics 包本体逻辑。

## 文件清单

| 文件 | 说明 |
| --- | --- |
| `yolo_data.py` | 核心工具：`auto-label`（旧模型自动标注）、`build-73-dataset`（构建 73 类数据集并继承标签）、`merge-dataset`（合并数据集，`--overwrite` 覆盖已有标签）、`apply-crop-review`（回填删除的裁剪图）、`review-old`（交互式裁剪图审核）、`annotate`（OpenCV 标注器）、`train`（启动 YOLO 训练） |
| `export_map_dataset.py` | 导出 mxdzlk 地图 YOLO 数据集 |
| `export_onnx.py` | 将模型导出为 ONNX |
| `predict_video.py` | 视频逐帧推理并输出带框视频 + 耗时统计 |

## 调用链

- 被调用方：命令行直接运行 `python tools/yolo_data.py <命令>`；`yolo_data.py` 内部调用 Ultralytics 的 `YOLO` 训练/推理 API。
- 数据目录约定：训练数据在 `data/`（已 gitignore）；`datasets/` 仅作为类别清单参考，不参与训练。
- 类别体系：主数据集为 73 类（`0: player` + 72 怪物，中文名）；类别名单取自 `datasets/mxdzlk_cmsc_player_mirror/dataset.yaml`。

## 关键规则 / 注意事项

- 标注器 `annotate` 特性：中文类别名与中文文件名通过 `read_image`/`write_image`（`imdecode`/`imencode` 兜底）支持；右侧常驻类别面板；画框后直接输数字定类别；拖四角实时缩放；`g`+数字跳转。
- 数据安全：`auto-label --overwrite`、`apply-crop-review`、`build-73-dataset --overwrite`、`merge-dataset --overwrite`、`annotate` 都会修改标签，执行前确认路径并建议备份。
- 用 ruff 检查代码风格（`uvx ruff check tools/yolo_data.py`）。
- 新增子命令时在 `main()` 用 `add_parser` 注册，并给函数加 `args` 参数、保持 argparse 风格。
