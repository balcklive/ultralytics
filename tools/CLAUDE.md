# tools/

本目录存放项目专用的数据准备、标注、训练辅助脚本，供本地数据工作流使用，不参与 Ultralytics 包本体逻辑。

> 自动标注全流程（录像→YOLO标签→VLM复核→回填）的架构、实测数据与下一步计划见
> `docs/auto_labeling_pipeline.md`。

## 文件清单

| 文件 | 说明 |
| --- | --- |
| `yolo_data.py` | 核心工具：`auto-label`（旧模型自动标注）、`build-73-dataset`（构建 73 类数据集并继承标签）、`merge-dataset`（合并数据集，`--overwrite` 覆盖已有标签）、`apply-crop-review`（回填删除的裁剪图）、`review-old`（交互式裁剪图审核）、`annotate`（OpenCV 标注器）、`train`（启动 YOLO 训练） |
| `merge_maps.py` | 把各图/各源数据集合并成统一训练集：逐源校验 `names`==主表（`--master`，默认 `data/wgc_review/dataset.yaml`；不一致则拒绝除非 `--map-source` 重映射，防类别错位）、内容哈希去重（去重复帧）、按类保证 val 覆盖稀有类的重切 train/val、写不含 `path` 的 dataset.yaml（本地/云通用）。`--dry-run` 只打印计数/直方图/覆盖情况 |
| `auto_label.py` | 自动标注管线（录像/图片 → YOLO 标签）：`build-templates` 从已标注数据构建怪物模板库（`data/templates/`，含 classes.json，各数据集 id 用 offset 对齐模型空间）、`run` 执行"YOLO 候选 → 中心裁剪模板分类 → 双源决策 → VLM 图鉴仲裁"分层标注（默认模型 `weights/20260823/best.pt`；`--map-classes` 传当前地图怪物白名单（来自 bot-cs `data/maps/names.json`），清单外的框即使高置信也降级送 VLM，防 UI 按钮/杂框误入库；`--vlm-recheck-classes` 传易混类（换色家族/绿色背景易误类如绿水灵），这些类即使 YOLO 高置信也不直采、降级送 VLM 迷你图鉴复核——实测高置信直采的绿水灵框 97% 实为静态花草（结论 14）；`--static-thresh` 启用静态背景剔除（仅视频源）：滚动背景中值模型（`_BgModel`，复用 frame_review 的运动热图思路）算框内运动量，低于阈值的静态花草/UI 框不直采、送 VLM；VLM 仲裁=按帧批量调 Codex 后端 gpt-5.6-luna 对照 monster_instances 图鉴"以图搜图"（禁止纯文字描述推理），3 次重试；输出 labels/ 与 review/ 待复核清单，支持视频抽帧+均值哈希近重复帧剔除）、`export-review` 把 review/*.json 转成整帧可视化 review_vis/（绿=已入库，橙=待复核）与按类裁剪 crops/（拼音目录名）、`vlm-review` 用 monster_instances 实例图+玩家图拼编号图鉴，经 Codex 后端 GPT 视觉自动裁决 crops（`--classes` 传地图白名单生成迷你图鉴，干扰格少准确率高——全 73 格图鉴会系统性错配如红蜗牛→火野猪；匹配→移动改判，不匹配→crops_rejected/，日志 vlm_review_log.jsonl；图鉴/拼图/prompt 与 run 阶段共用 `load_atlas`/`_sheet_b64`/`PROMPT`）、`refine-crops` 域内种子 NCC 重分类 crops（**仅适合同簇去杂**：本游戏怪物大量调色板换色，灰度 NCC 分不开蓝/红蜗牛（模板相似 0.92），换色家族归类须用 vlm-review 迷你图鉴）、`apply-review` 把整理后的 crops/ 裁决回填 labels/（删=拒绝、保留=采纳、移动目录=改判，自动备份 labels） |
| `review_pack.py` | 生成人工复核材料包：`--src`（auto_label run 输出）`--out`（复核数据集），产出标准数据集结构（images/train+labels/train+dataset.yaml，可直接喂 `yolo_data.py annotate`）、每类 crop 文件夹与编号拼图板（crops/ crops_vis/，index.json 可追溯格子→帧+框）、整帧全类别彩框可视化（labels_vis/）；人工改完 labels 重跑一遍即刷新全部材料 |
| `crop_review.py` | 交互式 crop 类别复核器：在 review_pack 生成的每类拼图板上点选错标格子——点格子选中、点右侧类名两下确认改判（移动 crop+改写 label）、右键/d 标记误检（crop 入 crops_bad/ + 删 label 行）、u 撤销；每次操作实时回写 labels/，改完重跑 review_pack 刷新派生材料 |
| `frame_review.py` | 整帧人工查漏复核器：逐帧浏览标注结果，左键拖拽补画漏检框（类别由右侧面板选定）、点已有框弹菜单改类/删除、滚轮缩放+中键/方向键平移、m 叠加运动热图（|帧-背景中值|，定位怪物活动区）；所有操作实时回写 labels，撤销栈，n/p 切帧。**右侧类别面板支持 `--classes` 过滤**（逗号分隔中文名或 id，自动含 player）：只显示本地图怪物，避免在 73 类里翻找（如 `--classes "猴子,火野猪,黑斧木妖"`）；滚轮在面板内滚动类别列表（73 类超出可视高度），画布上滚轮仍是缩放 |
| `template_match.py` | 模板匹配检测/评测工具：从标注数据裁模板、全图多模板匹配出框、`eval` 模式对比人工标签算 IoU/P/R（2026-08-30 实测：全图搜索在复杂背景下 FP 泛滥，已被 auto_label.py 的"框内分类"用法取代，此文件保留作基准测试） |
| `vlm_classify.py` | 批量调用豆包 VL（via `~/.claude/skills/vlm-image`）对裁剪图做 73 类判断并统计 top-1 准确率，JSONL 断点续跑（实测 73 类直判仅 52.5%，只用作仲裁而非主力分类；2026-08-31 起 VLM 统一走 Codex 后端，豆包仅留作基准实验） |
| `export_map_dataset.py` | 导出 mxdzlk 地图 YOLO 数据集（含怪物精灵合成，合成图天然带精确标签） |
| `export_onnx.py` | 将模型导出为 ONNX |
| `predict_video.py` | 视频逐帧推理并输出带框视频 + 耗时统计 |
| `test_codex_sdk.py` | Codex SDK（`openai-codex` 包）集成测试：thread_start + 逐轮 sandbox 切换；2026-09-02 自 `artifacts/` 迁入 |
| `test_openai_sdk.py` | 用 openai SDK 裸调 Codex 后端（`https://chatgpt.com/backend-api/codex`）：流式 Responses API、手动重发历史多轮对话；2026-09-02 自 `artifacts/` 迁入 |
| `test_openai_vision.py` | `gpt-5.6-luna` 视觉输入参考实现（`auto_label.py` 的 `_codex_vlm()` 由此改来）；2026-09-02 自 `artifacts/` 迁入 |

## 调用链

- 被调用方：命令行直接运行 `python tools/yolo_data.py <命令>` / `python tools/merge_maps.py <命令>`；`yolo_data.py` 内部调用 Ultralytics 的 `YOLO` 训练/推理 API；`merge_maps.py` 独立自足（仅依赖 yaml/stdlib）。
- 数据目录约定：训练数据在 `data/`（已 gitignore）；`datasets/` 仅作为类别清单参考，不参与训练。本地按图分夹用 `data/maps/<地图>/`（每夹一个 73 类表 dataset.yaml），最终统一训练用 `merge_maps.py` 合并为 `data/unified_<round>/`。
- 类别体系：主数据集为 73 类（`0: player` + 72 怪物，中文名）；类别名单取自 `datasets/mxdzlk_cmsc_player_mirror/dataset.yaml`。多源合并前**必须先**把各源 id 空间对齐到该主表。

## 关键规则 / 注意事项

- 标注器 `annotate` 特性：中文类别名与中文文件名通过 `read_image`/`write_image`（`imdecode`/`imencode` 兜底）支持；右侧常驻类别面板；画框后直接输数字定类别；拖四角实时缩放；`g`+数字跳转。
- 自动标注分域实测（2026-08-30）：合成图（`datasets/mxdzlk_cmsc`，网站 icon 合成、框为统一 42x42 标记框）上中心裁剪模板分类 top-1 97.1%；真实游戏帧（`data/annotated_73`，JPG 压缩+动画帧）上模板匹配失效（~40%），YOLO 模型反而好（conf=0.4 端到端 P=77.2%/类准确 96.8%）。故 `auto_label.py` 用双源决策：模板高分直采、双源一致直采、其余交 VLM 仲裁并落 review/。
- 模板必须与目标数据**同域**（同一渲染来源），跨域（网站 icon vs 游戏内渲染）精灵外观差异大；新域数据首次仍需少量人工/审核冷启动。
- Windows 下 `cv2.imread` 不支持中文路径，一律用 `auto_label.imread_u`（`np.fromfile`+`imdecode`）。
- 展示环节（图上文字、目录/文件名）一律用拼音（`auto_label.py` 的 `py()`，基于 pypinyin）：`cv2.putText` 不支持中文、终端/看图环节中文易乱码；中文仅保留在 json 数据与 dataset.yaml 中。
- 类别 id 空间：`datasets/mxdzlk_cmsc/dataset.yaml`（0=特殊小石球）与 `data/annotated_73/dataset.yaml`（0=player，怪物 +1）不一致，构建模板库时用 offset 对齐到模型空间。
- 数据安全：`auto-label --overwrite`、`apply-crop-review`、`build-73-dataset --overwrite`、`merge-dataset --overwrite`、`annotate` 都会修改标签，执行前确认路径并建议备份。
- 用 ruff 检查代码风格（`uvx ruff check tools/yolo_data.py tools/merge_maps.py`）。
- 新增子命令时在 `main()` 用 `add_parser` 注册，并给函数加 `args` 参数、保持 argparse 风格。
- `merge_maps.py` 的输出 dataset.yaml 故意**不含 `path` 键**（本地与 PAI-DLC 容器均以 yaml 目录为根，见 `cloud/dlc/prepare.py` 说明），可直接上传 OSS 训练。类别表不一致的源默认被拒绝（防静默错位），需人工提供 `--map-source` 映射表才可并入。
