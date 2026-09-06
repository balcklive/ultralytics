# 自动标注管线：从游戏录像到 YOLO 训练数据（工作文档）

> 目标：用「YOLO 模型 + 模板匹配 + VLM（视觉大模型）」把人工标注工作量压缩到
> 两轮快速人工复核，让"录像 → 自动标注 → 人工复核 → 人工启动训练"形成自迭代飞轮。
> 本文档记录 2026-08-30 ~ 08-31 全部实测结论、代码位置与完整 SOP。
> **飞轮第一圈已完整跑通**（wgc 录像 → v1 模型 mAP50 0.788）。

---

## 1. 一句话架构

```
录像/图片 → YOLO 出候选框（地图白名单约束）
         → 中心裁剪模板匹配分类 + VLM 图鉴仲裁（Codex 以图搜图，按帧批量）
         → 分层决策（自动采信 / 进 review）
         → export-review + apply-review（VLM 迷你图鉴裁决回填）
         → review_pack 生成人工复核材料
         → 人工复核 A：crop_review 拼图板核对类别（点选改判/标废）
         → 人工复核 B：frame_review 整帧查漏（补框/删框，运动热图辅助）
         → 人工启动训练 yolo_data.py train → 新模型投入下一卷录像
```

**核心原则（实测校准）：** 每个组件只干它擅长的事——

| 组件 | 擅长 | 不擅长（实测数据） |
| --- | --- | --- |
| YOLO 模型（域匹配） | 定位 + 真实帧分类（20260823 每帧 16~21 框） | 域外整类归零；会把 UI 按钮判成怪 → 必须配地图白名单 |
| 中心裁剪模板匹配 | **同域**像素级分类（97.1%） | 跨域失配、全图搜索 FP 泛滥（P=0.045）、调色板换色家族分不开 |
| VLM（Codex 图鉴） | 迷你图鉴以图搜图（85%）、杂框拒绝 | 全 73 格图鉴会系统性错配（红蜗牛→火野猪 conf0.86）；纯文字推理 52.5% |
| 人工 | 两轮复核（类别板 + 整帧查漏），最终把关 | —— |

---

## 2. 关键实测结论（2026-08-30 ~ 08-31，勿重复踩坑）

1. **VLM 纯文字描述直判 73 类只有 52.5% top-1**（122 样本，`artifacts/vlm_eval.jsonl`）。
   错误系统性的：蜗牛→蓝蜗牛、石膏士官→石膏士兵等同色系相近类，常以 0.95+ 置信度自信地错。
   **结论：VLM 只能以图搜图（对照 monster_instances 实例图鉴），禁止纯文字描述推理。**
2. **中心裁剪模板匹配在合成图域 top-1 97.1% / top-3 99.6%**
   （正确分均值 0.961 vs 错误 0.815，阈值 0.93 直采、0.85 以下丢弃有清晰分界）。
3. **跨域失配是真坑**：`datasets/mxdzlk_cmsc`（网站 icon 合成、42x42 统一标记框）与真实游戏帧
   （`data/annotated_73`）里同名怪物渲染差异大，模板分数 0.96 → 0.5。**模板必须与目标数据同域**。
4. **模型域匹配决定一切**：旧模型 monster-player-2（只在 annotated_73 真实帧训练）在新地图录像上
   花蘑菇**整类检出归零**、置信普遍 <0.5；换 `weights/20260823/best.pt`
   （cmsc_player_mirror_two_maps 域训练，YOLO26n）后每帧自动入库 16~21 框、类别多样。
   但它会把**底部 UI 按钮**（自动/技能栏）以 0.5+ 置信度判成"冰龙"→ 必须配地图白名单。
5. **推理尺度必须匹配训练尺度**：模型 imgsz=640 训练；真实帧用 imgsz=1344 推理 → 0 命中，
   用 640 正常。`run` 时默认传 `--imgsz 640`。
6. **全图模板搜索不可用**（复杂背景 FP 泛滥），模板只用于"框内分类"。
7. **双源决策的教训**："模板不认就送 VLM"会把 YOLO 0.86 的正确判断改错——
   高置信信号要直接采信，不能被弱信号否决。
8. **纯自动采信层的数据质量**（val 帧）：精确率 94.4%、类别正确率 100%、召回 32.1%。
   召回靠 review 闭环 + 人工查漏补。
9. **VLM 图鉴匹配有系统性错法**：全 73 格图鉴下红蜗牛→火野猪（conf 0.86 也错）、绿水灵→青螃蟹。
   **解法：`vlm-review --classes` 地图白名单迷你图鉴**（8 格实测 85%，
   huoyezhu/qingpangxie 错簇 59 张全数拉回）。crops 目录里出现地图不可能有的怪 = 这类错判。
10. **地图→怪物出没表来自 bot-cs**：`D:/07-games/bot-cs/data/maps/names.json`
    （约 90 张地图，含 mobId/中文名/刷新数）。wgc 录像地图=射手训练场Ⅰ（map-104040000）：
    蓝蜗牛、蘑菇仔、木妖、红蜗牛、绿水灵、花蘑菇、绿蘑菇。
11. **游戏怪物大量调色板换色**：蓝蜗牛/红蜗牛、蘑菇仔/花蘑菇/绿蘑菇是同型换色——灰度 NCC
    模板相似 0.92，彩色通道均值 NCC 也 0.82。经典模板匹配对换色家族失效
    （`refine-crops` 只能做同簇去杂），换色家族归类必须靠 VLM 迷你图鉴。
12. **人工复核拦截了真问题**：自动+VLM 产出 1250 框，人工两轮复核后定稿 1141 框
    （删 163 个绿水灵误框、清零木妖/绿蘑菇误检、补 26 player 框等）。
13. **VLM 只处理低置信疑难框，高置信直采路径绕过了 VLM**（2026-09-01 复盘）：
    最终 labels 447 个绿水灵框里 432 个（97%）是 YOLO 高置信（`>=high-conf 0.5`）
    且类在白名单内直接采信，从未进 VLM；送 VLM 的绿水灵框仅 13 个且 yolo_conf 中位
    0.293。蓝蜗牛 392/419（94%）、红蜗牛 193/212（91%）同样直采绕过 VLM。
14. **高置信直采的绿水灵框 97% 是静态花草/UI**（2026-09-01 实测）：抽 27 个
    `auto_yolo_high` 直采绿水灵裁剪图让 VLM 判读，约 44% 实为三叶草丛/灌木/UI 入口，
    其余才是真怪物——绿色球形团块与绿水灵（绿色果冻状）在 YOLO 眼里高置信相似。
    蜗牛颜色误判量级轻得多（直采蓝蜗牛框中心色：蓝 373/红 11/混合 5，蓝标红仅 ~3%），
    同样由高置信直采产生。**结论：直采路径（方案 1+3 加固）才是主战场，VLM 只兜底
    低置信疑难区。**

---

## 3. 源码位置

| 文件 | 职责 |
| --- | --- |
| `tools/auto_label.py` | **核心管线**：`build-templates`（建模板库）、`run`（自动标注，默认模型 `weights/20260823/best.pt`，`--map-classes` 白名单，VLM 按帧批量图鉴仲裁+3 次重试）、`export-review`（review JSON → 可视化+按类裁剪）、`vlm-review`（VLM 迷你图鉴裁决 crops，`--classes` 传地图白名单）、`refine-crops`（域内种子 NCC 精校，仅适合同簇去杂，见结论 11）、`apply-review`（裁决回填 labels） |
| `tools/review_pack.py` | 生成人工复核材料包：标准数据集结构（images/labels train+val + dataset.yaml）、每类 crop 文件夹与编号拼图板、整帧全类别彩框可视化；人工改完重跑即刷新 |
| `tools/crop_review.py` | 交互式 crop 类别复核器：拼图板上点格子→弹菜单改判/标废，实时回写 labels，可撤销 |
| `tools/frame_review.py` | 整帧人工查漏复核器：滚轮缩放浏览、拖拽补框、点框改类/删框、**m 键运动热图**（帧-背景中值差分，定位怪物活动区找漏检），实时回写 labels |
| `tools/yolo_data.py` | 数据集构建/合并/审核/训练：`merge-dataset`（下一圈并入）、`annotate`（OpenCV 整帧标注器）、`train`（启动 YOLO 训练） |
| `tools/template_match.py` | 模板匹配基准测试（全图搜索已弃用，P=0.045 证据） |
| `tools/vlm_classify.py` | 豆包 VL 73 类直判实验工具（52.5% 证据；VLM 已统一走 Codex） |
| `tools/test_openai_vision.py` | Codex 后端（gpt-5.6-luna）视觉调用参考实现；`auto_label.py` 的 `_codex_vlm()` 由此改来（原在 `artifacts/`，2026-09-02 迁至 `tools/`） |

依赖：`pypinyin`（已 `uv add`）；GPT 视觉走 `~/.codex/auth.json` + 本地代理 127.0.0.1:7890
（带 3 次网络重试）；豆包 VL 仅 `vlm_classify.py` 留作基准。

### Windows 注意事项（代码已内置处理）

- `cv2.imread` 不支持中文路径 → 统一用 `auto_label.imread_u()`（np.fromfile + imdecode）。
- `cv2.putText`/文件系统展示不支持中文 → 展示环节一律拼音（`py()`，pypinyin 无下划线）；
  `monster_instances` 目录是下划线拼音，两者用 `us2id` 映射转换。
- 标签行有 CRLF → 读标签用 strip 容错。

---

## 4. 中间数据位置

| 路径 | 内容 | 生成者 |
| --- | --- | --- |
| `data/templates/` | 模板库：73 类 552 个模板 + `classes.json` + `atlas.png`（73 格全图鉴）+ `atlas_mini.png`（地图白名单迷你图鉴） | `build-templates` / `vlm-review --classes` |
| `datasets/mxdzlk_cmsc/monster_instances/<下划线拼音>/` | 每类 1 张 96x96 实例图（72 类，图鉴源） | 用户 |
| `datasets/player/magic.png` | 玩家实例图（图鉴最后一格） | 用户 |
| `artifacts/auto_label_record/` | **wgc 批次主产物（训练数据源，保留）**：`images/`（77 帧）、`labels/`（1141 框定稿标签）、`review/`+`review_applied/`（待复核/已归档 JSON）、`review_vis/`、`crops/`+`crops_rejected/`（VLM 裁决留痕）、`vlm_review_log.jsonl`+`refine_log.jsonl`（审计日志）、`labels_backup-*/`（回填前备份） | 管线全流程 |
| `data/wgc_review/` | **训练就绪数据集（保留）**：images/labels train 67 帧 990 框 + val 10 帧 151 框 + dataset.yaml（73 类 id 空间） | `review_pack.py` + 人工复核 |
| `runs/train/wgc-review-v1/` | 飞轮 v1 模型（从 20260823 微调，YOLO26n，50ep，CPU） | 用户 `train` |
| `artifacts/auto_label_val/` `auto_label_out/` | 早期管线验证产物 | `run` |
| `artifacts/vlm_eval.jsonl` `template_eval.json` | 关键实验证据（VLM 52.5%、全图模板 P=0.045） | 实验工具 |
| `data/record/wgc-20260830-184840.mp4` | 本批源录像：1368x800@30fps、60s | 用户 |

> 2026-08-31 清理：探索期临时图片、旧管线整批产物 `auto_label_record_old/`（含蜗牛纠正
> 前备份，纠正结论见 6-P0）、`data/wgc_review` 派生复核材料（crops/crops_vis/labels_vis，
> 可由 review_pack 随时重新生成）均已删除；训练数据与审计日志全部保留。

类别 id 空间约定：**模型/模板库/管线输出都用 `data/annotated_73/dataset.yaml` 空间**
（0=player，1=特殊小石球 … 72=乌龟）；`datasets/mxdzlk_cmsc/dataset.yaml` 无 player（怪物 id 比
annotated_73 小 1），构建模板时用 offset=+1 对齐。

---

## 5. 完整流程 SOP（从输入录像到下一版模型，飞轮一圈）

以下为 wgc 批次（2026-08-31）实测跑通的完整流程，新录像按此复制。

### 第 1 步：准备（一次性）

```bash
uv run python tools/auto_label.py build-templates   # 模板库（数据有更新时重跑）
```

- 模型：当前用 `weights/20260823/best.pt`（之后每圈换成上一圈产物）
- 地图白名单：查 bot-cs `data/maps/names.json`（如射手训练场Ⅰ=map-104040000）

### 第 2 步：录像自动标注

```bash
uv run python tools/auto_label.py run \
    --source data/record/wgc-20260830-184840.mp4 \
    --out artifacts/auto_label_record \
    --imgsz 640 --fps 2 --high-conf 0.5 \
    --map-classes "蓝蜗牛,蘑菇仔,木妖,红蜗牛,绿水灵,花蘑菇,绿蘑菇" \
    --vlm-recheck-classes "蓝蜗牛,红蜗牛,绿水灵" \
    --static-thresh 3
```

- `--vlm-recheck-classes`（方案1）：换色家族（蓝/红蜗牛、蘑菇仔/花/绿蘑菇）与绿色
  背景易误类（绿水灵）即使 YOLO 高置信也降级送 VLM 复核——实测 97% 高置信直采
  绿水灵框是静态花草（结论 14）。
- `--static-thresh`（方案3）：静态背景剔除（仅视频源）。滚动背景中值模型算框内
  运动量，低于阈值的静态花草/UI 框不直采、送 VLM。推荐从 3~5 起步，看 review 量
  与误检平衡再调。新录像复跑时沿用。

决策级联（顺序不能换）：
1. `yolo_conf >= high-conf(0.5)` 且类在地图白名单 **且非易混类且非静态** → 直接采信
2. 模板 NCC `>= 0.93` 且类在白名单 **且非易混类且非静态** → 直接采信
3. YOLO 类别 == 模板 top1 且 `yolo_conf >= 0.35` 且在白名单 **且非易混类且非静态** → 采信
4. `yolo_conf >= 0.25` → VLM 图鉴仲裁（只落 review/，默认不写标签）
5. 其余丢弃

> **2026-09-01 加固（方案1+3）**：`--vlm-recheck-classes`（易混类高置信降级）与
> `--static-thresh`（静态背景剔除）把直采路径限定在"非易混类且非静态"的框上，
> 实测高置信直采的绿水灵框 97% 实为静态花草（结论 14），这两项把这类误框
> 从直采改为进 review 由 VLM 迷你图鉴裁决。

实测（新管线，60s → 77 帧）：自动入库 937 框、VLM 仲裁 330、拒绝杂框 560。
（旧管线 monster-player-2+豆包 同源录像仅 96+512 框，作对照存档。）

### 第 3 步：导出 + VLM 迷你图鉴裁决 + 回填

```bash
uv run python tools/auto_label.py export-review --out artifacts/auto_label_record
uv run python tools/auto_label.py vlm-review --out artifacts/auto_label_record \
    --batch 10 --conf 0.6 \
    --classes "蓝蜗牛,红蜗牛,绿水灵,蘑菇仔,花蘑菇,绿蘑菇,木妖"
uv run python tools/auto_label.py apply-review --out artifacts/auto_label_record
```

- `--classes` = 地图白名单 → 迷你图鉴（只含这些类+player）。实测 886 张 crops：
  保留 189、改判 123（huoyezhu→hongwoniu 42、qingpangxie→lvshuiling 17 等错簇全数拉回）、
  拒绝 574，约 45 分钟。
- `apply-review`：crops/ 里存在的文件按所在目录类采纳并入 labels/，不在=拒绝；
  自动备份 labels/ 到 `labels_backup-<时间戳>/`，review JSON 归档 `review_applied/`。

### 第 4 步：生成人工复核材料包

```bash
uv run python tools/review_pack.py --src artifacts/auto_label_record --out data/wgc_review
```

产出：标准数据集结构（dataset.yaml，73 类 id 空间）+ `crops/<拼音类>/` 每类文件夹 +
`crops_vis/<类>.png` 编号拼图板（`index.json` 可追溯格子→帧+框）+ `labels_vis/` 整帧彩框图。

### 第 5 步：人工复核 A —— crop 类别核对（tools/crop_review.py）

```bash
uv run python tools/crop_review.py --root data/wgc_review
```

- 看每类拼图板：**左键点格子** → 弹出菜单 → 点正确类别（改判）或 `X BAD`（非怪物）；
  右键快捷标废；`u` 撤销；顶部标签页/右侧面板切类；滚轮/滚动条翻页。
- 每次点击实时写盘（crop 移动 + labels 行改写），无需保存，`q` 退出。
- 实测：拦截 163 个绿水灵误框、清零木妖/绿蘑菇误检等。
- 建议顺序：小类（花蘑菇/绿蘑菇/木妖）→ 蘑菇仔（留意混入木妖）→ 三大蜗牛/水灵类抽翻。

### 第 6 步：人工复核 B —— 整帧查漏（tools/frame_review.py）

```bash
uv run python tools/frame_review.py --root data/wgc_review
```

- `n`/`p` 翻帧；**先按 `m` 开运动热图**（红=与静态背景的差异区=怪物活动区，
  无框红区即疑似漏检）；右面板选类别后**左键拖拽补框**；点已有框弹菜单改类/删除；
  滚轮缩放（8x）、中键/方向键平移；`u` 撤销；实时写盘。
- 重点：热图红区无框处、画面边缘刚刷出的怪、被伤害数字/特效遮挡的怪。
- 实测：补 26 player、17 红蜗牛、38 蓝蜗牛等漏检框。

### 第 7 步：刷新复核材料（可选，二次验证）

```bash
uv run python tools/review_pack.py --src artifacts/auto_label_record --out data/wgc_review
# 用定稿 labels 重新生成 crops/拼图板/整帧图，快速过一遍确认无残留错误
```

### 第 8 步：切 val + 人工启动训练

```bash
# 切 val（每第 8 帧进 val；review_pack 会自动归位已切的帧）
# 2026-08-31 实际操作：67 帧 train / 10 帧 val，train 990 框、val 151 框
uv run python tools/yolo_data.py train \
    --dataset data/wgc_review \
    --model weights/20260823/best.pt \
    --epochs 50 --imgsz 640 \
    --name wgc-review-v1 --device cpu
```

**v1 结果（2026-08-31，用户执行）**：`runs/train/wgc-review-v1/weights/best.pt`
（从 20260823 微调，YOLO26n，CPU）。val：**mAP50 0.788** / mAP50-95 0.535；
蓝蜗牛 0.890、绿水灵 0.893、红蜗牛 0.843、蘑菇仔 0.652（val 仅 9 实例，噪声大）、
**player 召回仅 0.40（短板，后续补样本）**。
注意：val 与训练同源录像（近重复帧），指标偏乐观；新域实测后再调阈值。

### 第 9 步：新模型投入下一卷录像（飞轮回转）

```bash
uv run python tools/auto_label.py run --source <新录像> \
    --model runs/train/wgc-review-v1/weights/best.pt \
    --imgsz 640 --fps 2 --high-conf 0.6 \
    --map-classes "<新地图的怪物清单，查 bot-cs names.json>"
```

`--high-conf` 从 0.6 起步观察 review 量，稳定后可上调 0.7。
数据积累后用 `yolo_data.py merge-dataset --target data/wgc_review --source <新批复核集>` 并入再训。

---

## 6. 已知问题（按优先级）

### P0：蜗牛家族标签约定 ✅ 已定夺（2026-08-31）

人工看图裁决（当时留档 `artifacts/snail_simple_choice.png`）：**录像里的蓝壳蜗牛 = 蓝蜗牛**
（annotated_73 id=3）。用户确认该录像地图**只出蓝蜗牛和红蜗牛，永不出绿壳蜗牛**。
cmsc 训练集 GT 无标反（cmsc id 空间 1=蜗牛/2=蓝蜗牛/5=红蜗牛，注意与 annotated_73 差 1）。
结论：YOLO 在旧域把蓝蜗牛系统性误判为蜗牛。旧批次的 85 行 id2→id3 纠正连同旧批次产物
已于 2026-08-31 清理（新管线重跑后该问题不复存在）。

### P1：VLM 图鉴裁决的系统性误判 ✅ 已解决（迷你图鉴）

全 73 格图鉴错配（红蜗牛→火野猪 conf0.86、绿水灵→青螃蟹）由
`vlm-review --classes` 地图白名单迷你图鉴解决（8 格实测 85%，错簇全数拉回）。

### P2：性能（已部分缓解）

旧 40s/帧 → 迷你图鉴按帧批量后约 30~45s/帧（瓶颈仍是 552 模板 x3 尺度 CPU 分类）。
进一步：模板库瘦身到地图白名单类、跟踪器（ByteTrack）合并轨迹复用判定。

### P3：cmsc GT 是 42x42 统一标记框

评测 IoU 会低估（YOLO 输出紧贴框）。彻底解决需要紧贴框 GT 或换评测协议。

### P4：漏检怪物没有任何补充检测（TODO）

当前候选框**唯一来源是 YOLO**（`--conf` 下限 0.05），两层漏检无人处理：
1. `yolo_conf` 0.05–0.25 且没过一致性门的 → 静默丢弃，不进 review；
2. conf < 0.05 或完全无框的 → 连候选都不是（conf=0.05 类别无关召回仅 79.3%）。

人工查漏（SOP 第 6 步）目前靠 `frame_review.py` 运动热图兜底，自动化方向：
- **跟踪器时序回填**：ByteTrack 轨迹丢失帧用预测框回填并触发 review（与 P2 合并做）。
- **帧间运动检测**：差分/背景建模作为第二候选源（frame_review 的热图逻辑可复用）。
- 全图模板搜索补召回已被证伪（P=0.045），不可用。

---

## 7. 路线图

1. ~~域内模板精校~~ ✅ `refine-crops` 已实现，实测对换色家族无效（结论 11），
   保留作同簇去杂工具。
2. ~~蜗牛家族定夺~~ ✅ 见 6-P0。
3. ~~apply-review 回填 + 数据集~~ ✅ 见 SOP 第 3~4 步。
4. ~~v1 训练~~ ✅ mAP50 0.788（SOP 第 8 步）。
5. **飞轮第二圈（火焰之地Ⅴ）** ✅ 见第 9 节：新域（地图 106000140，猴子/火野猪/
   黑斧木妖）无可用模型 → **冷启动工作流**（抽帧人工标注 + 小模型预填 + 人工复核）
   → hyd5-v3（mAP50 0.725）→ merge-dataset 并入主数据集 data/wgc_review（train 101 帧）。
   原数据备份 `data/backup_hyd5_20260902/`。
6. **双域主模型 full-v1** ✅（2026-09-02，见 §9.5）：从 hyd5-v3 续训 80ep，
   mAP50 0.808，player 召回短板已修复（0.40 → 0.864），归档 `weights/20260902/`。
7. **GPU 训练标准化** ✅（2026-09-06）：训练统一走 **AutoDL 实例**（RTX 4080，本机仅 CPU），
   SFTP 传数据+权重 → 远端 py3.8 训练 → 拉回权重 → 本地导出 ONNX；方法见 `docs/autodl_training.md`，
   脚本在 `cloud/autodl/`。（此前 PAI-DLC 路径已弃用删除。）
8. **漏检补充检测自动化**（6-P4）：跟踪器回填 + 运动检测候选源。
9. **性能优化**（需要时）：模板库按地图瘦身、跟踪器轨迹复用（与第 8 项共用改造）。
10. **player 类样本**：v1 短板已随 wgc+hyd5 样本修复，后续新域继续有意识补 player 框。

---

## 8. 快速命令参考（新录像全流程）

```bash
# ① 准备：模型路径 + 地图白名单（bot-cs data/maps/names.json）
uv run python tools/auto_label.py build-templates                  # 数据有更新时

# ② 自动标注（易混类高置信降级 + 静态背景剔除，见结论 13/14）
uv run python tools/auto_label.py run --source <视频> --out <目录> \
    --imgsz 640 --fps 2 --high-conf 0.5 \
    --map-classes "<该地图怪物清单>" \
    --vlm-recheck-classes "<该地图的换色家族/易误类，如 蓝蜗牛,红蜗牛,绿水灵>" \
    --static-thresh 3

# ③ 导出 + VLM 迷你图鉴裁决 + 回填
uv run python tools/auto_label.py export-review --out <目录>
uv run python tools/auto_label.py vlm-review --out <目录> --batch 10 --conf 0.6 \
    --classes "<该地图怪物清单>"
uv run python tools/auto_label.py apply-review --out <目录>

# ④ 人工复核（两轮）
uv run python tools/review_pack.py --src <目录> --out data/<批复核集>
uv run python tools/crop_review.py --root data/<批复核集>          # A：类别核对
uv run python tools/frame_review.py --root data/<批复核集>         # B：整帧查漏(m=热图)

# ⑤ 人工启动训练（切好 val 后）
uv run python tools/yolo_data.py train --dataset data/<批复核集> \
    --model <上一圈best.pt> --epochs 50 --imgsz 640 \
    --name <新版本名> --device cpu
```

---

## 9. 飞轮第二圈：火焰之地Ⅴ 冷启动工作流（2026-09-01 ~ 09-02）

> 与 wgc（第一圈，已有可用模型走全量管线）不同，火焰之地是新地图域，
> **v1 模型在新域几乎整类归零**（猴子 0/黑斧木妖 0/斧木妖 0，仅火野猪 2 个）——
> 重新印证结论 4。此工作流记录"无现成模型域"时的**冷启动 + 主动学习闭环**。

### 9.1 输入与判定

- 录像 `data/record/5396b6345c66e91c4965c5ad946cd88d_raw.mp4`：1368x800@30fps，120s。
- 地图 = **火焰之地Ⅴ（map-106000140）**：bot-cs 出没表 猴子(14)/火野猪(30)/黑斧木妖(16)。
  用 VLM 抽帧判读核对出没表与实拍一致（VLM 多帧判读不稳定，需多次采样交叉验证）。
- 目标类 id：猴子 18 / 火野猪 57 / 黑斧木妖 24 / 斧木妖 16 / player 0（全部在 73 类空间内）。

### 9.2 流程（冷启动 4 轮迭代）

| 轮 | 步骤 | 产物 |
|---|---|---|
| 0 | **VLM 逐帧标注尝试 → 放弃**：VLM 坐标定位不可靠（红框框到背景）、漏检严重（几乎只出火野猪，猴子/黑斧木妖漏检）、误判出没表外类（花蘑菇×8）。**结论：VLM 不能做新域冷启动的自动标注源**，只宜做人工标注的辅助参考 | 弃用 |
| 1 | 抽帧 15 + **frame_review 人工标注**（`--classes "猴子,火野猪,黑斧木妖"` 过滤面板）→ hyd5-v0 训练 | 15 帧 / 53 框，v0 mAP50 0.325 |
| 2 | 补后半段帧 18 + **hyd5-v1 自动识别火野猪预填** → 人工复核补充 → 重训 | 33 帧，v1 mAP50 0.652 / 火野猪 0.88 |
| 3 | **全片覆盖**：补帧 2020-3590 共 26 → hyd5-v2 预填（只预填怪物，**不预填 player**——实测 player 误检率高）+ 人工复核 | 47 帧，v2 mAP50 0.650 |
| 4 | **重切全片 train/val**（保证黑斧木妖/猴子进 val）→ hyd5-v3 | **hyd5-v3 mAP50 0.725** / 火野猪 0.995 / 黑斧木妖 0.745 |

### 9.3 关键经验

1. **冷启动正确姿势 = 人工画框 + 小模型预填 + 人工复核**，不是 VLM 自动标注。
   VLM 定位精度不足以直接生成标注（框错/漏检/误判出没表外类）。
2. **预填只做怪物、不做 player**：新域 player 形态杂（火焰之地多玩家），模型 player 误检率高，
   预填反而增加删除负担；player 由人工直接标。
3. **标注密度**：每 2 秒 1 帧 + dHash 去重足够，逐帧（30fps）标注无信息增益。
4. **跨域模板失配仍在**：模板库（icon 合成域）对新域帧 NCC 仅 0.741（域内应 >0.93）。
   本次因走"训练模型"路线绕开了模板依赖；后续全量管线需 `build-templates` 补新域模板。
5. **旧域残留误检随样本增多自然消失**：hyd5-v3 在未训练帧上不再检出水灵/蘑菇仔等 wgc 类。
6. **frame_review 面板过滤**：73 类超出面板可视高度（前 25 类），新增 `--classes` 只显示本地图类
   （自动含 player），滚轮在面板内滚动类别列表。见 tools/CLAUDE.md。

### 9.4 合并与备份

```bash
# 合并进主数据集（要求两边 dataset.yaml 73 类名完全一致，已满足）
uv run python tools/yolo_data.py merge-dataset --target data/wgc_review --source data/hyd5_coldstart
# 备份原始数据目录（合并前执行）
cp -r data/hyd5_coldstart data/backup_hyd5_20260902/
```

- 合并后主数据集 `data/wgc_review`：train 101 帧 / val 23 帧（原 wgc 67+10，新增 hyd5 34+13）。
- 备份保留在 `data/backup_hyd5_20260902/`（未跟踪，gitignore 由 `/data/` 覆盖）。

### 9.5 双域主模型 full-v1 + 云端化（2026-09-02 ✅）

对合并后的 `data/wgc_review`（train 101 / val 23，双地图域）跑完整训练（从 hyd5-v3 续训，
YOLO26n，80ep，本机 CPU 用时 ~47min，best epoch 64）：

```bash
uv run python tools/yolo_data.py train --dataset data/wgc_review \
    --model runs/train/hyd5-v3/weights/best.pt --epochs 80 --imgsz 640 --name full-v1 --device cpu
```

**结果**（`runs/train/full-v1/weights/best.pt`）：val 23 帧/189 实例，**mAP50 0.808 / mAP50-95 0.543**
（wgc-review-v1 0.788 / hyd5-v3 0.725）。关键改善：**player R 0.864 / mAP50 0.846**
（v1 的 player 召回 0.40 短板已修复）；绿水灵 0.972 / 蓝蜗牛 0.873 / 红蜗牛 0.833 / 火野猪 0.991；
猴子 R 0.5（val 仅 6 实例，样本噪声大，参考价值有限）。已归档 `weights/20260902/`
（best.pt / best.onnx / best.names）。

**GPU 训练（AutoDL）**：本机仅 CPU，完整训练是瓶颈 → 标准走 **AutoDL 实例**（RTX 4080）。数据整备后
SFTP 传数据+基础权重 → 远端 py3.8 训练 → 拉回 `best.pt` → 本地导出 `best.onnx`+`best.names` →
归档 `weights/<日期>/`。方法见 `docs/autodl_training.md`，脚本 `cloud/autodl/`。此前 PAI-DLC 路径已弃用删除。
第 8 步「人工启动训练」（§8 ⑤）用 AutoDL 替代：`merge_maps` → 上传 → 训练 → 拉回权重。

**跨多图迭代（2026-09-05 起）**：改按「每图一夹」组织，新图数据落 `data/maps/<地图>/`（每夹 images/{train,val}+labels/{train,val}+dataset.yaml，names 用完整 73 类表），不再逐个并入 `wgc_review`；最终统一训练用 `tools/merge_maps.py` 合并成 `data/unified_<round>/`（id 校验+去重+重切 val 保证稀有类进 val，输出无 `path` 键），再上 AutoDL（`docs/autodl_training.md`）。

快速验收（本地）：`crops_vis/<类>.png` 翻拼图板（格子编号可追溯）；`labels_vis/` 翻整帧图。
