# 云端 GPU 训练：阿里云 PAI-DLC（工作文档）

> 目标：把本地 `tools/yolo_data.py train` 原样搬到阿里云 PAI-DLC 单机 GPU 任务，
> 做到「推镜像 → 传数据集/基础权重到 OSS → 提交 job → 权重写回 OSS → 结束即停计费」，
> 一轮（YOLO26n / 124 图 / 80ep）从本地 CPU ~47 分钟压到几分钟。
> 实现文件在 `cloud/dlc/`；本文是完整工作流与配置参考。

## 0. 何时用 / 前置条件

- 训练数据以 `data/*`（gitignore）小数据集为代表，图片+标签共几十 MB，基础权重用每轮归档版（如 `weights/<日期>/best.pt`）。
- 已开通：阿里云 **PAI**、**OSS**（至少 1 个 bucket）、**ACR**（容器镜像服务）。
- 本机需可执行云侧命令：`docker`（build/push）、`ossutil`（传数据）。`aliyun`/`pai` CLI 可选。
- 涉及账号/费用/配额的操作只能由你自己执行；`cloud/dlc/` 里的脚本把参数全部走 env，不硬编码 bucket/region。

## 1. 设计与数据流（一次 OSS 读写挂载）

**镜像**：`python:3.12-slim` + uv 虚拟环境；torch/CUDA 用 PyPI Linux GPU wheel（自带 CUDA 运行库，宿主机只需 NVIDIA 驱动——DLC GPU 节点已具备）。这样不依赖 PAI 官方镜像 tag，规避版本漂移；`Dockerfile` 提供 `--build-arg PYPI_MIRROR=...` 指国内镜像加速。

镜像内 **只有仓库源码 + 依赖**（`COPY` 仅 pyproject/uv.lock/.python-version/README/LICENSE + `ultralytics/` + `tools/`），**不含** `data/` `weights/` `runs/` → build context 小、可复用。改训练/工具脚本后要 `build_push.sh` 重推镜像。

**数据流**：DLC 把 OSS 前缀 `oss://<bucket>/mxdzlk/` **读写**挂载到容器 `/mnt/data`（JindoFuse）：

| 容器内路径 | OSS 对象 | 内容 |
|---|---|---|
| `/mnt/data/datasets/<round>/` | `mxdzlk/datasets/<round>/` | 训练数据（含 images/train、images/val、labels、dataset.yaml） |
| `/mnt/data/models/base/*.pt` | `mxdzlk/models/base/` | 基础权重（只读使用） |
| `/mnt/data/out/<round>/` | `mxdzlk/out/<round>/` | 输出：`runs/<RUN_NAME>/` + 顶层 `best.pt/.onnx/.names` + `done.marker` |

**关键点：云端 dataset.yaml 必须去掉 `path:` 键。** 本地 `data/*/dataset.yaml` 的 `path:` 是 Windows 绝对路径，在 Linux 容器失效；ultralytics 在无 `path` 键时以 yaml 自身目录为根（`ultralytics/data/utils.py:542`），因此去掉后任意挂载路径都通用。用 `cloud/dlc/prepare.py` 生成，不改动本地数据。

## 2. 步骤 ① 构建并推送镜像

```bash
# 先建 ACR 仓库（控制台，命名空间建议 mxdzlk，仓库名 yolo-train），并让 docker 登录 ACR：
docker login --username <阿里云账号或RAM> registry.cn-hangzhou.aliyuncs.com   # 密码=ACR 访问凭证

# 构建 + 推送（默认 tag=当天日期；PUSH=0 只构建不推）
REGISTRY=registry.cn-hangzhou.aliyuncs.com ./cloud/dlc/build_push.sh
# 国内拉 torch 慢时：
REGISTRY=registry.cn-hangzhou.aliyuncs.com docker build \
  --build-arg PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple \
  -f cloud/dlc/Dockerfile -t <...>/mxdzlk/yolo-train:<tag> . && docker push <...>
```

## 3. 步骤 ② 数据整形 + 上传

```bash
# prepare 纯本地：复制 data/wgc_review 到 artifacts/cloud_round，写出无 path 的 dataset.yaml，打印 101/23 计数
python cloud/dlc/prepare.py --dataset data/wgc_review --out artifacts/cloud_round
# 基础权重：每轮用上一轮归档 best.pt（如 weights/20260902/best.pt），不要用 models/yolo26*.pt
BUCKET=mxdzlk-data ./cloud/dlc/upload.sh 20260902-r1 artifacts/cloud_round weights/20260902/best.pt
```

> 注：`upload.sh` 上传前会自动剔除 ultralytics 生成的 `*.cache`（它们内嵌本机绝对图片路径，在 Linux 容器无效）；容器首次扫描会用容器内路径重建 cache，勿手动传 cache 上去。

## 4. 步骤 ③ 提交 DLC 任务

PAI 控制台 → 分布式训练（DLC）→ 新建任务。参考配置（菜单名随地域/版本略异）：

| 项 | 值 |
|---|---|
| 任务名称 | 如 `mxdzlk-full-v1` |
| 节点镜像 | **镜像地址**：`<REGISTRY>/mxdzlk/yolo-train:<日期>`（在 ACR 里填密码授权或设为公开拉取） |
| 框架 | PyTorch（单机，忽略分布式注入 env 即可）或 Custom |
| 节点数 | 1（单机） |
| 实例规格 | GPU：选**抢占式**省钱。YOLO26n 很小，T4（ecs.gn6i 类）/A10 皆可，够用即可 |
| 数据集挂载 | 自定义数据集（OSS 类型，**读写**）指向 `oss://<bucket>/mxdzlk/` → 容器路径 `/mnt/data` |
| 启动命令 | **留空**（镜像已设 `ENTRYPOINT`），或 `bash -lc "true"` 兜底 |
| 环境变量 | 见下表 |

| 环境变量 | 值 |
|---|---|
| `DATA_DIR` | `/mnt/data/datasets/20260902-r1` |
| `BASE_PT` | `/mnt/data/models/base/best.pt` |
| `RUN_NAME` | `cloud-v1` |
| `OUT_DIR` | `/mnt/data/out/20260902-r1` |
| `EPOCHS` | `80`（可选 `IMGSZ/BATCH/DEVICE/EXPORT_ONNX`） |

等价的 `dlc submit pytorchjob`（装 `dlc` 命令行后）示例：

```bash
dlc submit pytorchjob \
  --name=mxdzlk-full-v1 --workers=1 \
  --worker_image=<REGISTRY>/mxdzlk/yolo-train:<tag> \
  --command='' \
  --data_source_uris='oss://<bucket>/.::/mnt/data/:{mountType:jindo}' \
  --worker_spec=<GPU 规格>
```

任务跑完 → DLC 状态自动变 Stopped/Succeeded，GPU 停计费。容器内结果已由 `entrypoint.sh` 写到 `/mnt/data/out/...`。

## 5. 步骤 ④ 取回权重并归位

```bash
ossutil cp -r -f oss://<bucket>/mxdzlk/out/20260902-r1 weights/20260902-cloud/
ls weights/20260902-cloud/          # best.pt / best.onnx / best.names / runs/<RUN_NAME>/
```

取回后可用 `weights/20260902-cloud/best.pt` 验证/投入下一卷录像标注（同 `weights/<日期>/` 惯例）。

## 6. 成本 / 清理提醒

- 用**抢占式**公共资源 + 训练完成即停：单轮通常几元内（T4 档时租更低）。
- OSS/NAS 存储按量另计；不再用的 `out/<round>`、旧 `datasets/<round>` 及时清理。
- 换卡/换地域/换 bucket 只改脚本 env 或控制台选项，不动代码。

## 7. 排障

| 症状 | 排查 |
|---|---|
| 容器报 `dataset.yaml not found` | `DATA_DIR` 指向的是含 dataset.yaml 的目录（`datasets/<round>`），不是更上层 |
| 图片加载/缓存路径报错 | 数据里残留本机生成的 `*.cache`（内嵌 Windows 绝对路径）；重新 `prepare.py`+`upload.sh` 上传，或本地 `find <dir> -name '*.cache' -delete` 后重传 |
| `base weights not found` | `BASE_PT` 是容器内 `/mnt/data/...` 路径，需先跑 upload.sh |
| OSS 里看不到输出 | 读挂载权限/RAM role（`AliyunPAIAccessingOSSRole`）；结尾 `done.marker` 未出现=训练中断 |
| 镜像拉取慢/失败 | `PYPI_MIRROR` 指国内 PyPI；确认 ACR 已授权 DLC 拉取（公开或填账号密码） |
| torch/onnx 依赖装不上 | Dockerfile 已含 opencv 的 `libgl1`；真机网络问题可换 `python:3.12-slim` 内置源镜像重试 |
| GPU 利用率/显存与本地不同 | `batch=-1` 走 GPU autobatch，与 CPU 结果数值可略异，属预期 |

## 8. 与自动标注飞轮的衔接

对应 `docs/auto_labeling_pipeline.md` 第 8 步「人工启动训练」的云化替代：新批复核集 `merge-dataset --target data/wgc_review` 并入后，重跑上面 ②③ 即完成新一轮云端训练，权重落 OSS 后归位 `weights/<日期>/`。

相关实现：`cloud/dlc/`（文件职责见其 `CLAUDE.md`）。
