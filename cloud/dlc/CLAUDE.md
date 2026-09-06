# cloud/dlc/

阿里云 PAI-DLC 云端 GPU 训练包：把本地 `tools/yolo_data.py train` 原样搬到 DLC 单机 GPU 任务，
做到「推镜像 → 传数据集/基础权重到 OSS → 提交 DLC job → 权重写回 OSS」。数据仍以 `data/wgc_review`
为代表的小数据集（124 图 / YOLO26n），单卡几分钟一轮。

> 完整工作流见 `docs/cloud_dlc_training.md`；镜像/数据流设计决策见该文档「设计」。

## 文件清单

| 文件 | 说明 |
| --- | --- |
| `Dockerfile` | 训练镜像：`python:3.12-slim` + uv 建虚拟环境，torch/CUDA 走 PyPI Linux GPU wheel；仅 COPY 仓库必要源码（pyproject/uv.lock/.python-version/README/LICENSE/`ultralytics/`/`tools/`），不含数据。`ARG PYPI_MIRROR` 指国内镜像加速 |
| `entrypoint.sh` | 镜像入口：读 env（`DATA_DIR/BASE_PT/RUN_NAME/OUT_DIR`，可选 `EPOCHS/IMGSZ/BATCH/DEVICE/EXPORT_ONNX`）跑 `tools/yolo_data.py train --device 0`；单 Worker 时清除 PAI 注入的伪 DDP rank；把 `best.pt`(+onnx/names) 拷到 `OUT_DIR` 顶层并落 `done.marker` 强制 JindoFuse 落盘 |
| `build_push.sh` | `docker build -f cloud/dlc/Dockerfile .` 推 ACR；env `REGISTRY/NAMESPACE/IMAGE/TAG`（默认 tag=日期） |
| `prepare.py` | **纯本地**数据整形：复制数据集到 `--out`，dataset.yaml **去掉 Windows 绝对 `path:` 键**（ultralytics 无 path 时以 yaml 目录为根，任意挂载路径通用），打印计数校验 |
| `upload.sh` | 把 prepare 产物传 `oss://<bucket>/mxdzlk/datasets/<round>/`、基础权重传 `models/base/`（需 ossutil 已配置） |
| `README.md` | 端到端 4 步速览，细节指向 docs/cloud_dlc_training.md |

## 调用链

- `prepare.py` → `upload.sh`（上传产物）→ 用户在 DLC 控制台/CLI 提交 job → 容器跑 `entrypoint.sh` → 权重回 OSS。
- 镜像只含仓库源码与依赖；**数据集/基础权重/输出全部来自 OSS 读写挂载** `/mnt/data`，单次挂载一个前缀 `oss://<bucket>/mxdzlk`。

## 关键规则 / 注意事项

- 云端 dataset.yaml **必须去掉 `path` 键**（本地 Windows 路径在 Linux 容器失效），用 `prepare.py` 生成。
- 输出目录 `OUT_DIR` 必须落在 OSS 读写挂载内；`done.marker` + `ls -R` 用于触发 JindoFuse 落盘。
- 基础权重用每轮归档版（如 `weights/<日期>/best.pt`），不重新下载 `models/yolo26s.pt`。
- 需要云凭据的步骤本机没有（无 aliyun/ossutil），由用户执行；脚本参数全走 env，不硬编码 bucket/region。
- 云端改动训练/工具脚本后需 `build_push.sh` 重新构建镜像（镜像内含源码副本，非挂载）。
- 镜像安装固定 PyTorch CUDA 12.4 wheel，匹配 PAI 当前 GPU 驱动；单卡任务不要使用 `BATCH=-1`，请设置正整数（如 `16`）。
