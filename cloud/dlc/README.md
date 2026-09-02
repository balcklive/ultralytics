# 云端 YOLO 训练（阿里云 PAI-DLC + 自定义镜像）

把本地 `tools/yolo_data.py train`（CPU ~47 分钟/80ep）搬到 DLC 单机 GPU，几分钟一轮。
完整工作流、DLC 控制台配置表、成本与排障见 **`docs/cloud_dlc_training.md`**。

## 数据流（一次 OSS 读写挂载）

`oss://<bucket>/mxdzlk/` 挂载到容器 `/mnt/data`：

```
/mnt/data/datasets/<round>/   ← 训练数据（prepare.py 产物，dataset.yaml 无 path 键）
/mnt/data/models/base/*.pt    ← 基础权重（如 weights/20260902/best.pt）
/mnt/data/out/<round>/        ← 输出（runs/ + best.pt/.onnx/.names + done.marker）
```

## 4 步

```bash
# 1) 构建并推送镜像（改代码后需重推）
REGISTRY=registry.cn-hangzhou.aliyuncs.com ./cloud/dlc/build_push.sh

# 2) 数据整形 + 上传（prepare 纯本地；upload 需 ossutil 已配置）
python cloud/dlc/prepare.py --dataset data/wgc_review --out artifacts/cloud_round
BUCKET=mxdzlk-data ./cloud/dlc/upload.sh 20260902-r1 artifacts/cloud_round weights/20260902/best.pt

# 3) 控制台提交 DLC 任务（或 dlc submit pytorchjob），关键参数见文档
#    - 挂载 mxdzlk/ 前缀（读写）→ /mnt/data
#    - 环境变量：DATA_DIR=/mnt/data/datasets/20260902-r1
#                BASE_PT=/mnt/data/models/base/best.pt
#                RUN_NAME=cloud-v1   OUT_DIR=/mnt/data/out/20260902-r1
#                EPOCHS=80
#    - 启动命令留空（镜像已设 ENTRYPOINT）

# 4) OSS 取回权重，本地归位
ossutil cp -r oss://mxdzlk-data/mxdzlk/out/20260902-r1 weights/20260902-cloud/
```
