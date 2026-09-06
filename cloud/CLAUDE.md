# cloud/

云端 GPU 训练（**AutoDL 实例**）相关脚本与说明，与 Ultralytics 包本体逻辑无关。本机无 GPU，训练统一走 AutoDL（已弃用阿里云 PAI-DLC）。

> 完整方法论见 `docs/autodl_training.md`。

## 文件清单

| 文件 | 说明 |
| --- | --- |
| `autodl/` | AutoDL 训练方法：数据/权重 SFTP 上传、远端 py3.8 训练模板、目录说明，详见 `autodl/CLAUDE.md` |

## 调用链

- 本地命令行运行 `autodl/` 下的脚本（`transfer_upload.py` 上传、`train_remote.py` 上传到远端跑）。
- 需要登录 AutoDL 实例（SSH + 凭据）的步骤由用户提供/执行；本目录只提供脚本/文档，凭据走环境变量不落盘。
- 数据/权重经 SFTP 传递，不走对象存储/镜像。
