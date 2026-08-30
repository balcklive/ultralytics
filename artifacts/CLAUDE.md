# artifacts/

## 目录职责

存放调试产物、工具脚本的输出文件（图片、标注结果、评估 JSON）以及临时的 SDK 测试脚本。本目录内容一般为生成物，不参与构建流程；仅 Python 测试脚本需要维护。

## 文件清单（仅列源码与关键产物）

- `test_codex_sdk.py` — 测试 OpenAI Codex SDK（`openai-codex` 包）：thread_start + 逐轮 sandbox 切换（workspace_write → read_only），复用 `~/.codex/auth.json` 的 ChatGPT 登录凭证
- `test_openai_sdk.py` — 测试用 openai SDK 裸调 Codex 后端（`https://chatgpt.com/backend-api/codex`）：流式 Responses API、手动重发历史实现多轮对话、验证后端限制（不支持 conversations API / previous_response_id / code_interpreter）
- `test_openai_vision.py` — 测试 `gpt-5.6-luna` 的视觉输入：base64 `input_image` 识别本目录的 YOLO 标注审查图
- `codex_out*.txt`、`*.png`、`*.json*`、`auto_label_out/`、`template_eval_vis/`、`vlm_test_*/`、`UI检查/`、`冰龙检查/` 等为各工具脚本的生成产物，可随时重新生成或清理

## 调用链

- 三个 `test_*.py` 均为独立可运行脚本（`python artifacts/test_*.py`），依赖项目 `.venv` 中手动安装的 `openai-codex`、`openai` 包（未写入 pyproject.toml）
- 均通过环境变量 `http_proxy/https_proxy=http://127.0.0.1:7890` 走本地代理，认证复用 `~/.codex/auth.json`
- `test_openai_vision.py` 读取本目录 `peek_labeled.png` 作为测试图片

## 关键规则/注意事项

- 三个测试脚本中的 `Make the requested change.` 类 prompt 会在 `workspace_write` 沙箱下**真实修改仓库文件**，运行前注意检查 `git status`
- openai SDK 走的是非官方 Codex 后端，接口约束多（input 必须为列表、必须流式、不支持 previous_response_id 与 code_interpreter 工具），详见 `test_openai_sdk.py` 内注释
- 本目录其余文件为生成物，不要手工编辑；提交时应选择性添加，避免把大体积产物带入 git
