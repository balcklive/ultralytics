import os
from pathlib import Path

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

import base64
import json
import uuid

from openai import OpenAI

with open(os.path.expanduser("~/.codex/auth.json"), encoding="utf-8") as f:
    auth = json.load(f)
client = OpenAI(
    base_url="https://chatgpt.com/backend-api/codex",
    api_key=auth["tokens"]["access_token"],
    default_headers={
        "chatgpt-account-id": auth["tokens"].get("account_id", ""),
        "originator": "codex_cli_rs",
        "OpenAI-Beta": "responses=experimental",
        "session_id": str(uuid.uuid4()),
    },
)

# 测试图取仓库根 artifacts/（脚本位于 tools/，据此定位仓库根）
IMG = Path(__file__).resolve().parent.parent / "artifacts" / "peek_labeled.png"
b64 = base64.b64encode(IMG.read_bytes()).decode()

input_items = [
    {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "详细描述这张图片的内容。它是什么风格的游戏素材？里面有哪些元素？"},
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
        ],
    }
]

parts = []
with client.responses.stream(input=input_items, model="gpt-5.6-luna", store=False) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            parts.append(event.delta)
        elif event.type == "response.completed":
            print("[status]", event.response.status, "| usage:", event.response.usage)
print("".join(parts))
