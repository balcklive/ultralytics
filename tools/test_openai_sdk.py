import os

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

import json
import uuid

from openai import OpenAI

with open(os.path.expanduser("~/.codex/auth.json"), encoding="utf-8") as f:
    auth = json.load(f)
token = auth["tokens"]["access_token"]
account_id = auth["tokens"].get("account_id", "")

client = OpenAI(
    base_url="https://chatgpt.com/backend-api/codex",
    api_key=token,
    default_headers={
        "chatgpt-account-id": account_id,
        "originator": "codex_cli_rs",
        "OpenAI-Beta": "responses=experimental",
        "session_id": str(uuid.uuid4()),
    },
)


def run(input_items, **kwargs):
    """模拟 thread.run：流式调用，收集 delta 拼出最终文本。"""
    text_parts = []
    response_id = None
    with client.responses.stream(input=input_items, store=False, **kwargs) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                text_parts.append(event.delta)
            elif event.type == "response.completed":
                response_id = event.response.id
    return "".join(text_parts), response_id


user_msg = lambda t: [
    {"role": "user", "content": [{"type": "input_text", "text": t}]}
]

# —— 第 1 轮（对应 thread_start + run）——
history = user_msg("Reply with exactly one word: the capital of France.")
text1, _ = run(history, model="gpt-5.6-luna")
print("turn 1:", text1)

# —— 第 2 轮（后端不支持 previous_response_id，手动重发完整历史）——
history += [
    {"role": "assistant", "content": [{"type": "output_text", "text": text1}]},
    *user_msg("Now tell me the capital of Japan, one word."),
]
text2, _ = run(history, model="gpt-5.6-luna")
print("turn 2:", text2)

# —— 沙箱模拟：code_interpreter 容器工具 ——
try:
    text3, _ = run(
        user_msg("Use the code interpreter to compute 2**64 - 1 and report the number."),
        model="gpt-5.6-luna",
        tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
    )
    print("code_interpreter:", text3[:200])
except Exception as e:
    print("code_interpreter not supported by this backend:", str(e)[:200])
