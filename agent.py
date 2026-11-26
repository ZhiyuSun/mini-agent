"""
最小 Agent Loop —— 不依赖任何 Agent 框架，只用 OpenAI 兼容的 Chat Completions 接口。
DeepSeek / 通义千问 / Kimi / MiniMax / OpenAI 等提供 OpenAI 兼容接口的模型都能直接用。

运行：
    pip install openai
    export LLM_BASE_URL=https://api.deepseek.com      # 换成你用的服务
    export LLM_API_KEY=sk-xxx
    export LLM_MODEL=deepseek-chat
    python agent.py "看看当前目录有哪些 Python 文件，统计一下总行数"

核心就一句话：
    while 模型还想调工具:
        调模型 -> 执行它要的工具 -> 把结果塞回上下文
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from openai import OpenAI

MAX_STEPS = 20              # 防止死循环：最多跑多少轮
MAX_TOOL_OUTPUT = 8000      # 工具结果太长会撑爆上下文，截断
WORKDIR = Path.cwd()        # 文件类工具只允许在这个目录下操作

SYSTEM_PROMPT = f"""你是一个能操作本地文件和命令行的助手。
工作目录：{WORKDIR}
- 需要信息时先用工具查，不要编造。
- 一步步来，每次调用工具后根据结果决定下一步。
- 任务完成后，直接用文字给出最终答复（不再调用工具）。"""


# ---------------------------------------------------------------------------
# 1. 工具：本质就是「函数 + 给模型看的说明书（JSON Schema）」
# ---------------------------------------------------------------------------

def _safe_path(p: str) -> Path:
    """把路径限制在 WORKDIR 内，防止模型读写任意文件。"""
    full = (WORKDIR / p).resolve()
    if full != WORKDIR and WORKDIR not in full.parents:
        raise ValueError(f"路径越界：{p}")
    return full


def list_dir(path: str = ".") -> str:
    entries = sorted(_safe_path(path).iterdir())
    return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries) or "(空目录)"


def read_file(path: str) -> str:
    return _safe_path(path).read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    f = _safe_path(path)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content, encoding="utf-8")
    return f"已写入 {path}（{len(content)} 字符）"


def run_shell(command: str) -> str:
    # 危险操作需要人确认 —— 这就是 Claude Code 里「是否允许执行」的雏形
    if input(f"\n⚠️  Agent 想执行：{command}\n允许吗？[y/N] ").strip().lower() != "y":
        return "用户拒绝执行该命令。"
    r = subprocess.run(command, shell=True, cwd=WORKDIR,
                       capture_output=True, text=True, timeout=60)
    return f"exit={r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"


# 名字 -> (函数, 给模型看的 schema)
TOOLS = {
    "list_dir": (list_dir, {
        "description": "列出目录内容",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "相对路径，默认 ."}}},
    }),
    "read_file": (read_file, {
        "description": "读取文本文件内容",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]},
    }),
    "write_file": (write_file, {
        "description": "写入（覆盖）文本文件",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]},
    }),
    "run_shell": (run_shell, {
        "description": "执行 shell 命令（需用户确认），返回退出码和输出",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]},
    }),
}

TOOL_SCHEMAS = [{"type": "function", "function": {"name": n, **s}}
                for n, (_, s) in TOOLS.items()]


def execute_tool(name: str, raw_args: str) -> str:
    """执行一次工具调用。关键：出错不要抛异常，而是把错误当结果还给模型，让它自己修正。"""
    if name not in TOOLS:
        return f"错误：不存在的工具 {name}"
    try:
        args = json.loads(raw_args or "{}")
        result = TOOLS[name][0](**args)
    except Exception as e:
        result = f"错误：{type(e).__name__}: {e}"
    if len(result) > MAX_TOOL_OUTPUT:
        result = result[:MAX_TOOL_OUTPUT] + f"\n...(已截断，原长 {len(result)} 字符)"
    return result


# ---------------------------------------------------------------------------
# 2. Agent Loop 本体
# ---------------------------------------------------------------------------

def run_agent(task: str, client, model: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    total_tokens = 0

    for step in range(1, MAX_STEPS + 1):
        # ① 调模型：把完整对话历史 + 工具说明书一起发过去
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOL_SCHEMAS)
        msg = resp.choices[0].message
        if resp.usage:
            total_tokens += resp.usage.total_tokens

        # ② 模型的回复必须原样追加进历史（包括它的 tool_calls），否则下一轮它会"失忆"
        messages.append(msg.model_dump(exclude_none=True))

        # ③ 没有工具调用 = 模型认为任务完成，退出循环
        if not msg.tool_calls:
            print(f"\n[完成] 共 {step} 轮，消耗 {total_tokens} tokens")
            return msg.content or ""

        # ④ 逐个执行工具，结果用 tool_call_id 对应回去
        for call in msg.tool_calls:
            name, args = call.function.name, call.function.arguments
            print(f"[第{step}轮] 🔧 {name}({args})")
            result = execute_tool(name, args)
            print(f"         ↳ {result[:200]}{'...' if len(result) > 200 else ''}")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })
        # ⑤ 回到 ①，带着工具结果再问模型

    return f"[中止] 超过最大轮数 {MAX_STEPS}，任务未完成"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('用法：python agent.py "你的任务"')
    client = OpenAI(base_url=os.environ.get("LLM_BASE_URL"),
                    api_key=os.environ["LLM_API_KEY"])
    answer = run_agent(sys.argv[1], client, os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    print("\n" + "=" * 60 + "\n" + answer)
