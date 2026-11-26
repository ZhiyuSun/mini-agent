# mini-agent

一个不依赖任何 Agent 框架的最小 Agent Loop（约 150 行），用来理解 Agent 的核心机制。

使用 OpenAI 兼容的 Chat Completions 接口，DeepSeek / 通义千问 / Kimi / MiniMax / OpenAI 等均可接入。

## 快速开始

```bash
pip install -r requirements.txt
export LLM_BASE_URL=https://api.deepseek.com   # 换成你用的服务
export LLM_API_KEY=sk-xxx
export LLM_MODEL=deepseek-chat
python agent.py "看看当前目录有哪些 Python 文件，统计一下总行数"
```

## 核心流程

```
while 模型还想调工具:
    ① 调模型（对话历史 + 工具 schema）
    ② 把模型回复原样追加进历史
    ③ 没有工具调用 → 任务完成，退出
    ④ 执行工具，结果按 tool_call_id 追加进历史
    ⑤ 回到 ①
```

## 内置工具

| 工具 | 说明 |
|---|---|
| `list_dir` | 列出目录 |
| `read_file` | 读文件 |
| `write_file` | 写文件 |
| `run_shell` | 执行命令（需人工确认） |

文件类工具限制在当前工作目录内。

## 设计要点

- **错误回传**：工具出错不抛异常，而是作为结果返回给模型，让它自我修正
- **结果截断**：工具输出超过 8000 字符截断，防止撑爆上下文
- **最大轮数**：默认 20 轮，防止死循环
- **人工确认**：危险操作（shell）执行前需用户确认

## 下一步

- [ ] 上下文压缩：历史过长时自动摘要
- [ ] 流式输出
- [ ] 接入 MCP Server
- [ ] 评测：一组固定任务自动跑分
