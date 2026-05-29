# llm-nest

[English](./README.md) | [中文](./README_zh.md)

本地 LLM Runtime 与 GGUF 模型管理工具。定位类似 `ollama`——轻量、离线优先、无需守护进程。

## 特性

- **模型管理** — 从 HuggingFace Hub 搜索、下载、列出、删除 GGUF 模型
- **交互式对话** — REPL 模式，支持多轮对话和流式输出
- **API 服务** — OpenAI 兼容的 `/v1/chat/completions` 接口，支持 SSE 流式响应

## 安装

```bash
pip install llm-nest-python
```

包含 API 服务支持（FastAPI + uvicorn）：

```bash
pip install llm-nest-python[server]
```

### 从源码安装（uv）

```bash
git clone https://github.com/NingCold/llm-nest-python.git
cd llm-nest-python
uv sync
uv build
uv tool install .
uv run llmn --help
```

## 快速开始

### 下载模型

```bash
llmn hub get tensorblock/tinyllama-15M-GGUF -f tinyllama-15M-Q4_K_M.gguf
```

### 列出本地模型

```bash
llmn model list
```

### 交互式对话

```bash
llmn run tinyllama-15M-Q4_K_M
```

```
Loading model: tinyllama-15M-Q4_K_M (Q4_K_M, 0.0GB) ...
Type 'exit' or press Ctrl+C to quit

你: 你好
助手: 你好！有什么可以帮你的吗？

你: exit
再见！
```

### 单次推理

```bash
llmn run tinyllama-15M-Q4_K_M -p "2+2等于几？" --max-tokens 50
```

### 带系统提示词

```bash
llmn run tinyllama-15M-Q4_K_M -s "你是一个有帮助的助手。"
```

## API 服务

启动 OpenAI 兼容的 HTTP 服务：

```bash
llmn serve tinyllama-15M-Q4_K_M --port 8000
```

### 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/v1/models` | GET | 列出已加载模型 |
| `/v1/chat/completions` | POST | 聊天补全（支持流式） |

### 使用 curl

```bash
# 非流式
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "你好"}]}'

# 流式（SSE）
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "你好"}], "stream": true}'
```

### 使用 OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
resp = client.chat.completions.create(
    model="tinyllama-15M-Q4_K_M",
    messages=[{"role": "user", "content": "你好"}],
)
print(resp.choices[0].message.content)
```

## 命令

```
llmn model list                          # 列出本地模型
llmn model search <query>                # 搜索本地 + Hub 模型
llmn model info <name>                   # 查看模型详情
llmn model remove <name>                 # 删除本地模型

llmn hub search <query>                  # 搜索 HuggingFace Hub
llmn hub get <repo_id> [-f filename]     # 从 Hub 下载模型

llmn run <model> [-p prompt] [-s system] # 交互式对话或单次推理
llmn serve <model> [--port 8000]         # 启动 API 服务

llmn lang zh                             # 切换到中文
llmn lang en                             # 切换到英文
llmn version                             # 显示版本
```

## 存储

模型存储在 `~/.llmn/models/`。配置文件在 `~/.config/llm-nest/config.json`。

## 开发

```bash
uv sync
uv run pytest
uv run ruff check .
uv run pyright
```

## 许可证

MIT
