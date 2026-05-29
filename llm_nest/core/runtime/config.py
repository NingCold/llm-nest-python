from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuntimeConfig:
    """推理运行时配置"""
    n_ctx: int = 4096
    n_threads: int = 4
    n_gpu_layers: int = 0
    n_batch: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40
    max_tokens: int = 512
    repeat_penalty: float = 1.1
    seed: int = -1
    stop: list[str] = field(default_factory=list)
    system_prompt: str = ""
