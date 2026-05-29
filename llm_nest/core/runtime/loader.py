from __future__ import annotations

import logging
from pathlib import Path

from llm_nest.core.runtime.config import RuntimeConfig

logger = logging.getLogger(__name__)

# GGUF 文件大小 -> 推荐 RAM（GB），粗略估算
RAM_MULTIPLIER = 0.6  # 量化模型通常需要文件大小的 ~60% RAM


def estimate_ram_gb(model_path: Path) -> float:
    """根据模型文件大小估算所需 RAM（GB）"""
    size_bytes = model_path.stat().st_size
    return (size_bytes / (1024 ** 3)) / RAM_MULTIPLIER


def create_default_config(
    model_path: Path,
    n_threads: int | None = None,
    n_gpu_layers: int = 0,
) -> RuntimeConfig:
    """根据系统情况创建默认配置"""
    import os

    if n_threads is None:
        n_threads = min(os.cpu_count() or 4, 8)

    ram_gb = estimate_ram_gb(model_path)
    n_ctx = 4096
    if ram_gb > 16:
        n_ctx = 8192
    elif ram_gb > 8:
        n_ctx = 4096
    else:
        n_ctx = 2048

    return RuntimeConfig(
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
    )
