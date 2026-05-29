from __future__ import annotations

import re
from pathlib import Path

from llm_nest.core.models import ModelInfo, QuantType
from llm_nest.core.storage.model_store import scan_models


def search_models(
    query: str,
    models: list[ModelInfo] | None = None,
    directory: Path | None = None,
) -> list[ModelInfo]:
    """模糊搜索模型（按名称匹配）"""
    all_models = models if models is not None else scan_models(directory)
    if not query:
        return all_models

    pattern = re.compile(re.escape(query), re.IGNORECASE)
    return [m for m in all_models if pattern.search(m.name)]


def filter_by_quant(
    models: list[ModelInfo],
    quant_type: QuantType,
) -> list[ModelInfo]:
    """按量化类型过滤"""
    return [m for m in models if m.quant_type == quant_type]


def filter_gguf_only(models: list[ModelInfo]) -> list[ModelInfo]:
    """只保留 GGUF 格式模型"""
    return [m for m in models if m.path.suffix.lower() == ".gguf"]
