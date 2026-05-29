from __future__ import annotations

from pathlib import Path

from llm_nest.core.models import ModelInfo, ModelMetadata, ModelStatus, QuantType
from llm_nest.core.models.gguf import parse_gguf_header
from llm_nest.core.storage.paths import models_dir


GGUF_EXTENSIONS = {".gguf"}


def scan_models(directory: Path | None = None) -> list[ModelInfo]:
    """扫描目录，发现所有 GGUF 模型"""
    target_dir = directory or models_dir()
    if not target_dir.exists():
        return []

    models: list[ModelInfo] = []
    for path in sorted(target_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in GGUF_EXTENSIONS:
            models.append(_path_to_model_info(path))
    return models


def find_model(name: str, directory: Path | None = None) -> ModelInfo | None:
    """按名称查找模型（精确匹配文件名）"""
    target_dir = directory or models_dir()
    for ext in GGUF_EXTENSIONS:
        path = target_dir / f"{name}{ext}"
        if path.exists():
            return _path_to_model_info(path)
    return None


def delete_model_file(path: Path) -> None:
    """删除模型文件"""
    if path.exists():
        path.unlink()


def _path_to_model_info(path: Path) -> ModelInfo:
    """将文件路径转换为 ModelInfo"""
    name = path.stem
    size_bytes = path.stat().st_size
    quant_type = QuantType.UNKNOWN
    metadata = ModelMetadata()

    try:
        gguf_meta = parse_gguf_header(path)
        quant_type = gguf_meta.quant_type
        metadata = ModelMetadata(
            arch=gguf_meta.arch,
            context_length=gguf_meta.context_length,
            vocab_size=gguf_meta.vocab_size,
            chat_template=gguf_meta.chat_template,
            embedding_length=gguf_meta.embedding_length,
            block_count=gguf_meta.block_count,
        )
    except Exception:
        pass  # 解析失败不影响基本文件信息

    return ModelInfo(
        name=name,
        path=path,
        size_bytes=size_bytes,
        quant_type=quant_type,
        metadata=metadata,
        status=ModelStatus.AVAILABLE,
    )
