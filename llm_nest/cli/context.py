from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm_nest.core.registry.registry import ModelRegistry
from llm_nest.core.runtime.config import RuntimeConfig


@dataclass
class CLIContext:
    """CLI 上下文，显式传递依赖"""
    registry: ModelRegistry
    config: RuntimeConfig
    models_dir: Path | None = None


def create_context(models_dir: Path | None = None) -> CLIContext:
    """创建 CLI 上下文"""
    registry = ModelRegistry(directory=models_dir)
    registry.scan()
    config = RuntimeConfig()
    return CLIContext(registry=registry, config=config, models_dir=models_dir)
