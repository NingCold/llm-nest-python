from __future__ import annotations

from pathlib import Path

from llm_nest.core.models import ModelInfo
from llm_nest.core.storage.model_store import delete_model_file, find_model, scan_models
from llm_nest.core.registry.aliases import load_aliases, resolve_alias, set_alias, remove_alias


class ModelRegistry:
    """模型注册表，管理本地模型的增删查"""

    def __init__(self, directory: Path | None = None) -> None:
        self._directory = directory
        self._models: list[ModelInfo] = []

    def scan(self) -> None:
        """扫描磁盘，刷新模型列表"""
        self._models = scan_models(self._directory)

    def list_models(self) -> list[ModelInfo]:
        """返回已扫描的模型列表"""
        return list(self._models)

    def get_model(self, name: str) -> ModelInfo | None:
        """按名称获取模型，支持别名解析"""
        resolved = resolve_alias(name)
        for model in self._models:
            if model.name == resolved:
                return model
        return find_model(resolved, self._directory)

    def delete_model(self, name: str) -> bool:
        """删除模型，返回是否成功"""
        resolved = resolve_alias(name)
        model = self.get_model(resolved)
        if model is None:
            return False
        delete_model_file(model.path)
        self._models = [m for m in self._models if m.path != model.path]
        return True

    def set_alias(self, alias: str, model_name: str) -> None:
        """设置模型别名"""
        set_alias(alias, model_name)

    def remove_alias(self, alias: str) -> bool:
        """删除别名"""
        return remove_alias(alias)

    def get_aliases(self) -> dict[str, str]:
        """获取所有别名"""
        return load_aliases()
