from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from llm_nest.core.runtime.config import RuntimeConfig


class RuntimeBackend(ABC):
    """推理后端抽象基类"""

    @abstractmethod
    def load_model(self, path: Path, config: RuntimeConfig | None = None) -> None:
        """加载模型"""

    @abstractmethod
    def generate(self, prompt: str, config: RuntimeConfig | None = None) -> Iterator[str]:
        """流式生成，逐 token 返回"""

    @abstractmethod
    def generate_chat(
        self,
        messages: list[dict[str, str]],
        config: RuntimeConfig | None = None,
    ) -> Iterator[str]:
        """多轮对话生成，流式返回 token"""

    @abstractmethod
    def unload(self) -> None:
        """卸载模型，释放资源"""

    @abstractmethod
    def is_loaded(self) -> bool:
        """模型是否已加载"""

    @property
    @abstractmethod
    def name(self) -> str:
        """后端名称"""
