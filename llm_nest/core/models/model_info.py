from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from llm_nest.core.models.enums import ModelStatus, QuantType


@dataclass(frozen=True)
class ModelMetadata:
    """GGUF 内嵌元数据"""
    arch: str = ""
    context_length: int = 0
    vocab_size: int = 0
    chat_template: str = ""
    embedding_length: int = 0
    block_count: int = 0
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """模型完整描述"""
    name: str
    path: Path
    size_bytes: int
    quant_type: QuantType = QuantType.UNKNOWN
    metadata: ModelMetadata = field(default_factory=ModelMetadata)
    status: ModelStatus = ModelStatus.AVAILABLE
    source: str = ""
    downloaded_at: datetime | None = None

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)

    @property
    def display_name(self) -> str:
        return f"{self.name} ({self.quant_type.value}, {self.size_gb:.1f}GB)"
