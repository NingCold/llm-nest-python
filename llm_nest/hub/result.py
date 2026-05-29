from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HubModelResult:
    """HuggingFace Hub 搜索结果"""
    repo_id: str
    filename: str
    size_bytes: int
    downloads: int = 0
    likes: int = 0
    tags: tuple[str, ...] = ()

    @property
    def display_name(self) -> str:
        return f"{self.repo_id}/{self.filename}"

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)
