from pathlib import Path


DEFAULT_MODELS_DIR = Path.home() / ".llmn" / "models"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "llm-nest"


def models_dir() -> Path:
    """模型存储目录，不存在时自动创建"""
    DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_MODELS_DIR


def cache_dir() -> Path:
    """缓存目录，不存在时自动创建"""
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_CACHE_DIR


def model_path(name: str) -> Path:
    """根据模型名生成完整路径"""
    return models_dir() / name
