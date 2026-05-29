from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from llm_nest.core.storage.paths import models_dir

logger = logging.getLogger(__name__)

HUB_TIMEOUT = 30  # seconds


def download_model(
    repo_id: str,
    filename: str,
    *,
    dest_dir: Path | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """从 HuggingFace Hub 下载 GGUF 模型文件

    Args:
        repo_id: 仓库 ID，如 "tensorblock/tinyllama-15M-GGUF"
        filename: 文件名，如 "tinyllama-15m-q4_k_m.gguf"
        dest_dir: 目标目录，默认 ~/.llmn/models
        on_progress: 进度回调 (bytes_downloaded, total_bytes)

    Returns:
        下载后的本地文件路径

    Raises:
        RuntimeError: 网络不可用或下载失败
    """
    from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]

    target_dir = dest_dir or models_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s/%s ...", repo_id, filename)

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,  # type: ignore[arg-type]
        )
    except Exception as e:
        raise RuntimeError(f"下载失败: {e}") from e

    logger.info("Downloaded to %s", path)
    return Path(path)
