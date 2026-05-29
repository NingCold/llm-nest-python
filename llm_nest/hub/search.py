from __future__ import annotations

import logging

from llm_nest.hub.result import HubModelResult

logger = logging.getLogger(__name__)


def search_gguf(
    query: str,
    *,
    limit: int = 10,
    sort: str = "downloads",
) -> list[HubModelResult]:
    """从 HuggingFace Hub 搜索 GGUF 模型文件

    先搜索 repo，再逐个获取 repo 内的 GGUF 文件列表。

    Args:
        query: 搜索关键词
        limit: 最大返回数量
        sort: 排序方式 (downloads, likes, lastModified)

    Returns:
        GGUF 文件列表

    Raises:
        RuntimeError: 网络不可用或 API 调用失败
    """
    from huggingface_hub import HfApi  # type: ignore[import-untyped]

    api = HfApi()

    try:
        models_iter = api.list_models(
            search=query,
            sort=sort,  # type: ignore[arg-type]
            limit=limit,
            filter="gguf",
        )
    except Exception as e:
        raise RuntimeError(f"搜索失败: {e}") from e

    results: list[HubModelResult] = []
    for model in models_iter:
        try:
            info = api.model_info(model.id, files_metadata=True)
            siblings = info.siblings or []
            for sibling in siblings:
                if sibling.rfilename and sibling.rfilename.endswith(".gguf"):
                    size = sibling.size if hasattr(sibling, "size") and sibling.size else 0
                    results.append(
                        HubModelResult(
                            repo_id=info.id,
                            filename=sibling.rfilename,
                            size_bytes=size or 0,
                            downloads=info.downloads or 0,
                            likes=info.likes or 0,
                            tags=tuple(info.tags or []),
                        )
                    )
        except Exception:
            logger.debug("Failed to get files for %s", model.id, exc_info=True)
            continue

    def _sort_key(r: HubModelResult) -> tuple[int, int]:
        org = r.repo_id.split("/")[0].lower() if "/" in r.repo_id else ""
        is_official = 1 if org != "thebloke" else 0
        return (-is_official, -r.downloads)

    results.sort(key=_sort_key)
    return results


def search_repo_files(repo_id: str) -> list[HubModelResult]:
    """列出某个 repo 下的所有 GGUF 文件

    Args:
        repo_id: 仓库 ID，如 "tensorblock/tinyllama-15M-GGUF"

    Returns:
        GGUF 文件列表

    Raises:
        RuntimeError: 网络不可用或 repo 不存在
    """
    from huggingface_hub import HfApi  # type: ignore[import-untyped]

    api = HfApi()

    try:
        model_info = api.model_info(repo_id, files_metadata=True)
    except Exception as e:
        raise RuntimeError(f"获取 repo 信息失败: {e}") from e

    siblings = model_info.siblings or []
    results: list[HubModelResult] = []
    for sibling in siblings:
        if sibling.rfilename and sibling.rfilename.endswith(".gguf"):
            size = sibling.size if hasattr(sibling, "size") and sibling.size else 0
            results.append(
                HubModelResult(
                    repo_id=repo_id,
                    filename=sibling.rfilename,
                    size_bytes=size or 0,
                    downloads=model_info.downloads or 0,
                    likes=model_info.likes or 0,
                    tags=tuple(model_info.tags or []),
                )
            )
    return results
