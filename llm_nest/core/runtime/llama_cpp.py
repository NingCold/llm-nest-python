from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from llm_nest.core.runtime.base import RuntimeBackend
from llm_nest.core.runtime.config import RuntimeConfig

logger = logging.getLogger(__name__)


class LlamaCppBackend(RuntimeBackend):
    """llama.cpp 推理后端"""

    def __init__(self) -> None:
        self._model = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "llama.cpp"

    def load_model(self, path: Path, config: RuntimeConfig | None = None) -> None:
        try:
            from llama_cpp import Llama  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python 未安装。请运行: uv add llama-cpp-python"
            ) from None

        cfg = config or RuntimeConfig()
        logger.info("Loading model: %s (n_ctx=%d, n_gpu_layers=%d)", path, cfg.n_ctx, cfg.n_gpu_layers)

        self._model = Llama(
            model_path=str(path),
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            n_gpu_layers=cfg.n_gpu_layers,
            n_batch=cfg.n_batch,
            verbose=False,
        )
        self._loaded = True

    def generate(self, prompt: str, config: RuntimeConfig | None = None) -> Iterator[str]:
        if not self._loaded or self._model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        cfg = config or RuntimeConfig()
        output = self._model(
            prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
            stream=True,
        )
        for chunk in output:
            token = chunk["choices"][0]["text"]  # type: ignore[index]
            if token:
                yield token

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        config: RuntimeConfig | None = None,
    ) -> Iterator[str]:
        if not self._loaded or self._model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        cfg = config or RuntimeConfig()
        output = self._model.create_chat_completion(
            messages=messages,  # type: ignore[arg-type]
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
            stream=True,
        )
        for chunk in output:
            delta = chunk["choices"][0].get("delta", {})  # type: ignore[union-attr]
            token = delta.get("content", "")
            if token:
                yield token

    def unload(self) -> None:
        self._model = None
        self._loaded = False
        logger.info("Model unloaded")

    def is_loaded(self) -> bool:
        return self._loaded
