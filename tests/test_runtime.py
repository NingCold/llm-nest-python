import pytest
from pathlib import Path
from collections.abc import Iterator

from llm_nest.core.runtime.config import RuntimeConfig
from llm_nest.core.runtime.base import RuntimeBackend
from llm_nest.core.runtime.loader import estimate_ram_gb, create_default_config


class TestRuntimeConfig:
    def test_defaults(self):
        cfg = RuntimeConfig()
        assert cfg.n_ctx == 4096
        assert cfg.n_threads == 4
        assert cfg.n_gpu_layers == 0
        assert cfg.temperature == 0.8
        assert cfg.max_tokens == 512

    def test_custom(self):
        cfg = RuntimeConfig(n_ctx=8192, n_threads=8, temperature=0.5)
        assert cfg.n_ctx == 8192
        assert cfg.n_threads == 8
        assert cfg.temperature == 0.5


class TestEstimateRam:
    def test_estimate(self, tmp_path: Path):
        # 模拟 4GB 模型文件
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * (4 * 1024 ** 3))
        ram = estimate_ram_gb(model)
        assert ram == pytest.approx(4.0 / 0.6, rel=0.1)


class TestCreateDefaultConfig:
    def test_basic(self, tmp_path: Path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * 1024)
        cfg = create_default_config(model)
        assert cfg.n_ctx in (2048, 4096, 8192)
        assert cfg.n_threads > 0
        assert cfg.n_gpu_layers == 0

    def test_with_gpu(self, tmp_path: Path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * 1024)
        cfg = create_default_config(model, n_gpu_layers=32)
        assert cfg.n_gpu_layers == 32


class _StubBackend(RuntimeBackend):
    """用于测试的 stub 后端"""

    def __init__(self) -> None:
        self._loaded = False

    @property
    def name(self) -> str:
        return "stub"

    def load_model(self, path: Path, config: RuntimeConfig | None = None) -> None:
        self._loaded = True

    def generate(self, prompt: str, config: RuntimeConfig | None = None) -> Iterator[str]:
        if not self._loaded:
            raise RuntimeError("模型未加载")
        yield "hello"
        yield " world"

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        config: RuntimeConfig | None = None,
    ) -> Iterator[str]:
        if not self._loaded:
            raise RuntimeError("模型未加载")
        yield "hello"
        yield " chat"

    def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded


class TestRuntimeBackend:
    def test_stub_lifecycle(self):
        backend = _StubBackend()
        assert not backend.is_loaded()
        assert backend.name == "stub"

        backend.load_model(Path("/fake/model.gguf"))
        assert backend.is_loaded()

        tokens = list(backend.generate("hi"))
        assert tokens == ["hello", " world"]

        tokens = list(backend.generate_chat([{"role": "user", "content": "hi"}]))
        assert tokens == ["hello", " chat"]

        backend.unload()
        assert not backend.is_loaded()

    def test_generate_without_load_raises(self):
        backend = _StubBackend()
        with pytest.raises(RuntimeError, match="未加载"):
            list(backend.generate("hi"))
