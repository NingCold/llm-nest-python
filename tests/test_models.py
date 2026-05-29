import pytest
from pathlib import Path

from llm_nest.core.models.enums import ModelStatus, QuantType
from llm_nest.core.models.model_info import ModelInfo, ModelMetadata


class TestQuantType:
    def test_str_enum(self):
        assert QuantType.Q4_K_M == "Q4_K_M"
        assert QuantType.Q4_K_M.value == "Q4_K_M"

    def test_all_values_unique(self):
        values = [q.value for q in QuantType]
        assert len(values) == len(set(values))


class TestModelStatus:
    def test_str_enum(self):
        assert ModelStatus.AVAILABLE == "available"
        assert ModelStatus.LOADED == "loaded"


class TestModelMetadata:
    def test_defaults(self):
        m = ModelMetadata()
        assert m.arch == ""
        assert m.context_length == 0
        assert m.extra == {}

    def test_frozen(self):
        m = ModelMetadata(arch="llama")
        with pytest.raises(AttributeError):
            m.arch = "other"  # type: ignore


class TestModelInfo:
    def test_basic(self, tmp_path: Path):
        p = tmp_path / "model.gguf"
        p.write_bytes(b"\x00" * 1024)
        info = ModelInfo(name="test", path=p, size_bytes=1024)
        assert info.name == "test"
        assert info.quant_type == QuantType.UNKNOWN
        assert info.status == ModelStatus.AVAILABLE

    def test_size_gb(self, tmp_path: Path):
        p = tmp_path / "model.gguf"
        p.write_bytes(b"\x00" * 1024)
        info = ModelInfo(name="test", path=p, size_bytes=1024 * 1024 * 1024)
        assert info.size_gb == pytest.approx(1.0)

    def test_display_name(self, tmp_path: Path):
        p = tmp_path / "model.gguf"
        p.write_bytes(b"\x00" * 1024)
        info = ModelInfo(
            name="qwen3",
            path=p,
            size_bytes=int(2.5 * 1024**3),
            quant_type=QuantType.Q4_K_M,
        )
        assert "qwen3" in info.display_name
        assert "Q4_K_M" in info.display_name
        assert "2.5GB" in info.display_name
