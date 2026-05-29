import struct
from pathlib import Path

from llm_nest.core.models.gguf import GGUF_MAGIC
from llm_nest.core.registry.registry import ModelRegistry
from llm_nest.core.registry.search import search_models
from llm_nest.core.registry.aliases import set_alias, load_aliases, resolve_alias, remove_alias


def _make_gguf(path: Path, name: str = "model.gguf") -> Path:
    fp = path / name
    with open(fp, "wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 0))
        f.write(struct.pack("<Q", 0))
    return fp


class TestAliases:
    def test_set_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr("llm_nest.core.registry.aliases.models_dir", lambda: tmp_path)
        set_alias("qwen", "qwen3-8b-q4_k_m")
        aliases = load_aliases()
        assert aliases["qwen"] == "qwen3-8b-q4_k_m"

    def test_resolve_existing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("llm_nest.core.registry.aliases.models_dir", lambda: tmp_path)
        set_alias("qwen", "qwen3-8b-q4_k_m")
        assert resolve_alias("qwen") == "qwen3-8b-q4_k_m"

    def test_resolve_passthrough(self, tmp_path, monkeypatch):
        monkeypatch.setattr("llm_nest.core.registry.aliases.models_dir", lambda: tmp_path)
        assert resolve_alias("not_an_alias") == "not_an_alias"

    def test_remove(self, tmp_path, monkeypatch):
        monkeypatch.setattr("llm_nest.core.registry.aliases.models_dir", lambda: tmp_path)
        set_alias("qwen", "qwen3-8b-q4_k_m")
        assert remove_alias("qwen") is True
        assert load_aliases() == {}

    def test_remove_nonexistent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("llm_nest.core.registry.aliases.models_dir", lambda: tmp_path)
        assert remove_alias("nope") is False


class TestSearch:
    def test_search_all(self, tmp_path):
        _make_gguf(tmp_path, "qwen3-8b.gguf")
        _make_gguf(tmp_path, "llama3-8b.gguf")
        models = search_models("", directory=tmp_path)
        assert len(models) == 2

    def test_search_by_name(self, tmp_path):
        _make_gguf(tmp_path, "qwen3-8b.gguf")
        _make_gguf(tmp_path, "llama3-8b.gguf")
        results = search_models("qwen", directory=tmp_path)
        assert len(results) == 1
        assert results[0].name == "qwen3-8b"

    def test_search_case_insensitive(self, tmp_path):
        _make_gguf(tmp_path, "Qwen3-8B.gguf")
        results = search_models("qwen", directory=tmp_path)
        assert len(results) == 1

    def test_search_no_match(self, tmp_path):
        _make_gguf(tmp_path, "qwen3-8b.gguf")
        results = search_models("llama", directory=tmp_path)
        assert len(results) == 0


class TestModelRegistry:
    def test_scan_and_list(self, tmp_path):
        _make_gguf(tmp_path, "model-a.gguf")
        _make_gguf(tmp_path, "model-b.gguf")
        reg = ModelRegistry(directory=tmp_path)
        reg.scan()
        assert len(reg.list_models()) == 2

    def test_get_model(self, tmp_path):
        _make_gguf(tmp_path, "qwen3.gguf")
        reg = ModelRegistry(directory=tmp_path)
        reg.scan()
        model = reg.get_model("qwen3")
        assert model is not None
        assert model.name == "qwen3"

    def test_get_model_not_found(self, tmp_path):
        reg = ModelRegistry(directory=tmp_path)
        reg.scan()
        assert reg.get_model("nope") is None

    def test_delete_model(self, tmp_path):
        _make_gguf(tmp_path, "qwen3.gguf")
        reg = ModelRegistry(directory=tmp_path)
        reg.scan()
        assert reg.delete_model("qwen3") is True
        assert len(reg.list_models()) == 0
        assert not (tmp_path / "qwen3.gguf").exists()

    def test_delete_nonexistent(self, tmp_path):
        reg = ModelRegistry(directory=tmp_path)
        reg.scan()
        assert reg.delete_model("nope") is False

    def test_alias_workflow(self, tmp_path):
        _make_gguf(tmp_path, "qwen3-8b-q4_k_m.gguf")
        reg = ModelRegistry(directory=tmp_path)
        reg.scan()
        reg.set_alias("qwen", "qwen3-8b-q4_k_m")
        model = reg.get_model("qwen")
        assert model is not None
        assert model.name == "qwen3-8b-q4_k_m"
