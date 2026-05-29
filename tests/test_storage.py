from pathlib import Path

from llm_nest.core.storage.paths import models_dir, cache_dir, model_path
from llm_nest.core.storage.file_ops import atomic_write, check_disk_space, file_hash
from llm_nest.core.storage.model_store import scan_models, find_model, delete_model_file


class TestPaths:
    def test_models_dir_creates(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("llm_nest.core.storage.paths.DEFAULT_MODELS_DIR", tmp_path / "models")
        result = models_dir()
        assert result.exists()
        assert result == tmp_path / "models"

    def test_cache_dir_creates(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("llm_nest.core.storage.paths.DEFAULT_CACHE_DIR", tmp_path / "cache")
        result = cache_dir()
        assert result.exists()
        assert result == tmp_path / "cache"

    def test_model_path(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("llm_nest.core.storage.paths.DEFAULT_MODELS_DIR", tmp_path)
        result = model_path("qwen3-q4_k_m.gguf")
        assert result == tmp_path / "qwen3-q4_k_m.gguf"


class TestFileOps:
    def test_atomic_write(self, tmp_path: Path):
        target = tmp_path / "test.gguf"
        data = b"GGUF" + b"\x00" * 100
        atomic_write(target, data)
        assert target.exists()
        assert target.read_bytes() == data

    def test_atomic_write_overwrite(self, tmp_path: Path):
        target = tmp_path / "test.gguf"
        atomic_write(target, b"first")
        atomic_write(target, b"second")
        assert target.read_bytes() == b"second"

    def test_check_disk_space_enough(self, tmp_path: Path):
        assert check_disk_space(tmp_path / "test.gguf", 1024)

    def test_file_hash(self, tmp_path: Path):
        target = tmp_path / "test.gguf"
        target.write_bytes(b"hello")
        h = file_hash(target)
        assert len(h) == 64  # sha256 hex length
        assert h == file_hash(target)  # deterministic


class TestModelStore:
    def _make_gguf(self, path: Path, name: str = "model.gguf") -> Path:
        """创建最小 GGUF 文件"""
        import struct
        from llm_nest.core.models.gguf import GGUF_MAGIC

        fp = path / name
        with open(fp, "wb") as f:
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", 3))
            f.write(struct.pack("<Q", 0))  # no tensors
            f.write(struct.pack("<Q", 0))  # no metadata
        return fp

    def test_scan_empty_dir(self, tmp_path: Path):
        assert scan_models(tmp_path) == []

    def test_scan_nonexistent_dir(self, tmp_path: Path):
        assert scan_models(tmp_path / "nope") == []

    def test_scan_finds_gguf(self, tmp_path: Path):
        self._make_gguf(tmp_path, "qwen3.gguf")
        models = scan_models(tmp_path)
        assert len(models) == 1
        assert models[0].name == "qwen3"

    def test_scan_ignores_non_gguf(self, tmp_path: Path):
        self._make_gguf(tmp_path, "model.gguf")
        (tmp_path / "readme.txt").write_text("hello")
        models = scan_models(tmp_path)
        assert len(models) == 1

    def test_find_model(self, tmp_path: Path):
        self._make_gguf(tmp_path, "qwen3.gguf")
        result = find_model("qwen3", tmp_path)
        assert result is not None
        assert result.name == "qwen3"

    def test_find_model_not_found(self, tmp_path: Path):
        assert find_model("nope", tmp_path) is None

    def test_delete_model_file(self, tmp_path: Path):
        fp = self._make_gguf(tmp_path)
        assert fp.exists()
        delete_model_file(fp)
        assert not fp.exists()
