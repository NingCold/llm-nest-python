import pytest
from unittest.mock import patch, MagicMock

from llm_nest.hub.result import HubModelResult
from llm_nest.hub.search import search_gguf, search_repo_files


class TestHubModelResult:
    def test_basic(self):
        result = HubModelResult(
            repo_id="Qwen/Qwen3-8B-GGUF",
            filename="qwen3-8b-q4_k_m.gguf",
            size_bytes=4_500_000_000,
            downloads=1000,
            likes=50,
        )
        assert result.repo_id == "Qwen/Qwen3-8B-GGUF"
        assert result.filename == "qwen3-8b-q4_k_m.gguf"
        assert result.size_bytes == 4_500_000_000
        assert result.downloads == 1000
        assert result.likes == 50

    def test_display_name(self):
        result = HubModelResult(
            repo_id="Qwen/Qwen3-8B-GGUF",
            filename="qwen3-8b-q4_k_m.gguf",
            size_bytes=1024,
        )
        assert result.display_name == "Qwen/Qwen3-8B-GGUF/qwen3-8b-q4_k_m.gguf"

    def test_size_gb(self):
        result = HubModelResult(
            repo_id="test/repo",
            filename="model.gguf",
            size_bytes=4 * 1024 ** 3,
        )
        assert result.size_gb == pytest.approx(4.0)

    def test_frozen(self):
        result = HubModelResult(
            repo_id="test/repo",
            filename="model.gguf",
            size_bytes=1024,
        )
        with pytest.raises(AttributeError):
            result.repo_id = "other"  # type: ignore

    def test_tags_tuple(self):
        result = HubModelResult(
            repo_id="test/repo",
            filename="model.gguf",
            size_bytes=1024,
            tags=("gguf", "llama"),
        )
        assert result.tags == ("gguf", "llama")


class TestSearchGguf:
    @patch("huggingface_hub.HfApi")
    def test_search_basic(self, mock_hf_api_cls):
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api

        mock_model = MagicMock()
        mock_model.id = "Qwen/Qwen3-8B-GGUF"

        mock_info = MagicMock()
        mock_info.id = "Qwen/Qwen3-8B-GGUF"
        mock_info.downloads = 5000
        mock_info.likes = 100
        mock_info.tags = ["gguf"]

        mock_sibling = MagicMock()
        mock_sibling.rfilename = "qwen3-8b-q4_k_m.gguf"
        mock_sibling.size = 4_500_000_000

        mock_info.siblings = [mock_sibling]
        mock_api.list_models.return_value = [mock_model]
        mock_api.model_info.return_value = mock_info

        results = search_gguf("qwen3", limit=5)

        assert len(results) == 1
        assert results[0].repo_id == "Qwen/Qwen3-8B-GGUF"
        assert results[0].filename == "qwen3-8b-q4_k_m.gguf"
        assert results[0].size_bytes == 4_500_000_000

    @patch("huggingface_hub.HfApi")
    def test_search_filters_non_gguf(self, mock_hf_api_cls):
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api

        mock_model = MagicMock()
        mock_model.id = "test/repo"

        mock_info = MagicMock()
        mock_info.id = "test/repo"
        mock_info.downloads = 0
        mock_info.likes = 0
        mock_info.tags = []

        mock_gguf = MagicMock()
        mock_gguf.rfilename = "model.gguf"
        mock_gguf.size = 1024

        mock_other = MagicMock()
        mock_other.rfilename = "README.md"
        mock_other.size = 100

        mock_info.siblings = [mock_gguf, mock_other]
        mock_api.list_models.return_value = [mock_model]
        mock_api.model_info.return_value = mock_info

        results = search_gguf("test")
        assert len(results) == 1
        assert results[0].filename == "model.gguf"


class TestSearchRepoFiles:
    @patch("huggingface_hub.HfApi")
    def test_list_repo_gguf_files(self, mock_hf_api_cls):
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api

        mock_info = MagicMock()
        mock_info.id = "test/repo"
        mock_info.downloads = 1000
        mock_info.likes = 50
        mock_info.tags = ["gguf"]

        mock_gguf = MagicMock()
        mock_gguf.rfilename = "model-q4_k_m.gguf"
        mock_gguf.size = 2_000_000_000

        mock_txt = MagicMock()
        mock_txt.rfilename = "README.md"
        mock_txt.size = 500

        mock_info.siblings = [mock_gguf, mock_txt]
        mock_api.model_info.return_value = mock_info

        results = search_repo_files("test/repo")
        assert len(results) == 1
        assert results[0].filename == "model-q4_k_m.gguf"


class TestDownload:
    @patch("huggingface_hub.hf_hub_download")
    def test_download_calls_hub(self, mock_download, tmp_path):
        from llm_nest.hub.download import download_model

        expected_path = tmp_path / "model.gguf"
        mock_download.return_value = str(expected_path)

        result = download_model(
            repo_id="test/repo",
            filename="model.gguf",
            dest_dir=tmp_path,
        )

        assert result == expected_path
        mock_download.assert_called_once_with(
            repo_id="test/repo",
            filename="model.gguf",
            local_dir=str(tmp_path),
            local_dir_use_symlinks=False,
        )
