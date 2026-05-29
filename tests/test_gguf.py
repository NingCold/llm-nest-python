import struct
import pytest
from pathlib import Path

from llm_nest.core.models.gguf import GGUF_MAGIC, parse_gguf_header
from llm_nest.core.models.enums import QuantType


def _write_minimal_gguf(path: Path, arch: str = "llama", context_length: int = 4096):
    """构造最小合法 GGUF 文件用于测试"""
    with open(path, "wb") as f:
        # Magic
        f.write(struct.pack("<I", GGUF_MAGIC))
        # Version
        f.write(struct.pack("<I", 3))
        # Tensor count
        f.write(struct.pack("<Q", 1))
        # Metadata KV count
        f.write(struct.pack("<Q", 2))

        # general.architecture
        key = b"general.architecture"
        f.write(struct.pack("<Q", len(key)))
        f.write(key)
        f.write(struct.pack("<I", 8))  # STRING type
        val = arch.encode()
        f.write(struct.pack("<Q", len(val)))
        f.write(val)

        # {arch}.context_length
        key = f"{arch}.context_length".encode()
        f.write(struct.pack("<Q", len(key)))
        f.write(key)
        f.write(struct.pack("<I", 4))  # UINT32 type
        f.write(struct.pack("<I", context_length))

        # 第一个张量信息（用于推断量化类型）
        tname = b"blk.0.attn_q.weight"
        f.write(struct.pack("<Q", len(tname)))
        f.write(tname)
        f.write(struct.pack("<I", 2))  # n_dims=2
        f.write(struct.pack("<Q", 4096))
        f.write(struct.pack("<Q", 4096))
        f.write(struct.pack("<I", 18))  # Q4_K_M


def test_parse_valid_gguf(tmp_path):
    path = tmp_path / "test.gguf"
    _write_minimal_gguf(path, arch="llama", context_length=8192)

    meta = parse_gguf_header(path)
    assert meta.version == 3
    assert meta.tensor_count == 1
    assert meta.arch == "llama"
    assert meta.context_length == 8192
    assert meta.quant_type == QuantType.Q4_K_M


def test_parse_non_gguf_file(tmp_path):
    path = tmp_path / "not_gguf.gguf"
    path.write_bytes(b"NOT_A_GGUF_FILE")

    with pytest.raises(ValueError, match="Not a GGUF file"):
        parse_gguf_header(path)


def test_parse_different_quant_type(tmp_path):
    path = tmp_path / "test.gguf"
    _write_minimal_gguf(path, arch="llama", context_length=2048)

    meta = parse_gguf_header(path)
    assert meta.arch == "llama"
    assert meta.context_length == 2048
    assert meta.quant_type == QuantType.Q4_K_M


def test_parse_gguf_with_chat_template(tmp_path):
    """测试带 chat_template 的 GGUF 解析"""
    path = tmp_path / "test.gguf"
    with open(path, "wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<Q", 3))  # 3 metadata entries

        # general.architecture
        key = b"general.architecture"
        f.write(struct.pack("<Q", len(key)))
        f.write(key)
        f.write(struct.pack("<I", 8))  # STRING
        val = b"llama"
        f.write(struct.pack("<Q", len(val)))
        f.write(val)

        # {arch}.context_length
        key = b"llama.context_length"
        f.write(struct.pack("<Q", len(key)))
        f.write(key)
        f.write(struct.pack("<I", 4))  # UINT32
        f.write(struct.pack("<I", 4096))

        # tokenizer.chat_template
        key = b"tokenizer.chat_template"
        f.write(struct.pack("<Q", len(key)))
        f.write(key)
        f.write(struct.pack("<I", 8))  # STRING
        val = b"{{ bos_token }}{% for message in messages %}..."
        f.write(struct.pack("<Q", len(val)))
        f.write(val)

        # 张量
        tname = b"blk.0.attn_q.weight"
        f.write(struct.pack("<Q", len(tname)))
        f.write(tname)
        f.write(struct.pack("<I", 2))
        f.write(struct.pack("<Q", 4096))
        f.write(struct.pack("<Q", 4096))
        f.write(struct.pack("<I", 21))  # Q6_K

    meta = parse_gguf_header(path)
    assert meta.chat_template == "{{ bos_token }}{% for message in messages %}..."
    assert meta.quant_type == QuantType.Q6_K
