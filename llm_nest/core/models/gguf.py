from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

from llm_nest.core.models.enums import QuantType

GGUF_MAGIC = 0x46475547  # "GGUF" in little-endian

GGUF_TYPE_MAP = {
    0: QuantType.F32,
    1: QuantType.F16,
    2: QuantType.Q4_0,
    3: QuantType.Q4_1,
    7: QuantType.Q5_0,
    8: QuantType.Q5_1,
    10: QuantType.Q8_0,
    11: QuantType.Q8_1,
    14: QuantType.Q2_K,
    15: QuantType.Q3_K_S,
    16: QuantType.Q3_K_M,
    17: QuantType.Q4_K_S,
    18: QuantType.Q4_K_M,
    19: QuantType.Q5_K_S,
    20: QuantType.Q5_K_M,
    21: QuantType.Q6_K,
    30: QuantType.IQ2_XXS,
    31: QuantType.IQ3_XXS,
    36: QuantType.IQ4_XS,
}


@dataclass
class GGUFMetadata:
    """GGUF 文件头解析结果"""
    version: int = 0
    tensor_count: int = 0
    metadata_kv: dict[str, str | int | float | bool | list[object]] = field(default_factory=dict)
    arch: str = ""
    context_length: int = 0
    vocab_size: int = 0
    chat_template: str = ""
    embedding_length: int = 0
    block_count: int = 0
    quant_type: QuantType = QuantType.UNKNOWN


def parse_gguf_header(path: Path) -> GGUFMetadata:
    """解析 GGUF 文件头，提取元数据和量化类型"""
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: {path} (magic: 0x{magic:08X})")

        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        metadata_kv_count = struct.unpack("<Q", f.read(8))[0]

        metadata_kv: dict[str, str | int | float | bool | list[object]] = {}
        arch = ""
        context_length = 0
        vocab_size = 0
        chat_template = ""
        embedding_length = 0
        block_count = 0
        quant_type = QuantType.UNKNOWN

        for _ in range(metadata_kv_count):
            key = _read_gguf_string(f)
            value_type = struct.unpack("<I", f.read(4))[0]
            value = _read_gguf_value(f, value_type)
            metadata_kv[key] = value

            if key == "general.architecture":
                arch = str(value)
            elif key == f"{arch}.context_length":
                context_length = int(value) if isinstance(value, int | str) else 0
            elif key == f"{arch}.vocab_size":
                vocab_size = int(value) if isinstance(value, int | str) else 0
            elif key == "tokenizer.chat_template":
                chat_template = str(value)
            elif key == f"{arch}.embedding_length":
                embedding_length = int(value) if isinstance(value, int | str) else 0
            elif key == f"{arch}.block_count":
                block_count = int(value) if isinstance(value, int | str) else 0

        # 从第一个张量类型推断量化类型
        if tensor_count > 0:
            quant_type = _infer_quant_type(f)

    return GGUFMetadata(
        version=version,
        tensor_count=tensor_count,
        metadata_kv=metadata_kv,
        arch=arch,
        context_length=context_length,
        vocab_size=vocab_size,
        chat_template=chat_template,
        embedding_length=embedding_length,
        block_count=block_count,
        quant_type=quant_type,
    )


def _read_gguf_string(f) -> str:
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")


def _read_gguf_value(f, value_type: int) -> str | int | float | bool | list[object]:
    match value_type:
        case 0:  # UINT8
            return struct.unpack("<B", f.read(1))[0]
        case 1:  # INT8
            return struct.unpack("<b", f.read(1))[0]
        case 2:  # UINT16
            return struct.unpack("<H", f.read(2))[0]
        case 3:  # INT16
            return struct.unpack("<h", f.read(2))[0]
        case 4:  # UINT32
            return struct.unpack("<I", f.read(4))[0]
        case 5:  # INT32
            return struct.unpack("<i", f.read(4))[0]
        case 6:  # FLOAT32
            return struct.unpack("<f", f.read(4))[0]
        case 7:  # BOOL
            return bool(struct.unpack("<B", f.read(1))[0])
        case 8:  # STRING
            return _read_gguf_string(f)
        case 9:  # ARRAY
            arr_type = struct.unpack("<I", f.read(4))[0]
            arr_len = struct.unpack("<Q", f.read(8))[0]
            return [_read_gguf_value(f, arr_type) for _ in range(arr_len)]
        case 10:  # UINT64
            return struct.unpack("<Q", f.read(8))[0]
        case 11:  # INT64
            return struct.unpack("<q", f.read(8))[0]
        case 12:  # FLOAT64
            return struct.unpack("<d", f.read(8))[0]
        case _:
            raise ValueError(f"Unknown GGUF value type: {value_type}")


def _infer_quant_type(f) -> QuantType:
    """从第一个张量类型推断量化方式"""
    try:
        _read_gguf_string(f)  # tensor name, not needed for quant type inference
        n_dims = struct.unpack("<I", f.read(4))[0]
        for _ in range(n_dims):
            struct.unpack("<Q", f.read(8))[0]  # dim size
        tensor_type = struct.unpack("<I", f.read(4))[0]
        return GGUF_TYPE_MAP.get(tensor_type, QuantType.UNKNOWN)
    except Exception:
        return QuantType.UNKNOWN
