from __future__ import annotations

import json
from pathlib import Path

from llm_nest.core.storage.paths import models_dir

ALIASES_FILE = "aliases.json"


def _aliases_path() -> Path:
    return models_dir() / ALIASES_FILE


def load_aliases() -> dict[str, str]:
    """加载别名映射 {别名: 模型名}"""
    path = _aliases_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_aliases(aliases: dict[str, str]) -> None:
    """保存别名映射"""
    path = _aliases_path()
    path.write_text(json.dumps(aliases, indent=2, ensure_ascii=False))


def set_alias(alias: str, model_name: str) -> None:
    """设置别名"""
    aliases = load_aliases()
    aliases[alias] = model_name
    save_aliases(aliases)


def remove_alias(alias: str) -> bool:
    """删除别名，返回是否成功"""
    aliases = load_aliases()
    if alias not in aliases:
        return False
    del aliases[alias]
    save_aliases(aliases)
    return True


def resolve_alias(name: str) -> str:
    """解析别名，如果不是别名则原样返回"""
    aliases = load_aliases()
    return aliases.get(name, name)
