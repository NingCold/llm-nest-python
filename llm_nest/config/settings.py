from __future__ import annotations

import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "llm-nest"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    return json.loads(CONFIG_FILE.read_text())


def save_config(config: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2, ensure_ascii=False))


def get_lang_from_config() -> str:
    return load_config().get("lang", "en")


def set_lang_to_config(lang: str) -> None:
    config = load_config()
    config["lang"] = lang
    save_config(config)
