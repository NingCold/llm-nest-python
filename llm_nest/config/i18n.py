from llm_nest.config.locales import LANG_EN, LANG_ZH

MESSAGES: dict[str, dict[str, str]] = {
    "zh": LANG_ZH,
    "en": LANG_EN,
}

_current_lang = "en"


def set_lang(lang: str) -> None:
    global _current_lang
    if lang in MESSAGES:
        _current_lang = lang


def get_lang() -> str:
    return _current_lang


def t(key: str, **kwargs: object) -> str:
    """获取当前语言的翻译文本，支持 {name} 格式化"""
    text = MESSAGES.get(_current_lang, LANG_EN).get(key, key)
    if kwargs:
        return text.format(**kwargs)
    return text
