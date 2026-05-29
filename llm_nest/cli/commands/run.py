from __future__ import annotations

import sys

import typer

from llm_nest.cli.ui.output import print_error, print_info
from llm_nest.config.i18n import t


def run_model(
    name: str = typer.Argument(..., help=t("arg.name")),
    prompt: str = typer.Option("", "-p", "--prompt", help=t("opt.prompt")),
    n_ctx: int = typer.Option(4096, help=t("opt.n_ctx")),
    max_tokens: int = typer.Option(512, help=t("opt.max_tokens")),
    temperature: float = typer.Option(0.8, help=t("opt.temperature")),
    system_prompt: str = typer.Option("", "-s", "--system", help=t("opt.system_prompt")),
) -> None:
    from llm_nest.cli.context import create_context
    from llm_nest.core.runtime.config import RuntimeConfig
    from llm_nest.core.runtime.llama_cpp import LlamaCppBackend

    run_ctx = create_context()
    model = run_ctx.registry.get_model(name)
    if model is None:
        print_error(t("msg.model_not_found", name=name))
        raise typer.Exit(1)

    config = RuntimeConfig(
        n_ctx=n_ctx,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    )

    print_info(t("msg.loading_model", name=model.display_name))
    backend = LlamaCppBackend()
    try:
        backend.load_model(model.path, config)
    except RuntimeError as e:
        print_error(str(e))
        raise typer.Exit(1)

    if prompt:
        _single_shot(backend, prompt, config)
    else:
        _interactive_loop(backend, config, system_prompt)

    backend.unload()


class _ThinkRenderer:
    """流式渲染 think 标签：标签内文本用浅色，标签本身不输出"""

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False
        self._tag_buf = ""

    def feed(self, token: str) -> None:
        self._buf += token
        self._flush()

    def _flush(self) -> None:
        while self._buf:
            if self._in_think:
                end = self._buf.find("</think>")
                if end == -1:
                    # 全部是 think 内容，用浅色输出
                    sys.stdout.write(f"\033[2m{self._buf}\033[0m")
                    sys.stdout.flush()
                    self._buf = ""
                else:
                    # 输出 think 内容，然后退出 think 模式
                    think_content = self._buf[:end]
                    if think_content:
                        sys.stdout.write(f"\033[2m{think_content}\033[0m")
                        sys.stdout.flush()
                    self._buf = self._buf[end + len("</think>"):]
                    self._in_think = False
            else:
                think_start = self._buf.find("<think>")
                if think_start == -1:
                    # 检查是否有部分标签在末尾
                    for i in range(min(6, len(self._buf)), 0, -1):
                        if self._buf[-i:].startswith("<"):
                            prefix = self._buf[:-i]
                            self._tag_buf = self._buf[-i:]
                            if prefix:
                                sys.stdout.write(prefix)
                                sys.stdout.flush()
                            self._buf = ""
                            return
                    sys.stdout.write(self._buf)
                    sys.stdout.flush()
                    self._buf = ""
                else:
                    # 输出 think 之前的普通文本
                    normal = self._buf[:think_start]
                    if normal:
                        sys.stdout.write(normal)
                        sys.stdout.flush()
                    self._buf = self._buf[think_start + len("<think>"):]
                    self._in_think = True

    def finish(self) -> None:
        # 处理残留的 tag_buf
        if self._tag_buf:
            sys.stdout.write(self._tag_buf)
            sys.stdout.flush()
            self._tag_buf = ""
        self._flush()


def _print_stream(backend, messages, config) -> str:
    renderer = _ThinkRenderer()
    response = ""
    for token in backend.generate_chat(messages, config):
        renderer.feed(token)
        response += token
    renderer.finish()
    return response


def _single_shot(backend, prompt: str, config) -> None:
    messages: list[dict[str, str]] = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": prompt})

    _print_stream(backend, messages, config)
    print()


def _interactive_loop(backend, config, system_prompt: str) -> None:
    print_info(t("msg.chat_welcome"))
    print()

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    while True:
        try:
            user_input = input(f"{t('msg.chat_user')}: ")
        except (EOFError, KeyboardInterrupt):
            print()
            print_info(t("msg.chat_bye"))
            break

        if user_input.strip().lower() in ("exit", "quit"):
            print_info(t("msg.chat_bye"))
            break

        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})

        print(f"{t('msg.chat_assistant')}: ", end="", flush=True)
        response = ""
        try:
            response = _print_stream(backend, messages, config)
        except KeyboardInterrupt:
            print()
        print()
        print()

        if response:
            messages.append({"role": "assistant", "content": response})
