from __future__ import annotations

from llm_nest.config.i18n import t
from llm_nest.core.models import ModelInfo

console = None


def _get_console():
    global console
    if console is None:
        from rich.console import Console
        console = Console()
    return console


def print_models(models: list[ModelInfo], source: str = "local") -> None:
    from rich.table import Table

    c = _get_console()
    if not models:
        c.print(f"[dim]{t('msg.no_models')}[/dim]")
        return

    table = Table(show_header=True, header_style="dim")
    table.add_column(t("table.source"), style="dim")
    table.add_column(t("table.name"), style="green")
    table.add_column(t("table.quant"))
    table.add_column(t("table.size"), justify="right")
    table.add_column(t("table.arch"))
    table.add_column(t("table.status"))

    source_label = t("source.local") if source == "local" else t("source.hub")
    for model in models:
        table.add_row(
            source_label,
            model.name,
            model.quant_type.value,
            f"{model.size_gb:.1f}GB",
            model.metadata.arch or "-",
            model.status.value,
        )

    c.print(table)


def print_search_results(
    local_models: list[ModelInfo],
    hub_results: list,
) -> None:
    from rich.table import Table

    c = _get_console()
    if not local_models and not hub_results:
        c.print(f"[dim]{t('msg.no_models')}[/dim]")
        return

    table = Table(show_header=True, header_style="dim")
    table.add_column(t("table.source"), style="dim")
    table.add_column(t("table.name"), style="green")
    table.add_column(t("table.quant"))
    table.add_column(t("table.size"), justify="right")
    table.add_column(t("table.arch"))

    for model in local_models:
        table.add_row(
            f"[green]{t('source.local')}[/green]",
            model.name,
            model.quant_type.value,
            f"{model.size_gb:.1f}GB",
            model.metadata.arch or "-",
        )

    for r in hub_results:
        size = f"{r.size_gb:.1f}GB" if r.size_bytes > 0 else "-"
        table.add_row(
            f"[yellow]{t('source.hub')}[/yellow]",
            r.display_name,
            "-",
            size,
            "-",
        )

    c.print(table)


def print_hub_results(results: list) -> None:
    from rich.table import Table

    c = _get_console()
    if not results:
        c.print(f"[dim]{t('msg.no_results')}[/dim]")
        return

    table = Table(show_header=True, header_style="dim")
    table.add_column("#", justify="right", style="dim")
    table.add_column(t("table.repo"))
    table.add_column(t("table.file"))
    table.add_column(t("table.size"), justify="right")
    table.add_column(t("table.downloads"), justify="right")

    for i, r in enumerate(results, 1):
        size = f"{r.size_gb:.1f}GB" if r.size_bytes > 0 else "-"
        table.add_row(
            str(i),
            r.repo_id,
            r.filename,
            size,
            f"{r.downloads:,}",
        )

    c.print(table)


def print_success(msg: str) -> None:
    _get_console().print(f"[green]{msg}[/green]")


def print_error(msg: str) -> None:
    _get_console().print(f"[red]{msg}[/red]")


def print_info(msg: str) -> None:
    _get_console().print(f"[blue]{msg}[/blue]")


def searching_status(message: str | None = None):
    """返回一个 Rich status context manager，用于显示搜索加载状态"""
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        msg = message or t("msg.searching")
        with _get_console().status(msg):
            yield

    return _ctx()
