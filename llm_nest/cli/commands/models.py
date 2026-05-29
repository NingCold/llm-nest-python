import typer

from llm_nest.cli.context import create_context
from llm_nest.cli.ui.output import print_error, print_info, print_models, print_search_results
from llm_nest.config.i18n import t

app = typer.Typer(help=t("cmd.model.help"))


@app.command("list", help=t("cmd.model.list.desc"))
def list_models() -> None:
    ctx = create_context()
    models = ctx.registry.list_models()
    print_models(models)


@app.command(help=t("cmd.model.search.desc"))
def search(
    query: str = typer.Argument(..., help=t("arg.query")),
    source: str = typer.Option("all", help=t("opt.source")),
    limit: int = typer.Option(10, help=t("opt.limit")),
) -> None:
    from llm_nest.core.registry.search import search_models

    local_results = []
    hub_results = []

    if source in ("local", "all"):
        ctx = create_context()
        local_results = search_models(query, ctx.registry.list_models())

    if source in ("hub", "all"):
        from llm_nest.hub.search import search_gguf
        from llm_nest.cli.ui.output import searching_status

        try:
            with searching_status():
                hub_results = search_gguf(query, limit=limit)
        except RuntimeError as e:
            if source == "hub":
                print_error(str(e))
                raise typer.Exit(1)
            print_info(t("msg.hub_search_failed", error=e))

    print_search_results(local_results, hub_results)


@app.command(help=t("cmd.model.info.desc"))
def info(
    name: str = typer.Argument(..., help=t("arg.name")),
) -> None:
    ctx = create_context()
    model = ctx.registry.get_model(name)
    if model is None:
        print_error(t("msg.model_not_found", name=name))
        raise typer.Exit(1)

    from rich.panel import Panel

    meta = model.metadata
    lines = [
        f"[bold]{t('info.name')}:[/bold] {model.name}",
        f"[bold]{t('info.path')}:[/bold] {model.path}",
        f"[bold]{t('info.size')}:[/bold] {model.size_gb:.2f} GB",
        f"[bold]{t('info.quant')}:[/bold] {model.quant_type.value}",
        f"[bold]{t('info.status')}:[/bold] {model.status.value}",
        f"[bold]{t('info.arch')}:[/bold] {meta.arch or t('info.unknown')}",
        f"[bold]{t('info.ctx_length')}:[/bold] {meta.context_length or t('info.unknown')}",
        f"[bold]{t('info.vocab_size')}:[/bold] {meta.vocab_size or t('info.unknown')}",
        f"[bold]{t('info.embedding')}:[/bold] {meta.embedding_length or t('info.unknown')}",
        f"[bold]{t('info.blocks')}:[/bold] {meta.block_count or t('info.unknown')}",
    ]
    _get_console().print(Panel("\n".join(lines), title=t("info.title")))


@app.command(help=t("cmd.model.remove.desc"))
def remove(
    name: str = typer.Argument(..., help=t("arg.name")),
    force: bool = typer.Option(False, "-f", "--force", help=t("opt.force")),
) -> None:
    from llm_nest.cli.ui.output import print_success

    ctx = create_context()
    model = ctx.registry.get_model(name)
    if model is None:
        print_error(t("msg.model_not_found", name=name))
        raise typer.Exit(1)

    if not force:
        typer.confirm(t("msg.confirm_delete", name=model.display_name), abort=True)

    ctx.registry.delete_model(name)
    print_success(t("msg.deleted", name=model.name))


def _get_console():
    from llm_nest.cli.ui.output import _get_console as gc
    return gc()
