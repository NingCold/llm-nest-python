import typer

from llm_nest.cli.ui.output import print_error, print_info, print_success
from llm_nest.config.i18n import t

app = typer.Typer(help=t("cmd.hub.help"))


@app.command(help=t("cmd.hub.get.desc"))
def get(
    repo_id: str = typer.Argument(..., help=t("arg.repo_id")),
    filename: str = typer.Option("", "-f", "--filename", help=t("opt.filename")),
) -> None:
    from llm_nest.cli.ui.output import print_hub_results
    from llm_nest.hub.search import search_repo_files

    try:
        from llm_nest.cli.ui.output import searching_status
        with searching_status():
            results = search_repo_files(repo_id)
    except RuntimeError as e:
        print_error(str(e))
        raise typer.Exit(1)

    if not results:
        print_error(t("msg.repo_no_gguf", repo_id=repo_id))
        raise typer.Exit(1)

    if not filename:
        print_hub_results(results)
        if len(results) == 1:
            selected = results[0]
        else:
            choice = typer.prompt(t("msg.select_file"), type=int, default=1) - 1
            if choice < 0 or choice >= len(results):
                print_error(t("msg.invalid_choice"))
                raise typer.Exit(1)
            selected = results[choice]
    else:
        matches = [r for r in results if r.filename == filename]
        if not matches:
            print_error(t("msg.file_not_exist", filename=filename))
            for r in results:
                print_info(f"  {r.filename}")
            raise typer.Exit(1)
        selected = matches[0]

    print_info(t("msg.downloading", name=selected.display_name))

    from llm_nest.hub.download import download_model

    try:
        path = download_model(
            repo_id=repo_id,
            filename=selected.filename,
        )
        print_success(t("msg.download_done", path=path))
    except RuntimeError as e:
        print_error(str(e))
        raise typer.Exit(1)


@app.command(help=t("cmd.hub.search.desc"))
def search(
    query: str = typer.Argument(..., help=t("arg.query")),
    limit: int = typer.Option(10, help=t("opt.limit")),
) -> None:
    from llm_nest.cli.ui.output import print_hub_results
    from llm_nest.hub.search import search_gguf

    try:
        from llm_nest.cli.ui.output import searching_status
        with searching_status():
            results = search_gguf(query, limit=limit)
        print_hub_results(results)
    except RuntimeError as e:
        print_error(str(e))
        raise typer.Exit(1)
