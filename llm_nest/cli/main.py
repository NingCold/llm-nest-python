import typer
import typer.core

from llm_nest.config.i18n import set_lang, t
from llm_nest.config.settings import get_lang_from_config, set_lang_to_config


def _init_lang() -> None:
    lang = get_lang_from_config()
    set_lang(lang)


_init_lang()


class I18nTyperGroup(typer.core.TyperGroup):
    def get_params(self, ctx):
        params = super().get_params(ctx)
        for p in params:
            if p.name == "help":
                p.help = t("opt.help")  # type: ignore[attr-defined]
            elif p.name == "install_completion":
                p.help = t("opt.install_completion")  # type: ignore[attr-defined]
            elif p.name == "show_completion":
                p.help = t("opt.show_completion")  # type: ignore[attr-defined]
        return params


app = typer.Typer(
    name="llmn",
    help=t("cmd.main.help"),
    no_args_is_help=True,
    cls=I18nTyperGroup,
    add_completion=False,
)

from llm_nest.cli.commands.models import app as models_app  # noqa: E402
from llm_nest.cli.commands.hub import app as hub_app  # noqa: E402
from llm_nest.cli.commands.run import run_model  # noqa: E402
from llm_nest.cli.commands.serve import serve_model  # noqa: E402

app.add_typer(models_app, name="model")
app.add_typer(hub_app, name="hub")
app.command(name="run", help=t("cmd.run.help"))(run_model)
app.command(name="serve", help=t("cmd.serve.help"))(serve_model)


@app.command(help=t("cmd.version.desc"))
def version() -> None:
    typer.echo("llm-nest-python 0.1.0")


@app.command(help=t("cmd.lang.desc"))
def lang(
    language: str = typer.Argument(..., help=t("arg.language")),
) -> None:
    if language not in ("zh", "en"):
        typer.echo(t("msg.lang_supported"))
        raise typer.Exit(1)
    set_lang(language)
    set_lang_to_config(language)
    typer.echo(t("msg.lang_set"))


def main() -> None:
    app()
