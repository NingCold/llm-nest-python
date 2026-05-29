import time
import uuid

import typer

from llm_nest.cli.ui.output import print_error, print_info
from llm_nest.config.i18n import t


def serve_model(
    name: str = typer.Argument(..., help=t("arg.name")),
    host: str = typer.Option("127.0.0.1", help=t("opt.host")),
    port: int = typer.Option(8000, help=t("opt.port")),
    n_ctx: int = typer.Option(4096, help=t("opt.n_ctx")),
    max_tokens: int = typer.Option(512, help=t("opt.max_tokens")),
    system_prompt: str = typer.Option("", "-s", "--system", help=t("opt.system_prompt")),
) -> None:
    try:
        import fastapi  # type: ignore[import-untyped]  # noqa: F401
        import uvicorn  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        print_error("fastapi/uvicorn 未安装。请运行: uv add 'llm-nest-python[server]'")
        raise typer.Exit(1)

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
        system_prompt=system_prompt,
    )

    print_info(t("msg.loading_model", name=model.display_name))
    backend = LlamaCppBackend()
    try:
        backend.load_model(model.path, config)
    except RuntimeError as e:
        print_error(str(e))
        raise typer.Exit(1)

    app = _create_app(backend, name, config)
    url = f"http://{host}:{port}"
    print_info(t("msg.server_ready", url=url))

    import uvicorn  # type: ignore[import-untyped]

    uvicorn.run(app, host=host, port=port, log_level="warning")


def _create_app(backend, model_name: str, default_config):
    from fastapi import FastAPI  # type: ignore[import-untyped]
    from fastapi.responses import StreamingResponse  # type: ignore[import-untyped]
    from pydantic import BaseModel  # type: ignore[import-untyped]

    app = FastAPI(title="llm-nest", version="0.1.0")

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        model: str = model_name
        messages: list[ChatMessage]
        stream: bool = False
        temperature: float | None = None
        max_tokens: int | None = None
        top_p: float | None = None

    class ChoiceMessage(BaseModel):
        role: str = "assistant"
        content: str

    class Choice(BaseModel):
        index: int = 0
        message: ChoiceMessage
        finish_reason: str = "stop"

    class Usage(BaseModel):
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0

    class ChatCompletionResponse(BaseModel):
        id: str
        object: str = "chat.completion"
        created: int
        model: str
        choices: list[Choice]
        usage: Usage = Usage()

    def _make_config(req: ChatCompletionRequest):
        from llm_nest.core.runtime.config import RuntimeConfig
        cfg = RuntimeConfig(
            n_ctx=default_config.n_ctx,
            n_threads=default_config.n_threads,
            n_gpu_layers=default_config.n_gpu_layers,
            max_tokens=req.max_tokens or default_config.max_tokens,
            temperature=req.temperature or default_config.temperature,
            top_p=req.top_p or default_config.top_p,
            system_prompt=default_config.system_prompt,
        )
        return cfg

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name, "loaded": backend.is_loaded()}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        if not backend.is_loaded():
            return {"error": "Model not loaded"}

        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        cfg = _make_config(req)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        if req.stream:
            return StreamingResponse(
                _stream_response(backend, messages, cfg, completion_id, created, model_name),
                media_type="text/event-stream",
            )

        response_text = ""
        for token in backend.generate_chat(messages, cfg):
            response_text += token

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=model_name,
            choices=[Choice(message=ChoiceMessage(content=response_text))],
        ).model_dump()

    return app


async def _stream_response(backend, messages, cfg, completion_id, created, model_name):
    import json

    for token in backend.generate_chat(messages, cfg):
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    done_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done_chunk)}\n\n"
    yield "data: [DONE]\n\n"
