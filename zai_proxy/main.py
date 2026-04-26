from __future__ import annotations

from contextlib import asynccontextmanager
import time
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException

from .client import ZaiClient, ZaiUpstreamError
from .config import Settings
from .deepseek_client import DeepSeekClient, DeepSeekUpstreamError
from .schemas import (
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    Choice,
    ModelCard,
    ModelsResponse,
    OpenAIMessage,
    ResponsesRequest,
    ResponsesResponse,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
    Usage,
    ZaiChatRequest,
    ZaiChatResponse,
    ZaiDeleteChatResponse,
)


settings = Settings.from_env()
client = ZaiClient(settings) if settings.zai_token else None
deepseek_client = DeepSeekClient(settings) if settings.deepseek_token else None


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    if client:
        await client.close()
    if deepseek_client:
        await deepseek_client.close()


app = FastAPI(title="z.ai and DeepSeek OpenAI Proxy", version="0.2.0", lifespan=lifespan)


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    if not settings.proxy_api_key:
        return

    expected = f"Bearer {settings.proxy_api_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/")
async def root() -> dict:
    return {
        "name": "z.ai and DeepSeek OpenAI Proxy",
        "endpoints": [
            "/v1/models",
            "/v1/chat/completions",
            "/v1/responses",
            "/deepseek/v1/models",
            "/deepseek/v1/chat/completions",
            "/deepseek/v1/responses",
            "/zai/chat",
            "/healthz",
            "/deepseek/healthz",
        ],
    }


@app.get("/healthz")
async def healthz(_: None = Depends(require_api_key)) -> dict:
    zai_client = _require_zai_client()
    try:
        data = await zai_client.healthcheck()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"ok": True, "upstream": data}


@app.get("/deepseek/healthz")
async def deepseek_healthz(_: None = Depends(require_api_key)) -> dict:
    ds_client = _require_deepseek_client()
    try:
        data = await ds_client.healthcheck()
    except DeepSeekUpstreamError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"ok": True, "upstream": data}


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models(_: None = Depends(require_api_key)) -> ModelsResponse:
    zai_client = _require_zai_client()
    try:
        upstream_models = await zai_client.list_models()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    models = [
        ModelCard(
            id=(model.get("openai") or {}).get("id") or model.get("id"),
            owned_by=model.get("owned_by") or "z.ai",
        )
        for model in upstream_models
    ]
    return ModelsResponse(data=models)


@app.get("/deepseek/v1/models", response_model=ModelsResponse)
async def list_deepseek_models(_: None = Depends(require_api_key)) -> ModelsResponse:
    ds_client = _require_deepseek_client()
    models = [
        ModelCard(id=model.id, owned_by=model.owned_by)
        for model in await ds_client.list_models()
    ]
    return ModelsResponse(data=models)


@app.post("/v1/chat/completions", response_model=ChatCompletionsResponse)
async def chat_completions(
    request: ChatCompletionsRequest,
    _: None = Depends(require_api_key),
) -> ChatCompletionsResponse:
    if request.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported by this proxy")

    zai_client = _require_zai_client()
    try:
        completion = await zai_client.complete(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ZaiUpstreamError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    usage = completion.usage or {}
    return ChatCompletionsResponse(
        id=completion.id,
        created=int(time.time()),
        model=completion.model,
        choices=[
            Choice(
                index=0,
                message=OpenAIMessage(role="assistant", content=completion.content),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        ),
    )


@app.post("/deepseek/v1/chat/completions", response_model=ChatCompletionsResponse)
async def deepseek_chat_completions(
    request: ChatCompletionsRequest,
    _: None = Depends(require_api_key),
) -> ChatCompletionsResponse:
    if request.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported by this proxy")

    ds_client = _require_deepseek_client()
    try:
        completion = await ds_client.complete(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DeepSeekUpstreamError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return _chat_completion_response(completion)


@app.post("/v1/responses", response_model=ResponsesResponse)
async def responses(
    request: ResponsesRequest,
    _: None = Depends(require_api_key),
) -> ResponsesResponse:
    if request.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported by this proxy")

    zai_client = _require_zai_client()
    try:
        chat_request = _responses_to_chat_request(request)
        completion = await zai_client.complete(chat_request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ZaiUpstreamError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    usage = completion.usage or {}
    return ResponsesResponse(
        id="resp_" + completion.id.removeprefix("chatcmpl-"),
        created_at=int(time.time()),
        model=completion.model,
        output=[
            ResponseOutputMessage(
                id="msg_" + completion.id.removeprefix("chatcmpl-"),
                content=[ResponseOutputText(text=completion.content)],
            )
        ],
        output_text=completion.content,
        usage=ResponseUsage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        ),
        metadata=request.metadata,
    )


@app.post("/deepseek/v1/responses", response_model=ResponsesResponse)
async def deepseek_responses(
    request: ResponsesRequest,
    _: None = Depends(require_api_key),
) -> ResponsesResponse:
    if request.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported by this proxy")

    ds_client = _require_deepseek_client()
    try:
        chat_request = _responses_to_chat_request(request)
        completion = await ds_client.complete(chat_request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DeepSeekUpstreamError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return _responses_response(completion, request)


@app.post("/zai/chat", response_model=ZaiChatResponse)
async def zai_chat(
    request: ZaiChatRequest,
    _: None = Depends(require_api_key),
) -> ZaiChatResponse:
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    zai_client = _require_zai_client()
    try:
        completion = await zai_client.complete_stateful(
            prompt=prompt,
            chat_id=request.chat_id,
            model=request.model,
            system=request.system,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ZaiUpstreamError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not completion.chat_id:
        raise HTTPException(status_code=502, detail="z.ai did not return a chat id")

    return ZaiChatResponse(
        id=completion.id,
        created=int(time.time()),
        chat_id=completion.chat_id,
        model=completion.model,
        answer=completion.content,
        usage=completion.usage or {},
        reasoning=completion.reasoning,
        new_chat=completion.created_chat,
        metadata=request.metadata,
    )


@app.delete("/zai/chat/{chat_id}", response_model=ZaiDeleteChatResponse)
async def delete_zai_chat(
    chat_id: str,
    _: None = Depends(require_api_key),
) -> ZaiDeleteChatResponse:
    zai_client = _require_zai_client()
    try:
        deleted = await zai_client.delete_chat(chat_id)
    except ZaiUpstreamError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ZaiDeleteChatResponse(chat_id=chat_id, deleted=deleted)


def _require_zai_client() -> ZaiClient:
    if client is None:
        raise HTTPException(status_code=503, detail="ZAI_TOKEN is not configured")
    return client


def _require_deepseek_client() -> DeepSeekClient:
    if deepseek_client is None:
        raise HTTPException(status_code=503, detail="DEEPSEEK_TOKEN is not configured")
    return deepseek_client


def _chat_completion_response(completion) -> ChatCompletionsResponse:
    usage = completion.usage or {}
    return ChatCompletionsResponse(
        id=completion.id,
        created=int(time.time()),
        model=completion.model,
        choices=[
            Choice(
                index=0,
                message=OpenAIMessage(role="assistant", content=completion.content),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        ),
    )


def _responses_response(completion, request: ResponsesRequest) -> ResponsesResponse:
    usage = completion.usage or {}
    return ResponsesResponse(
        id="resp_" + completion.id.removeprefix("chatcmpl-"),
        created_at=int(time.time()),
        model=completion.model,
        output=[
            ResponseOutputMessage(
                id="msg_" + completion.id.removeprefix("chatcmpl-"),
                content=[ResponseOutputText(text=completion.content)],
            )
        ],
        output_text=completion.content,
        usage=ResponseUsage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        ),
        metadata=request.metadata,
    )


def _responses_to_chat_request(request: ResponsesRequest) -> ChatCompletionsRequest:
    messages = _normalize_responses_input(request.input)
    if request.instructions:
        messages.insert(0, {"role": "system", "content": request.instructions})

    return ChatCompletionsRequest(
        model=request.model,
        messages=messages,
        max_tokens=request.max_output_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=False,
        user=request.user,
        metadata=request.metadata,
    )


def _normalize_responses_input(input_value: Any) -> list[dict[str, Any]]:
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]

    if not isinstance(input_value, list) or not input_value:
        raise ValueError("responses.input must be a string or a non-empty list")

    messages: list[dict[str, Any]] = []
    for item in input_value:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue

        if not isinstance(item, dict):
            raise ValueError("Unsupported responses input item")

        role = item.get("role")
        if item.get("type") == "message" and role:
            messages.append(
                {
                    "role": _normalize_role(role),
                    "content": _extract_responses_content(item.get("content")),
                }
            )
            continue

        if role:
            messages.append(
                {
                    "role": _normalize_role(role),
                    "content": _extract_responses_content(item.get("content")),
                }
            )
            continue

        messages.append(
            {
                "role": "user",
                "content": _extract_responses_content([item]),
            }
        )

    return messages


def _normalize_role(role: str) -> str:
    if role == "developer":
        return "system"
    if role not in {"system", "user", "assistant"}:
        raise ValueError(f"Unsupported role: {role}")
    return role


def _extract_responses_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("Unsupported responses content format")

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue

        if not isinstance(item, dict):
            raise ValueError("Unsupported responses content part")

        item_type = item.get("type")
        if item_type in {"input_text", "text", "output_text"}:
            parts.append(item.get("text") or "")
            continue

        raise ValueError(f"Unsupported responses content part type: {item_type}")

    return "\n".join(part for part in parts if part).strip()
