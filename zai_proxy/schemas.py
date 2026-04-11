from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MessageContentPart(BaseModel):
    type: str
    text: str | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[MessageContentPart] | None = None
    name: str | None = None


class ChatCompletionsRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    user: str | None = None
    metadata: dict[str, Any] | None = None


class ResponsesRequest(BaseModel):
    model: str | None = None
    input: Any
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    user: str | None = None
    metadata: dict[str, Any] | None = None


class OpenAIMessage(BaseModel):
    role: Literal["assistant"]
    content: str


class Choice(BaseModel):
    index: int = 0
    message: OpenAIMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionsResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "z.ai"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class ResponseOutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[dict[str, Any]] = Field(default_factory=list)


class ResponseOutputMessage(BaseModel):
    id: str
    type: Literal["message"] = "message"
    status: Literal["completed"] = "completed"
    role: Literal["assistant"] = "assistant"
    content: list[ResponseOutputText]


class ResponseUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ResponsesResponse(BaseModel):
    id: str
    object: Literal["response"] = "response"
    created_at: int
    status: Literal["completed"] = "completed"
    model: str
    output: list[ResponseOutputMessage]
    output_text: str
    usage: ResponseUsage
    error: None = None
    incomplete_details: None = None
    metadata: dict[str, Any] | None = None
