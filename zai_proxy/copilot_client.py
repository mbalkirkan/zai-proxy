from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import json

import httpx

from .client import UpstreamCompletion
from .config import Settings
from .parser import ParsedCompletion, parse_copilot_completion
from .schemas import ChatCompletionsRequest
from .utils import extract_text_content, new_id


@dataclass(frozen=True, slots=True)
class CopilotModel:
    id: str
    owned_by: str
    aliases: tuple[str, ...] = ()


_COPILOT_MODELS: tuple[CopilotModel, ...] = (
    CopilotModel("gemini-3.1-pro-preview", "google", ("gemini-3.1-pro", "gemini 3.1 pro")),
    CopilotModel("gpt-5.2-codex", "openai", ("gpt-5.2 codex", "gpt 5.2 codex")),
    CopilotModel("gpt-5.4-mini", "openai", ("gpt-5.4 mini", "gpt 5.4 mini")),
    CopilotModel("gpt-5-mini", "openai", ("gpt-5 mini", "gpt 5 mini")),
    CopilotModel("grok-code-fast-1", "xai", ("grok code fast 1", "grok-code")),
    CopilotModel("claude-haiku-4.5", "anthropic", ("claude haiku 4.5",)),
    CopilotModel("gemini-3-flash-preview", "google", ("gemini-3-flash", "gemini 3 flash")),
    CopilotModel("gemini-2.5-pro", "google", ("gemini 2.5 pro",)),
    CopilotModel("gpt-5.2", "openai", ("gpt 5.2",)),
    CopilotModel("gpt-4.1", "openai", ("gpt 4.1", "gpt-4.1-2025-04-14")),
    CopilotModel("gpt-4o", "openai", ("gpt 4o", "gpt-4o-2024-11-20")),
)


class CopilotUpstreamError(RuntimeError):
    def __init__(self, detail: str, *, status_code: int = 502) -> None:
        super().__init__(detail)
        self.status_code = status_code


class CopilotClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.copilot_token:
            raise RuntimeError("COPILOT_TOKEN is required")

        self.settings = settings
        self.token = settings.copilot_token
        self._http = httpx.AsyncClient(timeout=180.0)
        self._completion_lock = asyncio.Lock()
        self._next_completion_time = 0.0

    async def close(self) -> None:
        await self._http.aclose()

    async def healthcheck(self) -> dict:
        res = await self._send(
            "GET",
            f"{self.settings.copilot_api_base}/agents/tasks?page=1&archived=false&include_counts=false&per_page=1",
            headers=self._base_headers(),
        )
        self._raise_for_http_error(res)
        return {"status_code": res.status_code}

    async def list_models(self) -> list[CopilotModel]:
        return list(_COPILOT_MODELS)

    async def delete_thread(self, thread_id: str) -> bool:
        res = await self._send(
            "DELETE",
            f"{self.settings.copilot_api_base}/github/chat/threads/{thread_id}",
            headers=self._base_headers(),
        )
        self._raise_for_http_error(res)
        return res.status_code in {200, 202, 204}

    async def complete(self, request: ChatCompletionsRequest) -> UpstreamCompletion:
        async with self._completion_lock:
            await self._wait_for_completion_slot()
            try:
                return await self._complete(request)
            finally:
                interval = max(0.0, self.settings.copilot_min_request_interval_ms / 1000)
                self._next_completion_time = asyncio.get_running_loop().time() + interval

    async def _complete(self, request: ChatCompletionsRequest) -> UpstreamCompletion:
        if any(message.role == "tool" for message in request.messages):
            raise ValueError("tool role messages are not supported by this proxy")

        prompt = self._render_prompt(
            [
                {"role": message.role, "content": extract_text_content(message.content)}
                for message in request.messages
            ]
        )
        model_id = self._resolve_model(request.model or self.settings.copilot_default_model)

        thread_id: str | None = None
        success = False
        try:
            thread_id = await self._create_thread()
            response_message_id = new_id()
            body = {
                "responseMessageID": response_message_id,
                "content": prompt,
                "intent": "conversation",
                "references": [],
                "context": [],
                "currentURL": "https://github.com/copilot",
                "streaming": True,
                "confirmations": [],
                "customInstructions": [],
                "model": model_id,
                "mode": "immersive",
                "parentMessageID": "root",
                "mediaContent": [],
                "skillOptions": {"deepCodeSearch": False},
                "requestTrace": False,
            }

            res = await self._send(
                "POST",
                f"{self.settings.copilot_api_base}/github/chat/threads/{thread_id}/messages?",
                headers=self._base_headers(content_type="text/event-stream"),
                content=json.dumps(body, separators=(",", ":")),
            )
            self._raise_for_http_error(res)

            if "data:" not in res.text and "application/json" in res.headers.get("content-type", ""):
                raise self._api_payload_to_error(res.json())

            parsed = parse_copilot_completion(res.text)
            self._raise_upstream_error(parsed)
            if not parsed.answer:
                raise CopilotUpstreamError("Copilot returned no answer content")

            success = True
            return UpstreamCompletion(
                id="chatcmpl-" + new_id(),
                model=model_id,
                content=parsed.answer,
                usage=parsed.usage,
                reasoning=parsed.reasoning,
                chat_id=thread_id,
                created_chat=True,
            )
        finally:
            should_delete_failed_thread = thread_id and not success
            should_delete_success_thread = (
                thread_id and self.settings.copilot_delete_thread_after_response
            )
            if should_delete_failed_thread or should_delete_success_thread:
                with suppress(Exception):
                    await self.delete_thread(thread_id)

    async def _create_thread(self) -> str:
        res = await self._send(
            "POST",
            f"{self.settings.copilot_api_base}/github/chat/threads",
            headers=self._base_headers(content_type="text/plain;charset=UTF-8"),
            content="{}",
        )
        self._raise_for_http_error(res)
        data = res.json()
        thread_id = str(data.get("thread_id") or (data.get("thread") or {}).get("id") or "").strip()
        if not thread_id:
            raise CopilotUpstreamError("Copilot did not return a thread id")
        return thread_id

    async def _send(self, method: str, url: str, **kwargs) -> httpx.Response:
        return await self._http.request(method, url, **kwargs)

    async def _wait_for_completion_slot(self) -> None:
        loop = asyncio.get_running_loop()
        delay = self._next_completion_time - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)

    def _base_headers(self, *, content_type: str = "application/json") -> dict[str, str]:
        return {
            "authorization": f"GitHub-Bearer {self.token}",
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9,tr-TR;q=0.8,tr;q=0.7",
            "content-type": content_type,
            "copilot-integration-id": "copilot-chat",
            "origin": "https://github.com",
            "referer": "https://github.com/",
            "user-agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
            ),
            "x-github-api-version": self.settings.copilot_api_version,
        }

    @staticmethod
    def _resolve_model(requested: str) -> str:
        value = requested.strip()
        if not value:
            return "gemini-3.1-pro-preview"

        normalized = _normalize_model_name(value)
        for model in _COPILOT_MODELS:
            candidates = (model.id, *model.aliases)
            if normalized in {_normalize_model_name(candidate) for candidate in candidates}:
                return model.id

        return value

    @staticmethod
    def _render_prompt(messages: list[dict[str, str]]) -> str:
        normalized = [
            {"role": message["role"], "content": message["content"].strip()}
            for message in messages
            if message["content"].strip()
        ]
        if not normalized:
            raise ValueError("At least one message is required")
        if normalized[-1]["role"] != "user":
            raise ValueError("The last message must have role='user'")

        system_parts = [message["content"] for message in normalized if message["role"] == "system"]
        conversation = [message for message in normalized if message["role"] != "system"]

        if len(conversation) == 1 and not system_parts:
            return conversation[0]["content"]

        parts: list[str] = []
        if system_parts:
            parts.append("System instructions:\n" + "\n\n".join(system_parts))

        prior = conversation[:-1]
        if prior:
            rendered = []
            for message in prior:
                label = "User" if message["role"] == "user" else "Assistant"
                rendered.append(f"{label}: {message['content']}")
            parts.append("Conversation so far:\n" + "\n\n".join(rendered))

        parts.append("Current user message:\n" + conversation[-1]["content"])
        return "\n\n".join(parts).strip()

    @staticmethod
    def _raise_for_http_error(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise CopilotClient._http_status_to_error(response) from exc

    @staticmethod
    def _http_status_to_error(response: httpx.Response) -> CopilotUpstreamError:
        detail = ""
        try:
            payload = response.json()
            if isinstance(payload, dict):
                return CopilotClient._api_payload_to_error(payload)
        except Exception:
            detail = response.text.strip()

        detail = detail or f"Upstream HTTP {response.status_code}"
        if response.status_code in {401, 403}:
            return CopilotUpstreamError(
                "Copilot authentication failed; refresh COPILOT_TOKEN from the browser request",
                status_code=502,
            )
        if response.status_code == 429:
            return CopilotUpstreamError(
                "Copilot rate limited this token/session; increase request spacing and retry",
                status_code=503,
            )
        if response.status_code >= 500:
            return CopilotUpstreamError(f"Copilot upstream error: {detail}", status_code=503)
        return CopilotUpstreamError(detail, status_code=502)

    @staticmethod
    def _api_payload_to_error(payload: dict) -> CopilotUpstreamError:
        detail = (
            payload.get("message")
            or payload.get("error")
            or payload.get("detail")
            or "Copilot API error"
        )
        return CopilotUpstreamError(str(detail), status_code=502)

    @staticmethod
    def _raise_upstream_error(parsed: ParsedCompletion) -> None:
        if parsed.error:
            detail = parsed.error.get("detail") or parsed.error.get("code") or "Unknown upstream error"
            raise CopilotUpstreamError(str(detail))


def _normalize_model_name(value: str) -> str:
    return " ".join(value.lower().replace("_", "-").split())
