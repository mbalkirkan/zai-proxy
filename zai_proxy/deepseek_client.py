from __future__ import annotations

import asyncio
import base64
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
import json
from zoneinfo import ZoneInfo

import httpx

from .client import UpstreamCompletion
from .config import Settings
from .deepseek_pow import solve_deepseek_pow
from .parser import ParsedCompletion, parse_deepseek_completion
from .schemas import ChatCompletionsRequest
from .utils import extract_text_content, new_id


_COMPLETION_PATH = "/api/v0/chat/completion"
_BASE_URL = "https://chat.deepseek.com"


@dataclass(frozen=True, slots=True)
class DeepSeekModel:
    id: str
    owned_by: str = "deepseek"


class DeepSeekUpstreamError(RuntimeError):
    def __init__(self, detail: str, *, status_code: int = 502) -> None:
        super().__init__(detail)
        self.status_code = status_code


class DeepSeekClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.deepseek_token:
            raise RuntimeError("DEEPSEEK_TOKEN is required")

        self.settings = settings
        self.token = settings.deepseek_token
        self._http = httpx.AsyncClient(timeout=180.0)
        self._request_lock = asyncio.Lock()
        self._next_request_time = 0.0

    async def close(self) -> None:
        await self._http.aclose()

    async def healthcheck(self) -> dict:
        res = await self._send(
            "GET",
            f"{_BASE_URL}/api/v0/users/current",
            headers=self._base_headers(),
        )
        self._raise_for_http_error(res)
        return res.json()

    async def list_models(self) -> list[DeepSeekModel]:
        return [
            DeepSeekModel(id="deepseek-chat"),
            DeepSeekModel(id="deepseek-reasoner"),
            DeepSeekModel(id="default"),
            DeepSeekModel(id="expert"),
        ]

    async def delete_chat(self, chat_id: str) -> bool:
        res = await self._send(
            "POST",
            f"{_BASE_URL}/api/v0/chat_session/delete",
            headers=self._base_headers(),
            json={"chat_session_id": chat_id},
        )
        self._raise_for_http_error(res)
        data = res.json()
        return data.get("code") == 0 and (data.get("data") or {}).get("biz_code") == 0

    async def complete(self, request: ChatCompletionsRequest) -> UpstreamCompletion:
        if any(message.role == "tool" for message in request.messages):
            raise ValueError("tool role messages are not supported by this proxy")

        prompt = self._render_prompt(
            [
                {"role": message.role, "content": extract_text_content(message.content)}
                for message in request.messages
            ]
        )
        model_type = self._resolve_model(request.model or self.settings.deepseek_default_model)

        chat_id: str | None = None
        success = False
        try:
            chat_id = await self._create_chat()
            pow_response = await self._create_pow_response(_COMPLETION_PATH)
            body = {
                "chat_session_id": chat_id,
                "parent_message_id": None,
                "model_type": model_type,
                "prompt": prompt,
                "ref_file_ids": [],
                "thinking_enabled": self.settings.deepseek_thinking_enabled,
                "search_enabled": self.settings.deepseek_search_enabled,
                "preempt": False,
            }

            res = await self._send(
                "POST",
                f"{_BASE_URL}{_COMPLETION_PATH}",
                headers=self._base_headers(pow_response=pow_response),
                json=body,
            )
            self._raise_for_http_error(res)

            if "application/json" in res.headers.get("content-type", ""):
                raise self._api_payload_to_error(res.json())

            parsed = parse_deepseek_completion(res.text)
            self._raise_upstream_error(parsed)
            if not parsed.answer:
                raise DeepSeekUpstreamError("DeepSeek returned no answer content")

            success = True
            return UpstreamCompletion(
                id="chatcmpl-" + new_id(),
                model=request.model or model_type,
                content=parsed.answer,
                usage=parsed.usage,
                reasoning=parsed.reasoning,
                chat_id=chat_id,
                created_chat=True,
            )
        finally:
            should_delete_failed_chat = chat_id and not success
            should_delete_success_chat = chat_id and self.settings.deepseek_delete_chat_after_response
            if should_delete_failed_chat or should_delete_success_chat:
                with suppress(Exception):
                    await self.delete_chat(chat_id)

    async def _create_chat(self) -> str:
        res = await self._send(
            "POST",
            f"{_BASE_URL}/api/v0/chat_session/create",
            headers=self._base_headers(),
            json={},
        )
        self._raise_for_http_error(res)
        data = self._unwrap_biz_data(res)
        chat_id = ((data.get("chat_session") or {}).get("id") or "").strip()
        if not chat_id:
            raise DeepSeekUpstreamError("DeepSeek did not return a chat session id")
        return chat_id

    async def _create_pow_response(self, target_path: str) -> str:
        res = await self._send(
            "POST",
            f"{_BASE_URL}/api/v0/chat/create_pow_challenge",
            headers=self._base_headers(),
            json={"target_path": target_path},
        )
        self._raise_for_http_error(res)
        data = self._unwrap_biz_data(res)
        challenge = data.get("challenge")
        if not isinstance(challenge, dict):
            raise DeepSeekUpstreamError("DeepSeek did not return a PoW challenge")

        answer = solve_deepseek_pow(challenge)
        payload = json.dumps(answer.as_header_payload(), separators=(",", ":"))
        return base64.b64encode(payload.encode("utf-8")).decode("ascii")

    async def _send(self, method: str, url: str, **kwargs) -> httpx.Response:
        async with self._request_lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            delay = self._next_request_time - now
            if delay > 0:
                await asyncio.sleep(delay)

            response = await self._http.request(method, url, **kwargs)

            interval = max(0.0, self.settings.deepseek_min_request_interval_ms / 1000)
            self._next_request_time = loop.time() + interval
            return response

    def _base_headers(self, *, pow_response: str | None = None) -> dict[str, str]:
        headers = {
            "authorization": f"Bearer {self.token}",
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9,tr-TR;q=0.8,tr;q=0.7",
            "content-type": "application/json",
            "origin": _BASE_URL,
            "referer": f"{_BASE_URL}/",
            "user-agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
            ),
            "x-app-version": self.settings.deepseek_app_version,
            "x-client-locale": "en_US",
            "x-client-platform": "web",
            "x-client-timezone-offset": str(self._timezone_offset_seconds()),
            "x-client-version": self.settings.deepseek_client_version,
        }
        if self.settings.deepseek_cookie:
            headers["cookie"] = self.settings.deepseek_cookie
        if pow_response:
            headers["x-ds-pow-response"] = pow_response
        return headers

    def _timezone_offset_seconds(self) -> int:
        now = datetime.now(ZoneInfo(self.settings.context_timezone))
        offset = now.utcoffset()
        return int(offset.total_seconds()) if offset else 0

    @staticmethod
    def _resolve_model(requested: str) -> str:
        model = requested.strip()
        normalized = model.lower()
        aliases = {
            "deepseek-chat": "default",
            "deepseek-v3": "default",
            "instant": "default",
            "default": "default",
            "deepseek-reasoner": "expert",
            "deepseek-r1": "expert",
            "reasoner": "expert",
            "expert": "expert",
        }
        return aliases.get(normalized, model or "default")

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
    def _unwrap_biz_data(response: httpx.Response) -> dict:
        payload = response.json()
        if payload.get("code") != 0:
            raise DeepSeekClient._api_payload_to_error(payload)

        data = payload.get("data") or {}
        if data.get("biz_code") != 0:
            detail = data.get("biz_msg") or payload.get("msg") or "DeepSeek business error"
            raise DeepSeekUpstreamError(str(detail))

        biz_data = data.get("biz_data") or {}
        if not isinstance(biz_data, dict):
            raise DeepSeekUpstreamError("Unexpected DeepSeek response shape")
        return biz_data

    @staticmethod
    def _raise_for_http_error(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise DeepSeekClient._http_status_to_error(response) from exc

    @staticmethod
    def _http_status_to_error(response: httpx.Response) -> DeepSeekUpstreamError:
        detail = ""
        try:
            payload = response.json()
            if isinstance(payload, dict):
                return DeepSeekClient._api_payload_to_error(payload)
        except Exception:
            detail = response.text.strip()

        detail = detail or f"Upstream HTTP {response.status_code}"
        if response.status_code in {401, 403}:
            return DeepSeekUpstreamError(
                f"DeepSeek authentication or browser challenge failed: {detail}",
                status_code=502,
            )
        if response.status_code == 429:
            return DeepSeekUpstreamError(
                "DeepSeek rate limited this IP/session; increase request spacing and retry",
                status_code=503,
            )
        if response.status_code >= 500:
            return DeepSeekUpstreamError(f"DeepSeek upstream error: {detail}", status_code=503)
        return DeepSeekUpstreamError(detail, status_code=502)

    @staticmethod
    def _api_payload_to_error(payload: dict) -> DeepSeekUpstreamError:
        code = payload.get("code")
        msg = payload.get("msg") or payload.get("message") or "DeepSeek API error"
        if code in {40002, 40003}:
            return DeepSeekUpstreamError(f"DeepSeek authentication failed: {msg}", status_code=502)
        if code in {40300, 40301}:
            return DeepSeekUpstreamError(
                "DeepSeek rejected the PoW header; refresh DEEPSEEK_TOKEN/DEEPSEEK_COOKIE",
                status_code=502,
            )
        if code == 40029:
            return DeepSeekUpstreamError(
                "DeepSeek rejected this IP for the current session",
                status_code=503,
            )
        return DeepSeekUpstreamError(f"DeepSeek API error {code}: {msg}", status_code=502)

    @staticmethod
    def _raise_upstream_error(parsed: ParsedCompletion) -> None:
        if parsed.error:
            detail = parsed.error.get("detail") or parsed.error.get("code") or "Unknown upstream error"
            raise DeepSeekUpstreamError(str(detail))
