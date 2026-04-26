from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from .config import Settings
from .parser import ParsedCompletion, parse_sse_completion
from .schemas import ChatCompletionsRequest
from .utils import (
    build_signature,
    build_sorted_payload,
    extract_text_content,
    ms_now,
    new_id,
    parse_jwt_payload,
    unix_now,
)


@dataclass(slots=True)
class UpstreamCompletion:
    id: str
    model: str
    content: str
    usage: dict
    reasoning: str
    chat_id: str | None = None
    created_chat: bool = False


@dataclass(slots=True)
class ChatSeed:
    chat_id: str
    current_user_message_id: str
    current_user_message_parent_id: str | None


class ZaiUpstreamError(RuntimeError):
    def __init__(self, detail: str, *, status_code: int = 502) -> None:
        super().__init__(detail)
        self.status_code = status_code


class ZaiCapacityError(ZaiUpstreamError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail, status_code=503)


class ZaiClient:
    _capacity_retries = 3

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        jwt_payload = parse_jwt_payload(settings.zai_token)
        self.user_id = str(jwt_payload.get("id", ""))
        if not self.user_id:
            raise RuntimeError("Could not extract user id from ZAI_TOKEN")
        self._http = httpx.AsyncClient(timeout=120.0)
        self._models_cache: list[dict] | None = None
        self._request_lock = asyncio.Lock()
        self._next_request_time = 0.0

    async def close(self) -> None:
        await self._http.aclose()

    async def healthcheck(self) -> dict:
        res = await self._send(
            "GET",
            "https://chat.z.ai/api/v1/users/user/settings",
            headers=self._base_headers(),
        )
        res.raise_for_status()
        return res.json()

    async def list_models(self) -> list[dict]:
        if self._models_cache is not None:
            return self._models_cache
        res = await self._send(
            "GET",
            "https://chat.z.ai/api/models",
            headers=self._base_headers(),
        )
        res.raise_for_status()
        data = res.json().get("data", [])
        self._models_cache = data
        return data

    async def list_chats(self, *, page: int = 1, chat_type: str = "default") -> list[dict]:
        res = await self._send(
            "GET",
            f"https://chat.z.ai/api/v1/chats/?page={page}&type={chat_type}",
            headers=self._base_headers(),
        )
        res.raise_for_status()
        data = res.json()
        if not isinstance(data, list):
            raise RuntimeError("Unexpected chat list response from z.ai")
        return data

    async def delete_chat(self, chat_id: str) -> bool:
        res = await self._send(
            "DELETE",
            f"https://chat.z.ai/api/v1/chats/{chat_id}",
            headers=self._base_headers(),
        )
        self._raise_for_http_error(res)
        body = res.text.strip().lower()
        return body in {"true", '"true"'} or res.status_code == 200

    async def get_chat(self, chat_id: str) -> dict:
        res = await self._send(
            "GET",
            f"https://chat.z.ai/api/v1/chats/{chat_id}",
            headers=self._base_headers(),
        )
        self._raise_for_http_error(res)
        data = res.json()
        if not isinstance(data, dict):
            raise RuntimeError("Unexpected chat response from z.ai")
        return data

    async def complete(self, request: ChatCompletionsRequest) -> UpstreamCompletion:
        if any(message.role == "tool" for message in request.messages):
            raise ValueError("tool role messages are not supported by this proxy")

        raw_messages = [
            {"role": message.role, "content": extract_text_content(message.content)}
            for message in request.messages
        ]
        return await self._complete_turn(
            messages=raw_messages,
            model=request.model,
            params=self._build_params(request),
            chat_id=None,
            delete_after_response=self.settings.delete_chat_after_response,
        )

    async def complete_stateful(
        self,
        *,
        prompt: str,
        chat_id: str | None = None,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
    ) -> UpstreamCompletion:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return await self._complete_turn(
            messages=messages,
            model=model,
            params=self._build_params_dict(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            ),
            chat_id=chat_id,
            delete_after_response=False,
        )

    async def _complete_turn(
        self,
        *,
        messages: list[dict[str, str]],
        model: str | None,
        params: dict,
        chat_id: str | None,
        delete_after_response: bool,
    ) -> UpstreamCompletion:
        existing_chat = await self.get_chat(chat_id) if chat_id else None
        message_texts = self._normalize_messages_for_upstream(messages)

        if not message_texts:
            raise ValueError("At least one message is required")
        if message_texts[-1]["role"] != "user":
            raise ValueError("The last message must have role='user'")

        last_user = next((m["content"] for m in reversed(message_texts) if m["role"] == "user"), None)
        if not last_user:
            raise ValueError("At least one user message is required")

        model_id = await self._resolve_model(
            model or self._model_from_chat(existing_chat) or self.settings.default_model
        )
        model_info = await self._get_model_info(model_id)
        feature_bundle = self._build_feature_bundle(model_info)

        last_error: Exception | None = None
        for attempt in range(self._capacity_retries):
            active_chat_id: str | None = None
            created_chat = False
            success = False
            try:
                if existing_chat is not None:
                    chat_seed = self._seed_from_existing_chat(existing_chat)
                    completion_messages = self._messages_from_chat(existing_chat) + message_texts
                else:
                    chat_seed = await self._create_chat(
                        model_id=model_id,
                        messages=message_texts,
                        feature_bundle=feature_bundle,
                    )
                    completion_messages = message_texts
                    created_chat = True

                active_chat_id = chat_seed.chat_id

                timestamp_ms = str(ms_now())
                request_id = new_id()
                sorted_payload = build_sorted_payload(timestamp_ms, request_id, self.user_id)
                signature = build_signature(sorted_payload, last_user, timestamp_ms)
                url = "https://chat.z.ai/api/v2/chat/completions?" + self._build_query_params(
                    timestamp_ms=timestamp_ms,
                    request_id=request_id,
                )

                body = {
                    "stream": False,
                    "model": model_id,
                    "messages": completion_messages,
                    "signature_prompt": last_user,
                    "params": params,
                    "extra": feature_bundle["extra"],
                    "features": feature_bundle["completion_features"],
                    "variables": self._build_variables(),
                    "chat_id": chat_seed.chat_id,
                    "id": new_id(),
                    "current_user_message_id": chat_seed.current_user_message_id,
                    "current_user_message_parent_id": chat_seed.current_user_message_parent_id,
                    "background_tasks": {"title_generation": True, "tags_generation": True},
                }

                res = await self._send(
                    "POST",
                    url,
                    headers=self._base_headers(signature=signature),
                    json=body,
                )
                self._raise_for_http_error(res)

                parsed = parse_sse_completion(res.text)
                self._raise_upstream_error(parsed)
                if not parsed.answer:
                    raise ZaiUpstreamError("z.ai returned no answer content")

                success = True
                return UpstreamCompletion(
                    id="chatcmpl-" + new_id(),
                    model=model_id,
                    content=parsed.answer,
                    usage=parsed.usage,
                    reasoning=parsed.reasoning,
                    chat_id=chat_seed.chat_id,
                    created_chat=created_chat,
                )
            except ZaiCapacityError as exc:
                last_error = exc
                if attempt + 1 >= self._capacity_retries:
                    raise
                await asyncio.sleep(1.5 * (attempt + 1))
            except httpx.HTTPStatusError as exc:
                raise self._http_status_to_error(exc.response) from exc
            finally:
                should_delete_failed_new_chat = created_chat and not success and not delete_after_response
                if active_chat_id and (delete_after_response or should_delete_failed_new_chat):
                    with suppress(Exception):
                        await self.delete_chat(active_chat_id)

        raise ZaiUpstreamError(str(last_error or "Unknown upstream error"))

    async def _resolve_model(self, requested: str) -> str:
        models = await self.list_models()
        wanted = requested.strip()
        if not wanted:
            return self.settings.default_model

        for model in models:
            candidates = {
                model.get("id"),
                model.get("name"),
                ((model.get("openai") or {}).get("id")),
            }
            if wanted in candidates:
                return model["id"]

        return wanted

    async def _get_model_info(self, model_id: str) -> dict:
        models = await self.list_models()
        for model in models:
            if model.get("id") == model_id:
                return model
        return {}

    async def _create_chat(
        self,
        *,
        model_id: str,
        messages: list[dict[str, str]],
        feature_bundle: dict,
    ) -> ChatSeed:
        history_messages, current_id, current_user_message_id, current_user_message_parent_id = (
            self._build_history(messages, model_id)
        )
        body = {
            "chat": {
                "id": "",
                "title": "New Chat",
                "models": [model_id],
                "params": {},
                "history": {
                    "messages": history_messages,
                    "currentId": current_id,
                },
                "tags": [],
                "flags": [],
                "features": feature_bundle["chat_features"],
                "mcp_servers": [],
                "enable_thinking": self.settings.enable_thinking,
                "auto_web_search": self.settings.auto_web_search,
                "message_version": 1,
                "extra": feature_bundle["extra"],
                "timestamp": ms_now(),
                "type": "default",
            }
        }

        res = await self._send(
            "POST",
            "https://chat.z.ai/api/v1/chats/new",
            headers=self._base_headers(),
            json=body,
        )
        res.raise_for_status()
        data = res.json()
        chat_id = data.get("id")
        if not chat_id:
            raise RuntimeError("z.ai did not return a chat id")
        return ChatSeed(
            chat_id=str(chat_id),
            current_user_message_id=current_user_message_id,
            current_user_message_parent_id=current_user_message_parent_id,
        )

    @staticmethod
    def _build_history(
        messages: list[dict[str, str]],
        model_id: str,
    ) -> tuple[dict[str, dict], str, str, str | None]:
        if not messages:
            raise ValueError("At least one message is required")

        history_messages: dict[str, dict] = {}
        previous_id: str | None = None
        current_id: str | None = None
        current_user_message_id: str | None = None
        current_user_message_parent_id: str | None = None
        timestamp = unix_now()

        for message in messages:
            message_id = new_id()
            history_messages[message_id] = {
                "id": message_id,
                "parentId": previous_id,
                "childrenIds": [],
                "role": message["role"],
                "content": message["content"],
                "timestamp": timestamp,
                "models": [model_id],
            }
            if previous_id is not None:
                history_messages[previous_id]["childrenIds"].append(message_id)

            if message["role"] == "user":
                current_user_message_id = message_id
                current_user_message_parent_id = previous_id

            previous_id = message_id
            current_id = message_id
            timestamp += 1

        if current_id is None or current_user_message_id is None:
            raise ValueError("Message history is missing a user message")

        return history_messages, current_id, current_user_message_id, current_user_message_parent_id

    @staticmethod
    def _model_from_chat(chat_data: dict | None) -> str | None:
        if not chat_data:
            return None

        chat = chat_data.get("chat") or {}
        models = chat.get("models") or (chat_data.get("meta") or {}).get("models") or []
        if isinstance(models, list) and models:
            return str(models[0])
        return None

    @staticmethod
    def _seed_from_existing_chat(chat_data: dict) -> ChatSeed:
        chat_id = str(chat_data.get("id") or (chat_data.get("chat") or {}).get("id") or "").strip()
        history = ((chat_data.get("chat") or {}).get("history") or {})
        parent_id = history.get("currentId")

        if not chat_id:
            raise ValueError("Existing chat response is missing chat id")
        if not parent_id:
            raise ValueError("Existing chat has no current message to continue from")

        return ChatSeed(
            chat_id=chat_id,
            current_user_message_id=new_id(),
            current_user_message_parent_id=str(parent_id),
        )

    @staticmethod
    def _messages_from_chat(chat_data: dict) -> list[dict[str, str]]:
        history = ((chat_data.get("chat") or {}).get("history") or {})
        messages_by_id = history.get("messages") or {}
        current_id = history.get("currentId")
        chain: list[dict[str, str]] = []
        seen: set[str] = set()

        while current_id and current_id not in seen:
            seen.add(str(current_id))
            message = messages_by_id.get(current_id)
            if not isinstance(message, dict):
                break

            role = str(message.get("role") or "")
            content = str(message.get("content") or "").strip()
            if role in {"system", "user", "assistant"} and content:
                chain.append({"role": role, "content": content})

            current_id = message.get("parentId")

        chain.reverse()
        return chain

    def _build_feature_bundle(self, model_info: dict) -> dict:
        caps = (((model_info.get("info") or {}).get("meta") or {}).get("capabilities") or {})

        extra = {
            "vlm_tools_enable": bool(caps.get("vlm_tools_enable", False)),
            "vlm_web_search_enable": bool(caps.get("vlm_web_search_enable", False)),
            "vlm_website_mode": bool(caps.get("vlm_website_mode", False)),
        }

        chat_features: list[dict] = []
        if caps.get("web_search"):
            chat_features.append({"type": "web_search", "server": "web_search", "status": "selected"})
        if any(extra.values()):
            chat_features.append({"type": "vlm-tools", "server": "vlm-tools", "status": "selected"})

        completion_features = {
            "image_generation": False,
            "web_search": False,
            "auto_web_search": self.settings.auto_web_search,
            "preview_mode": True,
            "flags": [],
            **extra,
            "enable_thinking": self.settings.enable_thinking,
        }

        return {
            "extra": extra,
            "chat_features": chat_features,
            "completion_features": completion_features,
        }

    @staticmethod
    def _normalize_messages_for_upstream(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        pending_system_parts: list[str] = []

        for message in messages:
            role = message["role"]
            content = message["content"].strip()
            if not content:
                continue

            if role == "system":
                pending_system_parts.append(content)
                continue

            if role == "user" and pending_system_parts:
                content = ZaiClient._render_instruction_block(pending_system_parts, content)
                pending_system_parts = []

            normalized.append({"role": role, "content": content})

        if pending_system_parts:
            raise ValueError("System messages must be followed by a user message")

        return normalized

    @staticmethod
    def _render_instruction_block(instructions: list[str], user_content: str) -> str:
        joined = "\n\n".join(part for part in instructions if part.strip())
        if not joined:
            return user_content
        return (
            "You must follow these instructions for this conversation.\n"
            "<system>\n"
            f"{joined}\n"
            "</system>\n\n"
            "Respond to the user message below.\n"
            "<user>\n"
            f"{user_content}\n"
            "</user>"
        )

    def _build_variables(self) -> dict[str, str]:
        now = datetime.now(ZoneInfo(self.settings.context_timezone))
        return {
            "{{USER_NAME}}": self.settings.user_name,
            "{{USER_LOCATION}}": self.settings.user_location,
            "{{CURRENT_DATETIME}}": now.strftime("%Y-%m-%d %H:%M:%S"),
            "{{CURRENT_DATE}}": now.strftime("%Y-%m-%d"),
            "{{CURRENT_TIME}}": now.strftime("%H:%M:%S"),
            "{{CURRENT_WEEKDAY}}": now.strftime("%A"),
            "{{CURRENT_TIMEZONE}}": self.settings.context_timezone,
            "{{USER_LANGUAGE}}": "en-US",
        }

    def _build_params(self, request: ChatCompletionsRequest) -> dict:
        return self._build_params_dict(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
        )

    @staticmethod
    def _build_params_dict(
        *,
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        stop: str | list[str] | None,
    ) -> dict:
        params: dict = {}
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        if stop is not None:
            params["stop"] = stop
        return params

    def _build_query_params(self, *, timestamp_ms: str, request_id: str) -> str:
        now = datetime.now().astimezone()
        params = {
            "timestamp": timestamp_ms,
            "requestId": request_id,
            "user_id": self.user_id,
            "version": "0.0.1",
            "platform": "web",
            "token": self.settings.zai_token,
            "user_agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
            ),
            "language": "en-US",
            "languages": "en-US,tr-TR,tr,en,zh-TW,zh-CN,de",
            "timezone": "Europe/Vienna",
            "cookie_enabled": "true",
            "screen_width": "2056",
            "screen_height": "1329",
            "screen_resolution": "2056x1329",
            "viewport_height": "1061",
            "viewport_width": "2056",
            "viewport_size": "2056x1061",
            "color_depth": "30",
            "pixel_ratio": "2",
            "current_url": "https://chat.z.ai/",
            "pathname": "/",
            "search": "",
            "hash": "",
            "host": "chat.z.ai",
            "hostname": "chat.z.ai",
            "protocol": "https:",
            "referrer": "",
            "title": "Z.ai - Free AI Chatbot & Agent powered by GLM-5.1 & GLM-5",
            "timezone_offset": str(int(-(now.utcoffset().total_seconds() / 60)) if now.utcoffset() else 0),
            "local_time": now.astimezone(ZoneInfo("UTC")).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "utc_time": now.strftime("%a, %d %b %Y %H:%M:%S GMT"),
            "is_mobile": "false",
            "is_touch": "false",
            "max_touch_points": "0",
            "browser_name": "Chrome",
            "os_name": "Mac OS",
            "signature_timestamp": timestamp_ms,
        }
        return str(httpx.QueryParams(params))

    async def _send(self, method: str, url: str, **kwargs) -> httpx.Response:
        async with self._request_lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            delay = self._next_request_time - now
            if delay > 0:
                await asyncio.sleep(delay)

            response = await self._http.request(method, url, **kwargs)

            interval = max(0.0, self.settings.min_request_interval_ms / 1000)
            self._next_request_time = loop.time() + interval
            return response

    def _base_headers(self, *, signature: str | None = None) -> dict[str, str]:
        headers = {
            "authorization": f"Bearer {self.settings.zai_token}",
            "content-type": "application/json",
            "accept": "application/json",
            "accept-language": "en-US",
            "user-agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
            ),
        }
        if signature:
            headers["x-fe-version"] = self.settings.fe_version
            headers["x-signature"] = signature
        return headers

    @staticmethod
    def _raise_for_http_error(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ZaiClient._http_status_to_error(response) from exc

    @staticmethod
    def _http_status_to_error(response: httpx.Response) -> ZaiUpstreamError:
        detail = ""
        try:
            payload = response.json()
            if isinstance(payload, dict):
                detail = str(
                    payload.get("detail")
                    or payload.get("message")
                    or payload.get("error")
                    or ""
                )
        except Exception:
            detail = response.text.strip()

        detail = detail or f"Upstream HTTP {response.status_code}"
        lowered = detail.lower()
        if response.status_code == 405 and "potential threats" in lowered:
            return ZaiUpstreamError(
                "z.ai blocked the current IP temporarily; slow down request rate and retry later",
                status_code=503,
            )
        if response.status_code in {401, 403}:
            return ZaiUpstreamError(f"z.ai authentication failed: {detail}", status_code=502)
        if response.status_code == 429:
            return ZaiCapacityError(detail)
        if response.status_code >= 500:
            return ZaiUpstreamError(f"z.ai upstream error: {detail}", status_code=503)
        return ZaiUpstreamError(detail, status_code=502)

    @staticmethod
    def _raise_upstream_error(parsed: ParsedCompletion) -> None:
        if parsed.error:
            detail = parsed.error.get("detail") or parsed.error.get("code") or "Unknown upstream error"
            detail = str(detail)
            if "at capacity" in detail.lower():
                raise ZaiCapacityError(detail)
            raise ZaiUpstreamError(detail)
