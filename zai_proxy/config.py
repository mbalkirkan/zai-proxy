from __future__ import annotations

import os
import json
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(slots=True)
class Settings:
    zai_token: str | None
    default_model: str
    fe_version: str
    proxy_api_key: str | None
    delete_chat_after_response: bool
    min_request_interval_ms: int
    enable_thinking: bool
    auto_web_search: bool
    deepseek_token: str | None
    deepseek_cookie: str | None
    deepseek_default_model: str
    deepseek_delete_chat_after_response: bool
    deepseek_min_request_interval_ms: int
    deepseek_thinking_enabled: bool
    deepseek_search_enabled: bool
    deepseek_client_version: str
    deepseek_app_version: str
    copilot_token: str | None
    copilot_default_model: str
    copilot_delete_thread_after_response: bool
    copilot_min_request_interval_ms: int
    copilot_api_base: str
    copilot_api_version: str
    user_name: str
    user_location: str
    context_timezone: str
    host: str
    port: int

    @classmethod
    def from_env(cls) -> "Settings":
        token = os.getenv("ZAI_TOKEN", "").strip() or None
        deepseek_token = _normalize_deepseek_token(os.getenv("DEEPSEEK_TOKEN", ""))
        copilot_token = _normalize_copilot_token(os.getenv("COPILOT_TOKEN", ""))
        if not token and not deepseek_token and not copilot_token:
            raise RuntimeError("ZAI_TOKEN, DEEPSEEK_TOKEN, or COPILOT_TOKEN is required")

        return cls(
            zai_token=token,
            default_model=os.getenv("ZAI_MODEL", "GLM-5-Turbo").strip(),
            fe_version=os.getenv("ZAI_FE_VERSION", "prod-fe-1.1.2").strip(),
            proxy_api_key=(
                os.getenv("PROXY_BEARER_TOKEN", "").strip()
                or os.getenv("ZAI_PROXY_API_KEY", "").strip()
                or None
            ),
            delete_chat_after_response=_as_bool(
                os.getenv("ZAI_DELETE_CHAT_AFTER_RESPONSE", "true")
            ),
            min_request_interval_ms=max(0, int(os.getenv("ZAI_MIN_REQUEST_INTERVAL_MS", "2000"))),
            enable_thinking=_as_bool(os.getenv("ZAI_ENABLE_THINKING", "false")),
            auto_web_search=_as_bool(os.getenv("ZAI_AUTO_WEB_SEARCH", "false")),
            deepseek_token=deepseek_token,
            deepseek_cookie=os.getenv("DEEPSEEK_COOKIE", "").strip() or None,
            deepseek_default_model=os.getenv("DEEPSEEK_MODEL", "default").strip() or "default",
            deepseek_delete_chat_after_response=_as_bool(
                os.getenv("DEEPSEEK_DELETE_CHAT_AFTER_RESPONSE", "true")
            ),
            deepseek_min_request_interval_ms=max(
                0,
                int(os.getenv("DEEPSEEK_MIN_REQUEST_INTERVAL_MS", "500")),
            ),
            deepseek_thinking_enabled=_as_bool(
                os.getenv("DEEPSEEK_THINKING_ENABLED", "true")
            ),
            deepseek_search_enabled=_as_bool(os.getenv("DEEPSEEK_SEARCH_ENABLED", "false")),
            deepseek_client_version=os.getenv("DEEPSEEK_CLIENT_VERSION", "1.8.0").strip(),
            deepseek_app_version=os.getenv("DEEPSEEK_APP_VERSION", "20241129.1").strip(),
            copilot_token=copilot_token,
            copilot_default_model=(
                os.getenv("COPILOT_MODEL", "gemini-3.1-pro-preview").strip()
                or "gemini-3.1-pro-preview"
            ),
            copilot_delete_thread_after_response=_as_bool(
                os.getenv("COPILOT_DELETE_THREAD_AFTER_RESPONSE", "true")
            ),
            copilot_min_request_interval_ms=max(
                0,
                int(os.getenv("COPILOT_MIN_REQUEST_INTERVAL_MS", "500")),
            ),
            copilot_api_base=(
                os.getenv("COPILOT_API_BASE", "https://api.individual.githubcopilot.com").strip()
                or "https://api.individual.githubcopilot.com"
            ).rstrip("/"),
            copilot_api_version=(
                os.getenv("COPILOT_API_VERSION", "2025-05-01").strip()
                or "2025-05-01"
            ),
            user_name=os.getenv("ZAI_USER_NAME", "").strip(),
            user_location=os.getenv("ZAI_USER_LOCATION", "Unknown").strip() or "Unknown",
            context_timezone=os.getenv("ZAI_CONTEXT_TIMEZONE", "Europe/Istanbul").strip(),
            host=os.getenv("HOST", "0.0.0.0").strip(),
            port=int(os.getenv("PORT", "8000")),
        )


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_deepseek_token(value: str) -> str | None:
    token = value.strip()
    if not token:
        return None

    if token.startswith("Bearer "):
        token = token.removeprefix("Bearer ").strip()

    if token.startswith("{"):
        try:
            parsed = json.loads(token)
        except json.JSONDecodeError:
            return token
        if isinstance(parsed, dict):
            candidate = str(parsed.get("value") or "").strip()
            return candidate or None

    return token


def _normalize_copilot_token(value: str) -> str | None:
    token = value.strip()
    if not token:
        return None

    if token.startswith("GitHub-Bearer "):
        token = token.removeprefix("GitHub-Bearer ").strip()
    elif token.startswith("Bearer "):
        token = token.removeprefix("Bearer ").strip()

    return token or None
