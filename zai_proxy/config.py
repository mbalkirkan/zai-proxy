from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(slots=True)
class Settings:
    zai_token: str
    default_model: str
    fe_version: str
    proxy_api_key: str | None
    delete_chat_after_response: bool
    min_request_interval_ms: int
    enable_thinking: bool
    auto_web_search: bool
    user_name: str
    user_location: str
    context_timezone: str
    host: str
    port: int

    @classmethod
    def from_env(cls) -> "Settings":
        token = os.getenv("ZAI_TOKEN", "").strip()
        if not token:
            raise RuntimeError("ZAI_TOKEN is required")

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
            user_name=os.getenv("ZAI_USER_NAME", "").strip(),
            user_location=os.getenv("ZAI_USER_LOCATION", "Unknown").strip() or "Unknown",
            context_timezone=os.getenv("ZAI_CONTEXT_TIMEZONE", "Europe/Istanbul").strip(),
            host=os.getenv("HOST", "0.0.0.0").strip(),
            port=int(os.getenv("PORT", "8000")),
        )


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}
