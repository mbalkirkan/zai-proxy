from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from datetime import datetime
from uuid import uuid4


SIGNING_SECRET = "key-@@@@)))()((9))-xxxx&&&%%%%%"


def new_id() -> str:
    return str(uuid4())


def unix_now() -> int:
    return int(time.time())


def ms_now() -> int:
    return int(time.time() * 1000)


def build_sorted_payload(timestamp_ms: str, request_id: str, user_id: str) -> str:
    return ",".join(
        [
            "requestId",
            request_id,
            "timestamp",
            timestamp_ms,
            "user_id",
            user_id,
        ]
    )


def build_signature(sorted_payload: str, prompt: str, timestamp_ms: str) -> str:
    prompt_b64 = base64.b64encode(prompt.encode("utf-8")).decode("ascii")
    bucket = int(int(timestamp_ms) / (5 * 60 * 1000))
    inner = hmac.new(
        SIGNING_SECRET.encode("utf-8"),
        str(bucket).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    signature_base = f"{sorted_payload}|{prompt_b64}|{timestamp_ms}"
    return hmac.new(
        inner.encode("utf-8"),
        signature_base.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def parse_jwt_payload(token: str) -> dict:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return {}
        payload = parts[1]
        padding = "=" * (-len(payload) % 4)
        raw = base64.urlsafe_b64decode(payload + padding)
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}


def extract_text_content(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") != "text":
                    raise ValueError(f"Unsupported content part type: {item.get('type')}")
                texts.append(item.get("text") or "")
            else:
                data = getattr(item, "model_dump", lambda: item)()
                if data.get("type") != "text":
                    raise ValueError(f"Unsupported content part type: {data.get('type')}")
                texts.append(data.get("text") or "")
        return "\n".join(texts).strip()
    raise ValueError("Unsupported message content format")


def iso_now() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
