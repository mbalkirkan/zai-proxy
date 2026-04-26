from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(slots=True)
class ParsedCompletion:
    answer: str
    reasoning: str
    usage: dict
    error: dict | None
    request_message_id: int | None = None
    response_message_id: int | None = None


def parse_sse_completion(raw: str) -> ParsedCompletion:
    answer_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict = {}
    error: dict | None = None

    for block in raw.split("\n\n"):
        block = block.strip()
        if not block.startswith("data: "):
            continue
        payload = block[6:].strip()
        if payload == "[DONE]":
            continue

        event = json.loads(payload)
        data = event.get("data", {})
        if isinstance(data, str):
            continue

        phase = data.get("phase")
        delta = data.get("delta_content")
        if delta:
            if phase == "answer":
                answer_parts.append(delta)
            elif phase == "thinking":
                reasoning_parts.append(delta)

        if data.get("usage"):
            usage = data["usage"]

        if data.get("error"):
            error = data["error"]

    return ParsedCompletion(
        answer="".join(answer_parts).strip(),
        reasoning="".join(reasoning_parts).strip(),
        usage=usage,
        error=error,
    )


def parse_deepseek_completion(raw: str) -> ParsedCompletion:
    answer_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict = {}
    error: dict | None = None
    request_message_id: int | None = None
    response_message_id: int | None = None
    fragments: list[dict] = []
    last_path = ""
    last_op = "SET"

    for block in raw.split("\n\n"):
        event_name = ""
        data_lines: list[str] = []

        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())

        if not data_lines:
            continue

        payload = "\n".join(data_lines).strip()
        if not payload or payload == "[DONE]":
            continue

        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if event_name == "ready":
            request_message_id = event.get("request_message_id")
            response_message_id = event.get("response_message_id")
            continue

        if isinstance(event, dict) and event.get("code") not in {None, 0}:
            error = {
                "code": event.get("code"),
                "detail": event.get("msg") or event.get("message") or "DeepSeek upstream error",
            }
            continue

        if not isinstance(event, dict):
            continue

        path = event.get("p") or last_path
        op = event.get("o") or last_op
        value = event.get("v")
        last_path = path
        last_op = op

        if isinstance(value, dict):
            response = value.get("response")
            if isinstance(response, dict):
                usage_value = response.get("accumulated_token_usage")
                if isinstance(usage_value, int):
                    usage = {"total_tokens": usage_value}

                for fragment in response.get("fragments") or []:
                    if isinstance(fragment, dict):
                        _append_deepseek_fragment(fragment, fragments, answer_parts, reasoning_parts)
            continue

        if path == "response/fragments" and op == "APPEND" and isinstance(value, list):
            for fragment in value:
                if isinstance(fragment, dict):
                    _append_deepseek_fragment(fragment, fragments, answer_parts, reasoning_parts)
            continue

        if path.endswith("/content") and isinstance(value, str):
            target = fragments[-1] if fragments else {}
            fragment_type = target.get("type")
            if op == "SET":
                target["content"] = value
                if fragment_type in {"RESPONSE", "TEMPLATE_RESPONSE"}:
                    answer_parts.clear()
                    answer_parts.append(value)
                elif fragment_type == "THINK":
                    reasoning_parts.clear()
                    reasoning_parts.append(value)
            elif fragment_type in {"RESPONSE", "TEMPLATE_RESPONSE"}:
                target["content"] = str(target.get("content") or "") + value
                answer_parts.append(value)
            elif fragment_type == "THINK":
                target["content"] = str(target.get("content") or "") + value
                reasoning_parts.append(value)
            continue

        if path == "response" and isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and item.get("p") == "accumulated_token_usage":
                    token_count = item.get("v")
                    if isinstance(token_count, int):
                        usage = {"total_tokens": token_count}

    return ParsedCompletion(
        answer="".join(answer_parts).strip(),
        reasoning="".join(reasoning_parts).strip(),
        usage=usage,
        error=error,
        request_message_id=request_message_id,
        response_message_id=response_message_id,
    )


def _append_deepseek_fragment(
    fragment: dict,
    fragments: list[dict],
    answer_parts: list[str],
    reasoning_parts: list[str],
) -> None:
    fragments.append(fragment)
    content = str(fragment.get("content") or "")
    if not content:
        return

    fragment_type = fragment.get("type")
    if fragment_type in {"RESPONSE", "TEMPLATE_RESPONSE"}:
        answer_parts.append(content)
    elif fragment_type == "THINK":
        reasoning_parts.append(content)
