from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(slots=True)
class ParsedCompletion:
    answer: str
    reasoning: str
    usage: dict
    error: dict | None


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
