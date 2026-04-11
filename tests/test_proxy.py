import httpx

from zai_proxy.client import ZaiCapacityError, ZaiClient
from zai_proxy.parser import parse_sse_completion
from zai_proxy.utils import build_signature, build_sorted_payload


def test_signature_matches_captured_browser_value():
    timestamp = "1775599005668"
    request_id = "a8ac2475-2aab-4c5f-8a06-b0070753b258"
    user_id = "8f8cc6ad-5979-43a8-9712-14345520e1e9"
    prompt = "Test. Reply with exactly: pong"

    sorted_payload = build_sorted_payload(timestamp, request_id, user_id)
    signature = build_signature(sorted_payload, prompt, timestamp)

    assert (
        signature
        == "75df737cc60d03eceff6a77b38aff03cbd0f5ad81ea79f2d342a2d1a3f48304a"
    )


def test_sse_parser_collects_answer_and_usage():
    raw = (
        'data: {"type":"chat:completion","data":{"delta_content":"pon","phase":"answer"}}\n\n'
        'data: {"type":"chat:completion","data":{"delta_content":"g","phase":"answer"}}\n\n'
        'data: {"type":"chat:completion","data":{"phase":"other","usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}}\n\n'
        'data: {"type":"chat:completion","data":{"phase":"done","done":true}}\n\n'
        'data: {"data":"[DONE]","type":"chat:completion"}\n\n'
    )

    parsed = parse_sse_completion(raw)

    assert parsed.answer == "pong"
    assert parsed.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
    }


def test_system_messages_are_folded_into_next_user_message():
    normalized = ZaiClient._normalize_messages_for_upstream(
        [
            {"role": "system", "content": "Reply with exactly banana."},
            {"role": "user", "content": "What should you reply?"},
        ]
    )

    assert normalized == [
        {
            "role": "user",
            "content": (
                "You must follow these instructions for this conversation.\n"
                "<system>\n"
                "Reply with exactly banana.\n"
                "</system>\n\n"
                "Respond to the user message below.\n"
                "<user>\n"
                "What should you reply?\n"
                "</user>"
            ),
        }
    ]


def test_system_message_without_following_user_is_invalid():
    try:
        ZaiClient._normalize_messages_for_upstream(
            [
                {"role": "user", "content": "hello"},
                {"role": "system", "content": "be terse"},
            ]
        )
    except ValueError as exc:
        assert str(exc) == "System messages must be followed by a user message"
    else:
        raise AssertionError("Expected ValueError for dangling system message")


def test_capacity_error_is_detected_from_sse_payload():
    raw = (
        'data: {"type":"chat:completion","data":{"error":{"detail":"Model is currently at capacity. Please try again later or switch to another model."}}}\n\n'
    )

    parsed = parse_sse_completion(raw)

    try:
        ZaiClient._raise_upstream_error(parsed)
    except ZaiCapacityError as exc:
        assert "at capacity" in str(exc)
    else:
        raise AssertionError("Expected ZaiCapacityError")


def test_block_page_is_mapped_to_clear_error():
    request = httpx.Request("POST", "https://chat.z.ai/api/v2/chat/completions")
    response = httpx.Response(
        405,
        request=request,
        text="Sorry, your request has been blocked as it may cause potential threats to the server's security.",
    )

    error = ZaiClient._http_status_to_error(response)

    assert error.status_code == 503
    assert "blocked the current ip" in str(error).lower()
