import httpx

from zai_proxy.client import ZaiCapacityError, ZaiClient
from zai_proxy.config import _normalize_deepseek_token
from zai_proxy.deepseek_client import DeepSeekClient
from zai_proxy.deepseek_pow import _deepseek_hash_hex, solve_deepseek_pow
from zai_proxy.parser import parse_deepseek_completion, parse_sse_completion
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


def test_deepseek_sse_parser_collects_answer_reasoning_and_ids():
    raw = (
        'event: ready\n'
        'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n\n'
        'data: {"v":{"response":{"fragments":[{"type":"THINK","content":"Reason"}]}}}\n\n'
        'data: {"p":"response/fragments/-1/content","o":"APPEND","v":"ing"}\n\n'
        'data: {"p":"response/fragments","o":"APPEND","v":[{"type":"RESPONSE","content":"pon"}]}\n\n'
        'data: {"p":"response/fragments/-1/content","v":"g"}\n\n'
        'data: {"p":"response","o":"BATCH","v":[{"p":"accumulated_token_usage","v":12}]}\n\n'
    )

    parsed = parse_deepseek_completion(raw)

    assert parsed.answer == "pong"
    assert parsed.reasoning == "Reasoning"
    assert parsed.usage == {"total_tokens": 12}
    assert parsed.request_message_id == 1
    assert parsed.response_message_id == 2


def test_deepseek_pow_matches_captured_browser_worker_value():
    digest = _deepseek_hash_hex(b"e0172917344540087a50_1777238475948_72006")

    assert digest == "46b974fc7a992700f287b4c8e1092c7b522a4b1871bd85e8830ba54abe307f59"


def test_deepseek_pow_hash_helper_can_build_easy_challenge():
    challenge_hash = _deepseek_hash_hex(b"salt_123_0")
    challenge = {
        "algorithm": "DeepSeekHashV1",
        "challenge": challenge_hash,
        "salt": "salt",
        "signature": "sig",
        "difficulty": 1,
        "expire_at": 123,
        "target_path": "/api/v0/chat/completion",
    }

    assert solve_deepseek_pow(challenge).answer == 0


def test_deepseek_model_aliases_are_resolved():
    assert DeepSeekClient._resolve_model("deepseek-chat") == "default"
    assert DeepSeekClient._resolve_model("deepseek-reasoner") == "expert"
    assert DeepSeekClient._resolve_model("expert") == "expert"


def test_deepseek_token_accepts_raw_json_or_bearer_value():
    assert _normalize_deepseek_token('{"value":"abc","__version":"0"}') == "abc"
    assert _normalize_deepseek_token("Bearer abc") == "abc"


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


def test_messages_from_chat_returns_current_linear_history():
    chat_data = {
        "id": "chat-1",
        "chat": {
            "history": {
                "currentId": "a2",
                "messages": {
                    "u1": {
                        "id": "u1",
                        "parentId": None,
                        "role": "user",
                        "content": "hello",
                    },
                    "a1": {
                        "id": "a1",
                        "parentId": "u1",
                        "role": "assistant",
                        "content": "hi",
                    },
                    "u2": {
                        "id": "u2",
                        "parentId": "a1",
                        "role": "user",
                        "content": "again",
                    },
                    "a2": {
                        "id": "a2",
                        "parentId": "u2",
                        "role": "assistant",
                        "content": "done",
                    },
                },
            }
        },
    }

    assert ZaiClient._messages_from_chat(chat_data) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "again"},
        {"role": "assistant", "content": "done"},
    ]


def test_seed_from_existing_chat_uses_current_id_as_parent():
    chat_data = {
        "id": "chat-1",
        "chat": {
            "history": {
                "currentId": "assistant-last",
                "messages": {},
            }
        },
    }

    seed = ZaiClient._seed_from_existing_chat(chat_data)

    assert seed.chat_id == "chat-1"
    assert seed.current_user_message_parent_id == "assistant-last"
    assert seed.current_user_message_id
