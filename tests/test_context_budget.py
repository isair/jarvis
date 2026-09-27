"""The reply context stays bounded without splitting tool-call groups."""

from copy import deepcopy
from unittest.mock import patch

import pytest

from jarvis.reply.context_budget import (
    ContextBudgetExceeded,
    InvalidToolHistory,
    bound_context_messages,
    estimate_message_tokens,
)


def _call(call_id, name="webSearch"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": {}}}


def test_keeps_newest_complete_native_tool_batch_and_current_request():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "", "tool_calls": [_call("old")]},
        {"role": "tool", "tool_call_id": "old", "content": "old result"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current request"},
        {"role": "assistant", "content": "", "tool_calls": [_call("one"), _call("two")]},
        {"role": "tool", "tool_call_id": "one", "content": "first"},
        {"role": "tool", "tool_call_id": "two", "content": "second"},
    ]

    bounded = bound_context_messages(
        messages, current_user_index=5, max_tokens=5, estimator=lambda _: 1,
    )

    assert [item["role"] for item in bounded] == ["system", "user", "assistant", "tool", "tool"]
    assert {item["tool_call_id"] for item in bounded if item["role"] == "tool"} == {"one", "two"}
    assert bounded[1]["content"] == "current request"


def test_trimming_tool_text_closes_fence_and_preserves_result_pointer():
    result_id = "a" * 32
    tool_text = (
        "<<<BEGIN UNTRUSTED WEB EXTRACT>>>\n"
        + "untrusted text " * 80
        + "\n<<<END UNTRUSTED WEB EXTRACT>>>\n"
        + f"[Excerpt of 2000 characters. Full result ID: {result_id}; call readTaskResult to read more.]"
    )
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "summarise"},
        {"role": "assistant", "content": "", "tool_calls": [_call("one")]},
        {"role": "tool", "tool_call_id": "one", "content": tool_text},
    ]
    original = deepcopy(messages)

    bounded = bound_context_messages(
        messages, current_user_index=1, max_tokens=210,
        estimator=lambda msg: len(str(msg.get("content", ""))) + 1,
    )

    excerpt = bounded[3]["content"]
    assert len(excerpt) < len(tool_text)
    assert excerpt.count("<<<BEGIN UNTRUSTED WEB EXTRACT>>>") == 1
    assert excerpt.count("<<<END UNTRUSTED WEB EXTRACT>>>") == 1
    assert excerpt.index("<<<BEGIN") < excerpt.index("<<<END")
    assert f"Full result ID: {result_id}" in excerpt
    assert messages == original


def test_cannot_fit_required_system_and_current_user_raises():
    messages = [
        {"role": "system", "content": "rules " * 30},
        {"role": "user", "content": "current request"},
    ]

    with pytest.raises(ContextBudgetExceeded):
        bound_context_messages(
            messages, current_user_index=1, max_tokens=50,
            estimator=lambda msg: len(msg["content"]),
        )


def test_orphan_native_result_is_rejected():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "current"},
        {"role": "tool", "tool_call_id": "missing", "content": "orphan"},
    ]

    with pytest.raises(InvalidToolHistory):
        bound_context_messages(messages, current_user_index=1, max_tokens=100)


def test_mismatched_native_result_is_rejected_without_type_error():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "current"},
        {"role": "assistant", "content": "", "tool_calls": [_call("expected"), _call("another")]},
        {"role": "tool", "tool_call_id": "expected", "content": "first"},
        {"role": "tool", "content": "missing call ID"},
    ]

    with pytest.raises(InvalidToolHistory):
        bound_context_messages(messages, current_user_index=1, max_tokens=100)


def test_text_tool_request_and_result_stay_together():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "current"},
        {"role": "assistant", "content": "tool_calls: [{...}]"},
        {"role": "user", "tool_name": "webSearch", "content": "[Tool result] found"},
    ]

    bounded = bound_context_messages(
        messages, current_user_index=1, max_tokens=4, estimator=lambda _: 1,
    )

    assert [item["role"] for item in bounded] == ["system", "user", "assistant", "user"]
    assert bounded[-1]["tool_name"] == "webSearch"


def test_token_estimate_accounts_for_multibyte_text():
    ascii_message = {"role": "user", "content": "hello"}
    multilingual_message = {"role": "user", "content": "你好🙂"}

    assert estimate_message_tokens(multilingual_message) >= estimate_message_tokens(ascii_message)


@pytest.mark.parametrize("model", ["gpt-oss:20b", "gemma4:e2b"])
def test_default_budget_allows_a_routine_tool_request(
    model,
    mock_config, db, dialogue_memory,
):
    from jarvis.reply import engine as engine_mod

    mock_config.llm_chat_model = model
    seen = []
    original_bound = engine_mod.bound_context_messages

    def observe(messages, **kwargs):
        required = sum(estimate_message_tokens(item) for item in (messages[0], messages[kwargs["current_user_index"]]))
        seen.append((required, kwargs["reserve_tokens"], kwargs["max_tokens"]))
        return original_bound(messages, **kwargs)

    with patch.object(engine_mod, "bound_context_messages", side_effect=observe), \
         patch.object(engine_mod, "select_tools", return_value=["getWeather"]), \
         patch.object(engine_mod, "chat_with_messages", return_value={
             "message": {"role": "assistant", "content": "I can check the weather."}
         }):
        reply = engine_mod.run_reply_engine(
            db, mock_config, None, "what's the weather?", dialogue_memory,
        )

    assert reply == "I can check the weather.", seen
