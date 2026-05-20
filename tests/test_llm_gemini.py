"""Tests for Gemini hosted chat (payload mapping and response extraction)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_messages_to_gemini_payload_system_and_user():
    from jarvis.llm_gemini import _messages_to_gemini_payload

    payload = _messages_to_gemini_payload(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
    )
    assert "systemInstruction" in payload
    assert payload["systemInstruction"]["parts"][0]["text"] == "You are helpful."
    assert payload["contents"] == [
        {"role": "user", "parts": [{"text": "Hello"}]},
    ]


@pytest.mark.unit
def test_messages_to_gemini_payload_tool_as_user_turn():
    from jarvis.llm_gemini import _messages_to_gemini_payload

    payload = _messages_to_gemini_payload(
        [
            {"role": "assistant", "content": "Calling tool"},
            {"role": "tool", "tool_name": "webSearch", "content": '{"ok": true}'},
        ]
    )
    assert len(payload["contents"]) == 2
    assert payload["contents"][0]["role"] == "model"
    assert "[Tool result: webSearch]" in payload["contents"][1]["parts"][0]["text"]


@pytest.mark.unit
def test_extract_gemini_text_joins_parts():
    from jarvis.llm_gemini import _extract_gemini_text

    data = {
        "candidates": [
            {"content": {"parts": [{"text": "Hello"}, {"text": " world"}]}},
        ],
    }
    assert _extract_gemini_text(data) == "Hello world"


@pytest.mark.unit
def test_gemini_chat_with_messages_returns_ollama_shape():
    from jarvis.llm_gemini import gemini_chat_with_messages

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = json.dumps(
        {
            "candidates": [
                {"content": {"parts": [{"text": "Done"}]}},
            ],
        }
    )
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_resp)
    mock_cm.__exit__ = MagicMock(return_value=False)

    with patch("jarvis.llm_gemini.requests.post", return_value=mock_cm):
        out = gemini_chat_with_messages(
            api_key="secret",
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "Hi"}],
            timeout_sec=5.0,
        )

    assert isinstance(out, dict)
    assert out["message"]["content"] == "Done"


@pytest.mark.unit
def test_gemini_chat_with_messages_no_key():
    from jarvis.llm_gemini import gemini_chat_with_messages

    assert gemini_chat_with_messages("", "gemini-2.0-flash", [], timeout_sec=1.0) is None
