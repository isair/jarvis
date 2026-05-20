"""Chat event IPC and text-input processor behaviour."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_parse_chat_ipc_line():
    from jarvis.chat_events import CHAT_IPC_PREFIX, parse_chat_ipc_line

    line = f"{CHAT_IPC_PREFIX}{json.dumps({'role': 'assistant', 'text': 'Hello'})}\n"
    assert parse_chat_ipc_line(line) == ("assistant", "Hello")


@pytest.mark.unit
def test_emit_chat_message_calls_handlers():
    from jarvis import chat_events

    seen: list[tuple[str, str]] = []

    def _handler(role: str, text: str) -> None:
        seen.append((role, text))

    chat_events.register_chat_handler(_handler)
    try:
        with patch("builtins.print") as mock_print:
            chat_events.emit_chat_message("user", "  Hi  ")
        assert seen == [("user", "Hi")]
        mock_print.assert_not_called()
    finally:
        chat_events.unregister_chat_handler(_handler)


@pytest.mark.unit
def test_emit_chat_message_prints_ipc_without_handlers():
    from jarvis import chat_events

    with patch("builtins.print") as mock_print:
        chat_events.emit_chat_message("assistant", "Hello")
    assert mock_print.call_count == 1
    assert "__CHAT__:" in mock_print.call_args[0][0]


@pytest.mark.unit
def test_text_processor_dispatches_query():
    from jarvis.listening.listener import VoiceListener

    listener = MagicMock(spec=VoiceListener)
    listener._should_stop = False
    listener._text_query_queue = __import__("queue").Queue()
    listener._dispatch_query = MagicMock()

    def _get_with_stop(*_args, **_kwargs):
        listener._should_stop = True
        return ("test message", [])

    listener._text_query_queue.get = _get_with_stop

    with patch("jarvis.text_input.register_voice_listener"):
        VoiceListener._text_processor_loop(listener)

    listener._dispatch_query.assert_called_once_with(
        "test message", source="text", image_paths=[]
    )
