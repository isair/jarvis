"""Unit tests for the unified turn foundation (jarvis.core.turn).

Phase 1 of the unified-local-voice-chat integration. These tests are pure and
offline: no GPU, microphone, network, model, or external service is touched.
They verify behaviours (ids, validation, serialisation, invariants), not
implementation details.
"""

from __future__ import annotations

import ast
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from jarvis.core.turn import (
    AssistantMessage,
    CancellationToken,
    StructuredError,
    TtsStatus,
    Turn,
    TurnContext,
    TurnResult,
    TurnSource,
    TurnStatus,
    UserMessage,
    new_assistant_message,
    new_turn,
    new_user_message,
)


# 1. text input produces a UserMessage with source=text
def test_text_input_produces_text_user_message():
    turn = new_turn("text", "salut cora", "ro")
    assert isinstance(turn, Turn)
    assert isinstance(turn.user_message, UserMessage)
    assert turn.user_message.source == TurnSource.TEXT
    assert turn.context.source == TurnSource.TEXT
    assert turn.user_message.text == "salut cora"


# 2. voice input produces a UserMessage with source=voice
def test_voice_input_produces_voice_user_message():
    turn = new_turn("voice", "salut cora", "ro")
    assert turn.user_message.source == TurnSource.VOICE
    assert turn.context.source == TurnSource.VOICE


# 3. invalid sources are rejected
def test_invalid_source_rejected():
    with pytest.raises(ValueError):
        new_turn("keyboard", "hi", "en")
    with pytest.raises(ValueError):
        new_user_message("banana", "hi", "en", conversation_id="c", turn_id="t")


# 4. empty / whitespace text is rejected
@pytest.mark.parametrize("bad_text", ["", "   ", "\t\n", None])
def test_empty_text_rejected(bad_text):
    with pytest.raises(ValueError):
        new_turn("text", bad_text, "ro")
    with pytest.raises(ValueError):
        UserMessage(
            message_id="m", conversation_id="c", turn_id="t",
            source=TurnSource.TEXT, text=bad_text, language="ro", created_at="now",
        )


# 5. a supplied conversation_id is preserved
def test_supplied_conversation_id_preserved():
    turn = new_turn("text", "hi", "en", conversation_id="conv-123")
    assert turn.context.conversation_id == "conv-123"
    assert turn.user_message.conversation_id == "conv-123"


# 6. conversation_id is generated when missing
def test_conversation_id_generated_when_missing():
    turn = new_turn("text", "hi", "en")
    assert turn.context.conversation_id
    assert turn.user_message.conversation_id == turn.context.conversation_id
    # two turns without a conversation_id get distinct ones
    other = new_turn("text", "hi", "en")
    assert other.context.conversation_id != turn.context.conversation_id


# 7. two turns have different turn_id
def test_two_turns_have_different_turn_id():
    a = new_turn("text", "hi", "en")
    b = new_turn("text", "hi", "en")
    assert a.context.turn_id != b.context.turn_id
    assert a.user_message.turn_id == a.context.turn_id  # message bound to its turn


# 8. two turns have different correlation_id
def test_two_turns_have_different_correlation_id():
    a = new_turn("text", "hi", "en")
    b = new_turn("text", "hi", "en")
    assert a.context.correlation_id != b.context.correlation_id
    assert a.context.correlation_id != a.context.turn_id  # distinct id spaces


# 9. ids remain stable after creation (nothing regenerates them)
def test_ids_stable_after_creation():
    turn = new_turn("text", "hi", "en", conversation_id="c1")
    ids = (turn.context.turn_id, turn.context.correlation_id,
           turn.context.conversation_id, turn.user_message.message_id)
    # mutating unrelated state must not touch the ids
    turn.context.cancellation.cancel("user stop")
    turn.context.state = turn.context.state
    turn.to_dict()
    assert (turn.context.turn_id, turn.context.correlation_id,
            turn.context.conversation_id, turn.user_message.message_id) == ids


# 10. created_at is timezone-aware
def test_created_at_is_timezone_aware():
    turn = new_turn("voice", "hi", "en")
    for ts in (turn.context.created_at, turn.user_message.created_at):
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)  # UTC


# 11. language 'ro' stays 'ro'
def test_language_ro_preserved():
    assert new_turn("text", "buna", "ro").user_message.language == "ro"
    assert new_turn("text", "buna", " RO ").user_message.language == "ro"  # trimmed + lowered
    assert new_turn("text", "buna", "").user_message.language is None       # blank -> None


# 12. speak_response defaults to True for both text and voice
def test_speak_response_default_true_for_both_sources():
    assert new_turn("text", "hi", "en").context.speak_response is True
    assert new_turn("voice", "hi", "en").context.speak_response is True


# 13. speak_response=False is preserved
def test_speak_response_false_preserved():
    assert new_turn("text", "hi", "en", speak_response=False).context.speak_response is False
    assert new_turn("voice", "hi", "en", speak_response=False).context.speak_response is False


# 14. objects serialise predictably (JSON round-trip, enums -> values, stable)
def test_predictable_serialisation():
    turn = new_turn("voice", "salut", "ro", conversation_id="c9", speak_response=False)
    d = turn.to_dict()
    # enums become their string values
    assert d["user_message"]["source"] == "voice"
    assert d["context"]["source"] == "voice"
    assert d["context"]["state"] == "created"
    assert d["context"]["speak_response"] is False
    # cancellation token is a plain dict
    assert d["context"]["cancellation"] == {"cancelled": False, "reason": None}
    # fully JSON-serialisable and deterministic
    dumped = json.dumps(d, sort_keys=True)
    assert json.dumps(turn.to_dict(), sort_keys=True) == dumped
    # assistant message + turn result also serialise cleanly
    am = new_assistant_message(turn.context)
    am.error = StructuredError(kind="tool", message="boom")
    json.dumps(am.to_dict())
    result = TurnResult(status="completed", assistant_message=am)
    round_trip = json.loads(json.dumps(result.to_dict()))
    assert round_trip["status"] == "completed"
    assert round_trip["assistant_message"]["error"]["kind"] == "tool"


# 15. invalid TurnResult statuses are rejected
def test_invalid_turn_result_status_rejected():
    with pytest.raises(ValueError):
        TurnResult(status="banana")
    # valid values (string or enum) are accepted and coerced to the enum
    assert TurnResult(status="completed").status == TurnStatus.COMPLETED
    assert TurnResult(status=TurnStatus.BLOCKED).status == TurnStatus.BLOCKED


# 16. a failed TTS does not change a semantically completed TurnResult
def test_tts_failure_does_not_change_semantic_status():
    result = TurnResult(status=TurnStatus.COMPLETED, tts_status=TtsStatus.FAILED)
    assert result.status == TurnStatus.COMPLETED
    assert result.is_success is True
    assert result.tts_status == TtsStatus.FAILED  # secondary channel recorded, but independent


# 17. the turn module depends only on the standard library
def test_module_has_no_heavy_dependencies():
    src = Path(__file__).resolve().parents[1] / "src" / "jarvis" / "core" / "turn.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
    forbidden = {
        "numpy", "torch", "sounddevice", "faster_whisper", "whisper",
        "requests", "PyQt6", "openai", "webrtcvad", "piper", "ollama",
    }
    assert not (imported & forbidden), f"turn.py imports heavy deps: {imported & forbidden}"
    # only standard-library modules used
    assert imported <= {"__future__", "uuid", "dataclasses", "datetime", "enum", "typing"}
