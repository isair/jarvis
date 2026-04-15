"""Listening module - Voice capture and processing.

Imports are lazy so that importing a lightweight submodule (e.g.
echo_detection) does not drag in heavy dependencies like faster-whisper
or ctranslate2 via listener.py.
"""

from __future__ import annotations


def __getattr__(name: str):
    """Lazily import public names on first access."""
    _imports = {
        "VoiceListener": ".listener",
        "EchoDetector": ".echo_detection",
        "StateManager": ".state_manager",
        "ListeningState": ".state_manager",
        "is_wake_word_detected": ".wake_detection",
        "extract_query_after_wake": ".wake_detection",
        "is_stop_command": ".wake_detection",
        "TranscriptBuffer": ".transcript_buffer",
        "TranscriptSegment": ".transcript_buffer",
        "IntentJudge": ".intent_judge",
        "IntentJudgment": ".intent_judge",
        "create_intent_judge": ".intent_judge",
    }
    if name in _imports:
        import importlib
        mod = importlib.import_module(_imports[name], __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "VoiceListener",
    "EchoDetector",
    "StateManager",
    "ListeningState",
    "is_wake_word_detected",
    "extract_query_after_wake",
    "is_stop_command",
    "TranscriptBuffer",
    "TranscriptSegment",
    "IntentJudge",
    "IntentJudgment",
    "create_intent_judge",
]
