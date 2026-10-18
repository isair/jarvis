"""Only accepted query segments choose the reply's locale."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
from jarvis.config import get_default_config

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("kind", ["empty", "whitespace", "repetition", "echo", "undirected", "accepted", "unknown"])
def test_pending_reply_language_tracks_accepted_request(mock_config, monkeypatch, kind):
    from jarvis.listening import listener as module
    from jarvis.listening import state_manager as state
    from jarvis.reply import engine

    clock = {"now": 1000.0}
    monkeypatch.setattr(state.time, "time", lambda: clock["now"])
    monkeypatch.setattr(module, "create_intent_judge", lambda cfg: None)
    obj = module.VoiceListener(None, mock_config, None, None)
    obj._start_thinking_tune = lambda: None
    obj._stop_thinking_tune = lambda: None
    first = module._TranscriptionResult(
        text="Jarvis, tell me the weather", language="en", low_confidence_events=(),
        start_time=999, end_time=1000, energy=get_default_config()["voice_min_energy"],
        capture_generation=obj._capture_generation,
        captured_during_tts=False, captured_tts_start_time=0,
    )
    obj._handle_transcription_result(first)
    assert obj.state_manager.get_pending_query() == first.text
    texts = {
        "empty": "", "whitespace": "  ", "repetition": "noise " * 8,
        "echo": "Le ciel sera couvert demain à Londres.",
        "undirected": "Je parle à Jarvis, pas à toi.",
        "accepted": "Jarvis, 오늘 날씨는 어때요?", "unknown": "Jarvis, provide the forecast.",
    }
    if kind == "echo":
        obj.tts = SimpleNamespace(enabled=True, is_speaking=lambda: True)
        obj.echo_detector._last_tts_text = texts[kind]
        monkeypatch.setattr(obj.echo_detector, "should_reject_as_echo", lambda *args, **kwargs: True)
    if kind == "undirected":
        obj._intent_judge = SimpleNamespace(
            available=True, judge=lambda **kwargs: SimpleNamespace(
                directed=False, stop=False, confidence="high", reasoning="Addressed another person",
            ),
        )
    following_language = None if kind == "unknown" else ("fr" if kind in ("echo", "undirected") else "ko")
    following = replace(first, text=texts[kind], language=following_language,
                        captured_during_tts=kind == "echo")
    obj._handle_transcription_result(following)
    expected = following if kind in ("accepted", "unknown") else first
    assert obj.state_manager.get_pending_query() == expected.text
    observed = []

    def reply(*args, language, **kwargs):
        observed.append((args[3], language))
        return None

    monkeypatch.setattr(engine, "run_reply_engine", reply)
    clock["now"] += 100
    obj._check_query_timeout()
    assert observed == [(expected.text, expected.language)]


def test_collection_snapshot_retains_only_accepted_segment_metadata():
    from jarvis.listening.state_manager import StateManager

    state = StateManager()
    state.start_collection("Jarvis", language="en")
    state.add_to_collection("quel temps fera-t-il ?", language="fr")
    query = state.clear_collection()
    assert (query.text, query.language) == ("Jarvis quel temps fera-t-il ?", "fr")
    state.start_collection("Another request", language="ko")
    assert (query.text, query.language) == ("Jarvis quel temps fera-t-il ?", "fr")
    state.reset_engagement()
    empty = state.clear_collection()
    assert (empty.text, empty.language) == ("", None)


def test_unknown_accepted_segment_does_not_inherit_a_previous_locale():
    from jarvis.listening.state_manager import StateManager

    state = StateManager()
    state.start_collection("Jarvis", language="en")
    state.add_to_collection("tell me the weather", language=None)
    query = state.clear_collection()
    assert (query.text, query.language) == ("Jarvis tell me the weather", None)
