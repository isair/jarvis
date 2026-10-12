"""Rejected recognition results cannot choose a pending reply's language."""
from dataclasses import replace
from jarvis.config import get_default_config

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("rejected", ["empty", "whitespace", "repetition", "accepted"])
def test_rejected_recognition_preserves_the_reply_language(mock_config, monkeypatch, rejected):
    from jarvis.listening import listener as module
    from jarvis.reply import engine

    monkeypatch.setattr(module, "create_intent_judge", lambda cfg: None)
    listener = module.VoiceListener(None, mock_config, None, None)
    query = "Jarvis, tell me the weather"
    recognised = []
    listener._process_transcript = lambda text, *args, **kwargs: recognised.append(text)
    first = module._TranscriptionResult(
        text=query, language="en", low_confidence_events=(),
        start_time=1, end_time=2, energy=get_default_config()["voice_min_energy"],
        capture_generation=listener._capture_generation,
        captured_during_tts=False, captured_tts_start_time=0,
    )
    listener._handle_transcription_result(first)
    assert recognised == [query]
    text = {"empty": "", "whitespace": "  ", "repetition": "noise " * 8, "accepted": "오늘 날씨는 어때요?"}[rejected]
    listener._handle_transcription_result(replace(first, text=text, language="ko"))
    observed = []
    def reply(*args, language, **kwargs):
        observed.append((args[3], language))
        return None
    monkeypatch.setattr(engine, "run_reply_engine", reply)
    dispatch_text = text if rejected == "accepted" else query
    expected_language = "ko" if rejected == "accepted" else first.language
    listener._dispatch_query(dispatch_text, generation=listener._capture_generation)
    assert observed == [(dispatch_text, expected_language)]
    assert recognised == ([query, text] if rejected == "accepted" else [query])
