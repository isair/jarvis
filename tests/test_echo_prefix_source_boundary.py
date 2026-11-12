import pytest
from jarvis.listening.echo_detection import EchoDetector

pytestmark = pytest.mark.unit

@pytest.mark.parametrize('tts,heard', [
    ('कल शाम', 'कला शाम विषय अलग है'),
    ('ฝน วันนี้', 'ฝัน วันนี้ เป็น เรื่อง อื่น'),
    ('Rain today.', 'Train today costs twelve pounds'),
    ('Ready now.', 'Read now the full text'),
    ('... rain today', '... train today costs twelve pounds'),
])
def test_similar_short_prefixes_do_not_strip_user_speech(tts, heard):
    detector = EchoDetector()
    detector.track_tts_start(tts)
    assert detector.cleanup_leading_echo(heard) == heard


@pytest.mark.parametrize('minimum', [2, 3, 4])
def test_fuzzy_prefix_uses_the_configured_minimum_evidence(minimum):
    detector = EchoDetector()
    detector.min_salvage_words = minimum
    tts, heard = 'weather will clear', 'weether will clear tell me more'
    detector.track_tts_start(tts)
    expected = 'tell me more' if len(tts.split()) >= minimum else heard
    assert detector.cleanup_leading_echo(heard) == expected


def test_literal_short_prefix_preserves_exact_echo_cleanup():
    detector = EchoDetector()
    detector.track_tts_start('Weather today.')
    assert detector.cleanup_leading_echo('WEATHER today tell me more') == 'tell me more'


def test_punctuation_does_not_supply_a_second_literal_echo_word():
    detector = EchoDetector()
    detector.track_tts_start('... today')
    text = '... today tell me more'
    assert detector.cleanup_leading_echo(text) == text


@pytest.mark.parametrize('tts,text', [
    ('Rain today.', 'Train today costs twelve pounds'),
    ('Ready now.', 'Read now the full text'),
    ('कल शाम', 'कला शाम विषय अलग है'),
    ('ฝน วันนี้', 'ฝัน วันนี้ เป็น เรื่อง อื่น'),
])
def test_short_tts_reference_preserves_the_complete_user_request(tts, text):
    from test_hot_window_input import (
        _accepted_query, _create_listener, _install_intent_judge, _make_judgment,
        _process_transcript, _simulate_tts_finish, _wait_for_hot_window_state,
    )
    listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3)
    try:
        listener.echo_detector.track_tts_start(tts)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_state(listener)
        _install_intent_judge(listener, _make_judgment(directed=True))
        _process_transcript(listener, text, utterance_energy=0.01)
        assert _accepted_query(listener) == text
    finally:
        listener.state_manager.stop()
