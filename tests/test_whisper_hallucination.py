"""
Tests for Whisper repetitive hallucination detection.

VoiceListener._is_repetitive_hallucination detects repeated words/characters
that Whisper produces on quiet or ambiguous audio input.
"""

import pytest
from unittest.mock import patch, MagicMock


def _create_listener():
    """Create a VoiceListener with mocked dependencies for testing."""
    mock_cfg = MagicMock()
    mock_cfg.whisper_model = "small"
    mock_cfg.whisper_device = "auto"
    mock_cfg.whisper_compute_type = "int8"
    mock_cfg.whisper_backend = "faster-whisper"
    mock_cfg.sample_rate = 16000
    mock_cfg.vad_enabled = False
    mock_cfg.vad_aggressiveness = 2
    mock_cfg.echo_tolerance = 0.3
    mock_cfg.echo_energy_threshold = 2.0
    mock_cfg.hot_window_seconds = 3.0
    mock_cfg.hot_window_enabled = True
    mock_cfg.voice_collect_seconds = 2.0
    mock_cfg.voice_max_collect_seconds = 60.0
    mock_cfg.voice_device = None
    mock_cfg.voice_debug = False
    mock_cfg.voice_min_energy = 0.0045
    mock_cfg.tune_enabled = False
    mock_cfg.wake_word = "jarvis"
    mock_cfg.wake_aliases = []
    mock_cfg.wake_fuzzy_ratio = 0.78
    mock_cfg.stop_commands = ["stop", "quiet"]
    mock_cfg.tts_rate = 200
    mock_cfg.transcript_buffer_duration_sec = 120.0
    mock_cfg.intent_judge_model = "gemma4:e2b"
    mock_cfg.ollama_base_url = "http://127.0.0.1:11434"
    mock_cfg.intent_judge_timeout_sec = 3.0
    mock_cfg.audio_wake_enabled = False
    mock_cfg.audio_wake_threshold = 0.5

    with patch("jarvis.listening.listener.webrtcvad", None), \
         patch("jarvis.listening.listener.sd", None), \
         patch("jarvis.listening.listener.np", None), \
         patch("jarvis.listening.listener.create_intent_judge", return_value=None), \
         patch("jarvis.listening.listener.WakeWordDetector"):
        from jarvis.listening.listener import VoiceListener
        listener = VoiceListener(
            MagicMock(), mock_cfg, MagicMock(), MagicMock()
        )

    return listener


@pytest.mark.unit
class TestCharLevelHallucination:
    """Character-level repetition detection (e.g. multilingual gibberish)."""

    def test_repeated_two_char_pattern(self):
        """Detects repeating 2-char pattern like Whisper's multilingual hallucinations."""
        listener = _create_listener()
        # Simulates "Jろ Jろ Jろ Jろ Jろ Jろ" — common Whisper hallucination
        assert listener._is_repetitive_hallucination("Jろ Jろ Jろ Jろ Jろ Jろ") is True

    def test_repeated_single_char_pattern(self):
        """Detects repeating single character."""
        listener = _create_listener()
        assert listener._is_repetitive_hallucination("aaaaaaaaa") is True

    def test_repeated_pattern_no_spaces(self):
        """Detects repetition even without spaces."""
        listener = _create_listener()
        assert listener._is_repetitive_hallucination("abcabcabcabcabc") is True

    def test_short_text_not_hallucination(self):
        """Text shorter than 6 chars is never a hallucination."""
        listener = _create_listener()
        assert listener._is_repetitive_hallucination("aaaa") is False

    def test_empty_text(self):
        """Empty/None text is not a hallucination."""
        listener = _create_listener()
        assert listener._is_repetitive_hallucination("") is False

    def test_low_coverage_not_hallucination(self):
        """Pattern repeating but not covering 60% of text is not flagged."""
        listener = _create_listener()
        # "ab" repeats 4x (8 chars) but rest is different (coverage < 60%)
        assert listener._is_repetitive_hallucination("abababab this is a completely different long sentence") is False


@pytest.mark.unit
class TestWordLevelHallucination:
    """Word-level repetition detection (e.g. "don't don't don't...")."""

    def test_single_word_repeated(self):
        """Detects single word repeated many times (>50% of words, 4+ times)."""
        listener = _create_listener()
        assert listener._is_repetitive_hallucination("don't don't don't don't don't") is True

    def test_word_repeated_with_punctuation(self):
        """Detects repetition even with varying punctuation."""
        listener = _create_listener()
        assert listener._is_repetitive_hallucination("stop. stop! stop? stop, stop") is True

    def test_consecutive_repetition(self):
        """Detects 3+ consecutive identical words."""
        listener = _create_listener()
        assert listener._is_repetitive_hallucination("hello world stop stop stop now") is True

    def test_few_words_not_hallucination(self):
        """Fewer than 4 words is never flagged as word-level hallucination."""
        listener = _create_listener()
        assert listener._is_repetitive_hallucination("stop stop stop") is False

    def test_normal_sentence_not_hallucination(self):
        """Normal speech is not flagged."""
        listener = _create_listener()
        assert listener._is_repetitive_hallucination("what is the weather like today in London") is False

    def test_repeated_word_below_threshold(self):
        """Word repeated but below 50% of total words is not flagged."""
        listener = _create_listener()
        # "the" appears 4 times but only 4/10 = 40% of words
        assert listener._is_repetitive_hallucination(
            "the cat and the dog and the bird and the fish"
        ) is False

    def test_two_consecutive_not_hallucination(self):
        """Only 2 consecutive repetitions — not enough to flag."""
        listener = _create_listener()
        assert listener._is_repetitive_hallucination("I think think that is fine really") is False
