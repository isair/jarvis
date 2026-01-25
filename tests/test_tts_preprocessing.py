"""Tests for TTS link preprocessing functionality."""

import pytest
from src.jarvis.output.tts import (
    _preprocess_for_speech,
    _extract_domain_description,
    _estimate_tts_duration,
    DEFAULT_WPM,
    AUDIO_BUFFER_DELAY_SEC,
)


class TestExtractDomainDescription:
    """Tests for domain extraction utility."""

    def test_extracts_domain_from_simple_url(self):
        domain, is_homepage = _extract_domain_description("https://google.com")
        assert domain == "google.com"
        assert is_homepage is True

    def test_extracts_domain_from_url_with_www(self):
        domain, is_homepage = _extract_domain_description("https://www.google.com")
        assert domain == "google.com"
        assert is_homepage is True

    def test_detects_non_homepage_path(self):
        domain, is_homepage = _extract_domain_description("https://google.com/search")
        assert domain == "google.com"
        assert is_homepage is False

    def test_detects_homepage_with_trailing_slash(self):
        domain, is_homepage = _extract_domain_description("https://google.com/")
        assert domain == "google.com"
        assert is_homepage is True

    def test_handles_complex_path(self):
        domain, is_homepage = _extract_domain_description("https://docs.python.org/3/library/re.html")
        assert domain == "docs.python.org"
        assert is_homepage is False


class TestPreprocessForSpeech:
    """Tests for the main preprocessing function."""

    def test_converts_markdown_link_to_homepage(self):
        text = "Check out [Google](https://google.com) for more info."
        result = _preprocess_for_speech(text)
        assert "Link to google.com homepage with the text 'Google'" in result
        assert "[Google]" not in result
        assert "https://google.com" not in result

    def test_converts_markdown_link_to_page(self):
        text = "See [the documentation](https://docs.python.org/3/library/re.html) here."
        result = _preprocess_for_speech(text)
        assert "Link to a page under docs.python.org with the text 'the documentation'" in result

    def test_converts_raw_url_homepage(self):
        text = "Visit https://google.com for more."
        result = _preprocess_for_speech(text)
        assert "google.com homepage" in result
        assert "https://google.com" not in result

    def test_converts_raw_url_with_path(self):
        text = "Check out https://example.com/some/path for details."
        result = _preprocess_for_speech(text)
        assert "a page under example.com" in result
        assert "https://example.com/some/path" not in result

    def test_converts_www_url(self):
        text = "Go to www.example.com for more."
        result = _preprocess_for_speech(text)
        assert "example.com homepage" in result
        assert "www.example.com" not in result

    def test_handles_multiple_markdown_links(self):
        text = "Visit [Google](https://google.com) or [GitHub](https://github.com/user/repo)."
        result = _preprocess_for_speech(text)
        assert "Link to google.com homepage with the text 'Google'" in result
        assert "Link to a page under github.com with the text 'GitHub'" in result

    def test_handles_mixed_links(self):
        text = "See [docs](https://docs.example.com/api) and also https://example.com for more."
        result = _preprocess_for_speech(text)
        assert "Link to a page under docs.example.com with the text 'docs'" in result
        assert "example.com homepage" in result

    def test_preserves_text_without_links(self):
        text = "This is just regular text with no links at all."
        result = _preprocess_for_speech(text)
        assert result == text

    def test_handles_empty_string(self):
        result = _preprocess_for_speech("")
        assert result == ""

    def test_handles_link_at_start_of_text(self):
        text = "https://example.com is a great site."
        result = _preprocess_for_speech(text)
        assert result.startswith("example.com homepage")

    def test_handles_link_at_end_of_text(self):
        text = "Check this: https://example.com/page"
        result = _preprocess_for_speech(text)
        assert "a page under example.com" in result

    def test_removes_www_prefix_in_output(self):
        text = "[Site](https://www.example.com/path)"
        result = _preprocess_for_speech(text)
        # Should say "example.com" not "www.example.com"
        assert "www." not in result
        assert "example.com" in result


class TestEstimateTtsDuration:
    """Tests for TTS duration estimation (for audio buffer timing)."""

    def test_estimates_duration_based_on_word_count(self):
        # 175 WPM means 175 words takes 60 seconds
        # So 35 words should take ~12 seconds + buffer
        text = " ".join(["word"] * 35)
        duration = _estimate_tts_duration(text, 175)
        expected = (35 / 175) * 60 + AUDIO_BUFFER_DELAY_SEC
        assert abs(duration - expected) < 0.01

    def test_includes_audio_buffer_delay(self):
        # Even for short text, should include buffer delay
        text = "hello"
        duration = _estimate_tts_duration(text, 175)
        assert duration >= AUDIO_BUFFER_DELAY_SEC

    def test_uses_default_wpm_for_zero(self):
        text = "one two three four five"  # 5 words
        duration_zero = _estimate_tts_duration(text, 0)
        duration_default = _estimate_tts_duration(text, DEFAULT_WPM)
        assert duration_zero == duration_default

    def test_uses_default_wpm_for_negative(self):
        text = "one two three four five"
        duration_negative = _estimate_tts_duration(text, -100)
        duration_default = _estimate_tts_duration(text, DEFAULT_WPM)
        assert duration_negative == duration_default

    def test_faster_rate_means_shorter_duration(self):
        text = " ".join(["word"] * 50)
        slow_duration = _estimate_tts_duration(text, 100)
        fast_duration = _estimate_tts_duration(text, 200)
        assert fast_duration < slow_duration

    def test_longer_text_means_longer_duration(self):
        short_text = "hello world"
        long_text = " ".join(["word"] * 100)
        short_duration = _estimate_tts_duration(short_text, 175)
        long_duration = _estimate_tts_duration(long_text, 175)
        assert long_duration > short_duration

    def test_empty_text_returns_buffer_only(self):
        duration = _estimate_tts_duration("", 175)
        assert duration == AUDIO_BUFFER_DELAY_SEC

    def test_realistic_sentence_duration(self):
        # "Hello, how are you doing today?" is ~7 words at 175 WPM
        text = "Hello, how are you doing today?"
        duration = _estimate_tts_duration(text, 175)
        # Should be about 2.4 seconds (7/175*60) + 0.5 buffer = ~2.9 seconds
        assert 2.5 < duration < 3.5

