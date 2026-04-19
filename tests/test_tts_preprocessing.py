"""Tests for TTS link preprocessing functionality."""

import pytest
from src.jarvis.output.tts import (
    _preprocess_for_speech,
    _strip_markdown_for_speech,
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


class TestStripMarkdownForSpeech:
    """Tests that markdown formatting is stripped before TTS reads the text aloud.

    Piper and similar TTS engines read literal characters — "**bold**" becomes
    "asterisk asterisk bold asterisk asterisk" if the markers aren't stripped.
    """

    def test_strips_bold_asterisks(self):
        assert _strip_markdown_for_speech("this is **important** info") == "this is important info"

    def test_strips_bold_underscores(self):
        assert _strip_markdown_for_speech("this is __important__ info") == "this is important info"

    def test_strips_italic_asterisks(self):
        assert _strip_markdown_for_speech("this is *emphasised* text") == "this is emphasised text"

    def test_strips_italic_underscores(self):
        assert _strip_markdown_for_speech("this is _emphasised_ text") == "this is emphasised text"

    def test_preserves_word_internal_underscores(self):
        # Variable-name-style underscores must survive so spoken code/identifiers
        # aren't mangled into concatenated words.
        assert _strip_markdown_for_speech("call my_function now") == "call my_function now"

    def test_strips_strikethrough(self):
        assert _strip_markdown_for_speech("was ~~wrong~~ right") == "was wrong right"

    def test_strips_inline_code(self):
        assert _strip_markdown_for_speech("run `ls -la` in the shell") == "run ls -la in the shell"

    def test_strips_fenced_code_block(self):
        text = "here is some code:\n```python\nprint('hi')\n```\ndone"
        result = _strip_markdown_for_speech(text)
        assert "```" not in result
        assert "print('hi')" in result

    def test_strips_heading_markers(self):
        text = "# Title\n## Subtitle\nbody"
        result = _strip_markdown_for_speech(text)
        assert "Title" in result
        assert "Subtitle" in result
        assert "#" not in result

    def test_strips_bullet_list_markers(self):
        text = "- first item\n- second item\n* third item"
        result = _strip_markdown_for_speech(text)
        for item in ("first item", "second item", "third item"):
            assert item in result
        assert "- " not in result
        assert "* " not in result

    def test_strips_numbered_list_markers(self):
        text = "1. first\n2. second\n3) third"
        result = _strip_markdown_for_speech(text)
        for item in ("first", "second", "third"):
            assert item in result
        # No leading digit-and-punct sequences remain.
        assert "1." not in result
        assert "3)" not in result

    def test_preserves_plain_text(self):
        text = "hello there, how are you today?"
        assert _strip_markdown_for_speech(text) == text

    def test_handles_empty_string(self):
        assert _strip_markdown_for_speech("") == ""

    def test_real_world_combined_case(self):
        # The exact failure case from the field session: model produced a
        # bulleted list with bolded items; TTS spoke "asterisk asterisk" for
        # each one. After stripping, the text should be speakable plain prose.
        text = (
            "1. **Find information about the movie** (like plot, cast, release date)?\n"
            "2. **Watch the movie?**\n"
            "3. **Find a link to the movie?**"
        )
        result = _strip_markdown_for_speech(text)
        assert "*" not in result
        assert "**" not in result
        for fragment in ("Find information about the movie", "Watch the movie", "Find a link to the movie"):
            assert fragment in result

    def test_preprocess_strips_markdown_end_to_end(self):
        # Full pipeline: URL handling + markdown stripping in one call.
        text = "See **[the docs](https://docs.example.com/api)** for details"
        result = _preprocess_for_speech(text)
        assert "**" not in result
        assert "Link to a page under docs.example.com" in result

    def test_preserves_isolated_year_at_line_start(self):
        # True list detection: a single line beginning with "YYYY. " is prose,
        # not a one-item numbered list. "2024. The year..." must survive intact.
        text = "2024. The year the breakthrough happened"
        assert _strip_markdown_for_speech(text) == text

    def test_preserves_single_numbered_line_as_prose(self):
        # A lone line like "1. done" with no sibling list items is treated as
        # prose. Mildly odd if it was intended as a one-item list, but safer
        # than mangling prose that coincidentally starts with a digit.
        text = "1. done and dusted"
        assert _strip_markdown_for_speech(text) == text

    def test_strips_numbered_list_when_grouped(self):
        # Two adjacent numbered lines form a real list and get stripped.
        text = "1. first\n2. second"
        result = _strip_markdown_for_speech(text)
        assert result == "first\nsecond"

    def test_does_not_strip_large_numbers_as_list_markers(self):
        # Large integers (years, counts) are never list markers, even if two
        # adjacent lines happen to start with them.
        text = "2023. The prior year\n2024. The current year"
        result = _strip_markdown_for_speech(text)
        assert "2023." in result
        assert "2024." in result

    def test_strips_blockquote_markers(self):
        text = "> a quoted line\n> another quote"
        result = _strip_markdown_for_speech(text)
        assert result == "a quoted line\nanother quote"

    def test_strips_setext_heading_underlines(self):
        # Setext-style headings use === or --- under the title line.
        text = "Main Title\n==========\nbody text\n\nSubtitle\n--------\nmore body"
        result = _strip_markdown_for_speech(text)
        assert "=====" not in result
        assert "-----" not in result
        assert "Main Title" in result
        assert "Subtitle" in result
        assert "body text" in result

    def test_strips_html_tags(self):
        text = "this is <b>bold</b> and <em>italic</em> text"
        result = _strip_markdown_for_speech(text)
        assert result == "this is bold and italic text"


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

