"""Tests for TTS link preprocessing functionality."""

import pytest
from src.jarvis.output.tts import _preprocess_for_speech, _extract_domain_description


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

