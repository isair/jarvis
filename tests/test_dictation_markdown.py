"""Tests for the Markdown voice-note formatter (dictation/markdown.py)."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jarvis.dictation.markdown import format_markdown, format_markdown_lines


def test_heading_cues():
    assert format_markdown("new heading: Project Ideas") == "# Project Ideas"
    assert format_markdown("heading: Ideas") == "# Ideas"
    assert format_markdown("sub heading: This Week") == "## This Week"
    assert format_markdown("subheading: X") == "## X"


def test_bullet_cues():
    assert format_markdown("bullet point: buy milk") == "- buy milk"
    assert format_markdown("bullet: milk") == "- milk"


def test_numbered_cues():
    assert format_markdown("numbered: first step") == "1. first step"
    assert format_markdown("number: step") == "1. step"


def test_emphasis_cues():
    assert format_markdown("bold: important") == "**important**"
    assert format_markdown("italic: maybe") == "*maybe*"
    assert format_markdown("italics: x") == "*x*"
    assert format_markdown("code: pip install") == "`pip install`"


def test_quote_link_divider():
    assert format_markdown("quote: watch this") == "> watch this"
    assert format_markdown("link: Jarvis https://x.io") == "[Jarvis](https://x.io)"
    assert format_markdown("divider") == "---"
    assert format_markdown("horizontal rule") == "---"


def test_paragraph_break():
    assert format_markdown("new paragraph") == "\n\n"
    assert format_markdown("new line") == "\n\n"


def test_prose_untouched():
    assert format_markdown("the quick brown fox jumps") == "the quick brown fox jumps"
    # inline cue mid-sentence must NOT be converted
    assert format_markdown("I have a new heading for you about cats") == \
        "I have a new heading for you about cats"


def test_colon_optional():
    assert format_markdown("heading Ideas") == "# Ideas"


def test_auto_numbering_across_lines():
    out = format_markdown_lines(["numbered: first", "numbered: second", "numbered: third"])
    assert out == "1. first\n2. second\n3. third"


def test_counter_resets_on_prose():
    out = format_markdown_lines(["numbered: one", "some prose", "numbered: two"])
    assert out == "1. one\nsome prose\n1. two"


def test_paragraph_join():
    out = format_markdown_lines(["first line", "new paragraph", "second line"])
    assert out == "first line\n\nsecond line"


def test_bullets_in_lines():
    out = format_markdown_lines(["bullet: milk", "bullet: eggs"])
    assert out == "- milk\n- eggs"
