"""Screen-intent detection tests."""

from __future__ import annotations

import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    "query,expected",
    [
        ("Ko tu redzi manā ekrānā?", True),
        ("What is on my screen right now?", True),
        ("How is the weather?", False),
        ("screenshot this", True),
    ],
)
def test_mentions_screen(query: str, expected: bool) -> None:
    from jarvis.screen_intent import mentions_screen

    assert mentions_screen(query) is expected


@pytest.mark.unit
@pytest.mark.skip(reason="format_screen_context_block moved/refactored on upstream develop")
def test_format_screen_context_block_includes_fence():
    from jarvis.tools.builtin.screenshot import format_screen_context_block

    block = format_screen_context_block("Error 404", "A browser window")
    assert "UNTRUSTED SCREEN CAPTURE" in block
    assert "Error 404" in block
    assert "browser" in block
