"""Tests for desktop automation intent helpers."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_mentions_desktop_action_open_app():
    from jarvis.desktop_automation_intent import mentions_desktop_action

    assert mentions_desktop_action("please find Adobe Illustrator and open new file A4")
    assert mentions_desktop_action("atver Excel un izveido jaunu failu")


@pytest.mark.unit
def test_is_false_tool_refusal():
    from jarvis.desktop_automation_intent import is_false_tool_refusal

    assert is_false_tool_refusal(
        "I cannot directly open applications or create files on your system."
    )
    assert not is_false_tool_refusal("Launching Adobe Illustrator now.")


@pytest.mark.unit
def test_try_resolve_launch_illustrator():
    from jarvis.desktop_automation_intent import try_resolve_desktop_tool_call

    got = try_resolve_desktop_tool_call(
        "please find Adobe Illustrator and open new file A4 format",
        ["windows__App", "stop"],
    )
    assert got == ("windows__App", {"mode": "launch", "name": "Adobe Illustrator"})


@pytest.mark.unit
def test_windows_prompt_block_lists_tools():
    from jarvis.desktop_automation_intent import build_windows_automation_prompt_block

    block = build_windows_automation_prompt_block(["windows__App", "stop"])
    assert "windows__" in block
    assert "MUST call" in block
