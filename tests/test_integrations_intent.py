"""Tests for Gmail integration intent helpers."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_mentions_gmail_query_access_question():
    from jarvis.integrations_intent import mentions_gmail_query

    assert mentions_gmail_query("can you see my gmail?")
    assert mentions_gmail_query("pārbaudi manu gmail inbox")


@pytest.mark.unit
def test_try_resolve_gmail_search_inbox():
    from jarvis.integrations_intent import try_resolve_gmail_tool_call

    got = try_resolve_gmail_tool_call(
        "can you see my gmail?",
        ["google_workspace__searchGmail", "stop"],
    )
    assert got is not None
    name, args = got
    assert name == "google_workspace__searchGmail"
    assert args["query"] == "in:inbox"
    assert args["maxResults"] == 8


@pytest.mark.unit
def test_gmail_prompt_block_lists_tools():
    from jarvis.integrations_intent import build_gmail_integration_prompt_block

    block = build_gmail_integration_prompt_block(
        ["google_workspace__searchGmail", "google_workspace__readGmailMessage"]
    )
    assert "google_workspace__" in block
    assert "MUST call" in block
    assert "lack access" in block


@pytest.mark.unit
def test_false_refusal_reused_for_gmail_denial():
    from jarvis.integrations_intent import is_false_tool_refusal

    assert is_false_tool_refusal("I do not have access to your Gmail account.")
