"""Tests for proactive Gmail draft suggestions."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_parse_suggestion_json_from_fenced_block():
    from jarvis.email_drafting import parse_draft_suggestion_response

    raw = (
        'Here you go:\n```json\n'
        '{"angles": ["Apstiprināt", "Noraidīt"], "subject": "Re: Test", '
        '"body": "Sveiki,\\n\\nPaldies."}\n```'
    )
    got = parse_draft_suggestion_response(raw)
    assert got is not None
    assert got["angles"] == ["Apstiprināt", "Noraidīt"]
    assert got["body"].startswith("Sveiki")


@pytest.mark.unit
def test_suggest_reply_uses_llm_and_fences_input():
    from jarvis.email_drafting import suggest_reply_for_message

    cfg = MagicMock(
        intent_judge_model="gemma4:e2b",
        ollama_base_url="http://127.0.0.1:11434",
        reply_language="lv",
        operator_name="Jansona kungs",
    )
    llm_json = json.dumps(
        {
            "angles": ["Pateikt paldies", "Noraidīt piekļuvi"],
            "subject": "Re: Konts",
            "body": "Labdien,\n\nPaldies par ziņu.",
        },
        ensure_ascii=False,
    )
    with patch("jarvis.email_drafting.call_llm_direct", return_value=llm_json):
        got = suggest_reply_for_message(
            cfg,
            {
                "from": "Google <no-reply@accounts.google.com>",
                "subject": "Konta piekļuve",
                "snippet": "Jūs atļāvāt lietotnei iOS piekļūt…",
            },
        )
    assert got is not None
    assert len(got["angles"]) == 2
    assert "Paldies" in got["body"]


@pytest.mark.unit
def test_reply_language_english_when_latvian_quality_only():
    from jarvis.email_drafting import _reply_language

    cfg = MagicMock(reply_language="", latvian_quality_enabled=True)
    assert _reply_language(cfg) == "en"


@pytest.mark.unit
def test_enrich_gmail_messages_skips_when_disabled():
    from jarvis.email_drafting import enrich_gmail_messages

    cfg = MagicMock(sulainis_email_draft_suggestions=False)
    msgs = [{"from": "a", "subject": "s", "snippet": "x"}]
    out = enrich_gmail_messages(cfg, msgs)
    assert out == msgs
    assert "draft_suggestion" not in out[0]


@pytest.mark.unit
def test_find_gmail_draft_tool():
    from jarvis.email_drafting import find_gmail_draft_tool

    catalog = [
        "google_workspace__searchGmail",
        "google_workspace__createGmailDraft",
        "stop",
    ]
    assert find_gmail_draft_tool(catalog) == "google_workspace__createGmailDraft"
