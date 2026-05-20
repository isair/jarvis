"""Tests for Pulse dashboard sync helpers."""

from __future__ import annotations

import pytest

from desktop_app.pulse_api import unescape_pulse_text
from desktop_app.pulse_sync import (
    _parse_gmail_search_markdown,
    _parse_whatsapp_list_messages,
    build_whatsapp_comms_entry,
    format_whatsapp_display_from,
    sanitize_whatsapp_message_text,
)


@pytest.mark.unit
def test_unescape_pulse_text_decodes_html_entities():
    raw = "Vec&#x101;&#x137;i | Riga"
    assert "Vecāķi" in unescape_pulse_text(raw) or "Vec" in unescape_pulse_text(raw)


@pytest.mark.unit
def test_unescape_pulse_text_strips_tags():
    assert unescape_pulse_text("<b>Hello</b> world") == "Hello world"


_GMAIL_SAMPLE = """\
**Search Results for:** "in:inbox"
Total estimate: 2 messages

**1. Daily report**
   From: SumUp <no-reply@sumup.com>
   Date: Tue, 19 May 2026 06:29:34 +0000
   ID: abc123
   Labels: UNREAD, INBOX
   Preview: Your daily summary is attached.
   Link: https://mail.google.com/mail/#all/abc123

**2. Shift swap**
   From: When I Work <noreply@wheniwork.com>
   Date: Mon, 18 May 2026 13:36:13 +0000
   Preview: Shift swap request from Alex
   Link: https://mail.google.com/mail/#all/def456
"""


@pytest.mark.unit
def test_parse_gmail_search_markdown_extracts_messages():
    rows = _parse_gmail_search_markdown(_GMAIL_SAMPLE)
    assert len(rows) == 2
    assert rows[0]["subject"] == "Daily report"
    assert "SumUp" in rows[0]["from"]
    assert "daily summary" in rows[0]["snippet"].lower()
    assert rows[1]["subject"] == "Shift swap"
    assert "When I Work" in rows[1]["from"]


_WA_SAMPLE = (
    "[2026-05-19 17:56:40] Chat: 4x4 Par ap kur kad. From: 218923140157523@lid: Laaabs 😂\n"
    "[2026-05-19 17:55:53] Chat: Kalvis Grauds From: Me: Nu real time\n"
)


@pytest.mark.unit
def test_parse_whatsapp_list_messages_plain_text():
    rows = _parse_whatsapp_list_messages(_WA_SAMPLE)
    assert len(rows) == 2
    assert "4x4" in rows[0]["from"]
    assert "@lid" not in rows[0]["from"]
    assert "218923" not in rows[0]["from"]
    assert "Laaabs" in rows[0]["text"]
    assert rows[1]["from"] == "Kalvis Grauds"
    assert "real time" in rows[1]["text"]


@pytest.mark.unit
def test_format_whatsapp_display_from_hides_opaque_sender():
    assert format_whatsapp_display_from("4x4 Par ap kur kad.", "218923140157523@lid") == (
        "4x4 Par ap kur kad."
    )
    assert format_whatsapp_display_from("Kalvis Grauds", "Me") == "Kalvis Grauds"


@pytest.mark.unit
def test_build_whatsapp_comms_entry_hides_numeric_sender_and_image_meta():
    row = build_whatsapp_comms_entry(
        {
            "sender": "120363169319669622",
            "text": (
                "[image - Message ID: 3EB0473A7E331B784E3E0A - "
                "Chat JID: 120363169319669622@newsletter]"
            ),
        }
    )
    assert row["from"] == "Jaunumu kanāls"
    assert "Message ID" not in row["text"]
    assert "120363" not in row["text"]
    assert row["text"] == "📷 Attēls"


@pytest.mark.unit
def test_build_whatsapp_comms_entry_newsletter_text():
    row = build_whatsapp_comms_entry(
        {
            "sender": "120363169319669622",
            "text": "Carlos Alcaraz will miss Wimbledon due to injury.",
        }
    )
    assert row["from"] == "Jaunumu kanāls"
    assert "Carlos" in row["text"]


@pytest.mark.unit
def test_sanitize_whatsapp_message_text_plain_passthrough():
    assert sanitize_whatsapp_message_text("Sveiki") == "Sveiki"
