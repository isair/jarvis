"""Tests for integration_setup helpers."""

from __future__ import annotations

import pytest

from desktop_app.integration_setup import (
    gmail_mcp_config,
    infer_social_platform,
    parse_socials_text,
    socials_to_text,
)


@pytest.mark.unit
def test_gmail_mcp_config_includes_stdio_subcommand():
    cfg = gmail_mcp_config("client-id", "client-secret")
    assert cfg["args"] == ["-y", "google-workspace-mcp", "mcp"]


@pytest.mark.unit
class TestBusinessSocials:
    def test_infer_platform_from_url(self):
        assert infer_social_platform("https://www.instagram.com/cafe") == "instagram"
        assert infer_social_platform("https://facebook.com/page") == "facebook"

    def test_parse_socials_text(self):
        text = "Instagram | https://instagram.com/mycafe\nhttps://tiktok.com/@mycafe"
        out = parse_socials_text(text)
        assert len(out) == 2
        assert out[0]["platform"] == "instagram"
        assert out[1]["platform"] == "tiktok"

    def test_socials_round_trip(self):
        original = [
            {"platform": "instagram", "label": "Instagram", "url": "https://instagram.com/x"},
        ]
        assert parse_socials_text(socials_to_text(original)) == original
