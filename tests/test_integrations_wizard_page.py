"""Tests for setup wizard integrations page."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.mark.unit
class TestIntegrationsPageInit:
    def test_constructs_gmail_fields(self, qapp):
        from desktop_app.integrations_wizard_page import IntegrationsPage

        with patch.object(IntegrationsPage, "_mcp_enabled", return_value=False):
            with patch.object(IntegrationsPage, "_load_gmail_fields") as load_gmail:
                page = IntegrationsPage()
                load_gmail.assert_called_once()
        assert hasattr(page, "_gmail_id")
        assert hasattr(page, "_gmail_secret")

    def test_load_gmail_fields_reads_config(self, qapp):
        from desktop_app.integrations_wizard_page import IntegrationsPage

        config = {
            "mcps": {
                "google_workspace": {
                    "env": {
                        "GOOGLE_CLIENT_ID": "id-123",
                        "GOOGLE_CLIENT_SECRET": "sec-456",
                    }
                }
            }
        }
        with patch.object(IntegrationsPage, "_mcp_enabled", return_value=True):
            with patch.object(IntegrationsPage, "_load_config", return_value=config):
                with patch.object(IntegrationsPage, "_load_pages_field"):
                    page = IntegrationsPage()
        assert page._gmail_id.text() == "id-123"
        assert page._gmail_secret.text() == "sec-456"
