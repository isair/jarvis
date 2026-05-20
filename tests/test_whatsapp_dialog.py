"""Tests for WhatsApp setup dialog parent resolution."""

from __future__ import annotations

import pytest

from desktop_app.whatsapp_setup_dialog import _top_level_parent


@pytest.mark.unit
class TestWhatsAppDialogParent:
    def test_none_parent(self):
        assert _top_level_parent(None) is None
