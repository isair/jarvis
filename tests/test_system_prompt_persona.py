"""Tests for persona / majordomo prompt overlay."""

from __future__ import annotations

import pytest

from jarvis.system_prompt import build_system_prompt


@pytest.mark.unit
class TestSystemPromptPersona:
    def test_default_witty_butler(self):
        prompt = build_system_prompt("Johnny")
        assert "Johnny" in prompt
        assert "dry, witty" in prompt.lower() or "witty" in prompt.lower()

    def test_formal_majordomo_includes_operator(self):
        prompt = build_system_prompt(
            "Johnny",
            operator_name="Mr. Johnson",
            persona_style="formal_majordomo",
        )
        assert "Mr. Johnson" in prompt
        assert "majordomo" in prompt.lower()

    def test_formal_majordomo_latvian(self):
        prompt = build_system_prompt(
            "Johnny",
            operator_name="Jansona kungs",
            persona_style="formal_majordomo",
            latvian_responses=True,
        )
        assert "Jansona kungs" in prompt
        assert "latviešu" in prompt.lower() or "Latvian" in prompt
