"""
Unit tests for the prompts module.

Tests model size detection and prompt component selection.
"""

import pytest


class TestModelSizeDetection:
    """Tests for detect_model_size function."""

    @pytest.mark.parametrize("model_name,expected_small", [
        # Small models (should return SMALL)
        ("gemma4", True),
        ("gemma4:e2b", True),
        ("gemma4:e4b", True),
        ("llama3.2:3b", True),
        ("llama3.2:1b", True),
        ("mistral:7b", True),
        ("gemma:7b", True),
        ("phi3:3b", True),
        ("qwen2:7b", True),
        # Various separators
        ("model-3b-instruct", True),
        ("model_1b_chat", True),
        # Large models (should return LARGE)
        ("gpt-oss:20b", False),
        ("llama3.1:8b", False),
        ("qwen2.5:14b", False),
        ("gemma2:27b", False),
        ("llama3:70b", False),
        ("mixtral:8x7b", False),  # 8x7b is effectively large
        # Edge cases
        (None, False),  # None defaults to LARGE
        ("", False),    # Empty defaults to LARGE
        ("custom-model", False),  # No size indicator = LARGE
    ])
    def test_detect_model_size(self, model_name, expected_small):
        """Model size detection works for various model names."""
        from jarvis.reply.prompts import detect_model_size, ModelSize

        result = detect_model_size(model_name)
        expected = ModelSize.SMALL if expected_small else ModelSize.LARGE

        assert result == expected, \
            f"Expected {expected.value} for '{model_name}', got {result.value}"


class TestPromptComponents:
    """Tests for get_system_prompts function."""

    def test_small_model_has_tool_constraints(self):
        """Small models get explicit tool constraints covering every rule.

        Constraints are phrased language-agnostically (per CLAUDE.md: no
        hardcoded English greetings / English unit names / etc.), so we
        assert against BEHAVIOURAL sections, not specific tokens in one
        language.
        """
        from jarvis.reply.prompts import get_system_prompts, ModelSize

        prompts = get_system_prompts(ModelSize.SMALL)

        assert prompts.tool_constraints is not None
        text = prompts.tool_constraints.lower()
        # Each section header must be present — they structure the rules.
        for section in (
            "greeting handling",
            "user instructions",
            "unknown named entities",
            "arguments the tool can auto-derive",
        ):
            assert section in text, f"Missing section {section!r} in small-model constraints"

    def test_large_model_has_tool_constraints(self):
        """Large models also get constraints — a shorter restatement of the
        named-entity and auto-derive rules. gpt-oss:20b and similar
        confabulate specifics and occasionally ask for tool args the tool
        already auto-derives, so the large variant is not a no-op."""
        from jarvis.reply.prompts import get_system_prompts, ModelSize

        prompts = get_system_prompts(ModelSize.LARGE)

        assert prompts.tool_constraints is not None
        text = prompts.tool_constraints.lower()
        assert "unknown named entities" in text
        assert "arguments the tool can auto-derive" in text

    def test_small_model_balanced_incentives(self):
        """Small models get balanced tool incentives - use tools but not for greetings."""
        from jarvis.reply.prompts import get_system_prompts, ModelSize

        prompts = get_system_prompts(ModelSize.SMALL)

        # Should encourage tool use for legitimate cases
        assert "use tools" in prompts.tool_incentives.lower()
        # But mention greetings specifically
        assert "greeting" in prompts.tool_incentives.lower()

    def test_large_model_proactive_incentives(self):
        """Large models get proactive tool incentives."""
        from jarvis.reply.prompts import get_system_prompts, ModelSize

        prompts = get_system_prompts(ModelSize.LARGE)

        # Should encourage proactive tool use
        assert "proactively" in prompts.tool_incentives.lower()

    def test_both_sizes_have_core_components(self):
        """Both model sizes have the core prompt components."""
        from jarvis.reply.prompts import get_system_prompts, ModelSize

        for size in [ModelSize.SMALL, ModelSize.LARGE]:
            prompts = get_system_prompts(size)

            # All core components should be present
            assert prompts.asr_note, f"{size.value} missing asr_note"
            assert prompts.inference_guidance, f"{size.value} missing inference_guidance"
            assert prompts.tool_incentives, f"{size.value} missing tool_incentives"
            assert prompts.voice_style, f"{size.value} missing voice_style"
            assert prompts.tool_guidance, f"{size.value} missing tool_guidance"

    def test_to_list_returns_non_empty_strings(self):
        """to_list() returns only non-empty prompt strings."""
        from jarvis.reply.prompts import get_system_prompts, ModelSize

        for size in [ModelSize.SMALL, ModelSize.LARGE]:
            prompts = get_system_prompts(size)
            prompt_list = prompts.to_list()

            assert len(prompt_list) >= 5, f"{size.value} should have at least 5 components"
            assert all(isinstance(p, str) and p for p in prompt_list), \
                f"{size.value} has empty or non-string components"

    def test_small_model_to_list_includes_constraints(self):
        """Small model to_list() includes tool constraints."""
        from jarvis.reply.prompts import get_system_prompts, ModelSize

        prompts = get_system_prompts(ModelSize.SMALL)
        prompt_list = prompts.to_list()

        # Should have more items due to tool_constraints
        assert len(prompt_list) == 6

        # Tool constraints should be in the list (greeting handling)
        has_constraints = any("greeting" in p.lower() for p in prompt_list)
        assert has_constraints, "Small model should include greeting constraints"

    def test_large_model_to_list_includes_constraints(self):
        """Large model to_list() now includes tool constraints too. The large
        variant covers the named-entity and auto-derive rules — without it,
        larger models confabulate for unfamiliar entities or nag the user
        for args the tool already auto-derives (field failure 2026-04-20).
        """
        from jarvis.reply.prompts import get_system_prompts, ModelSize

        prompts = get_system_prompts(ModelSize.LARGE)
        prompt_list = prompts.to_list()

        # Both sizes now carry all 6 components.
        assert len(prompt_list) == 6

        has_named_entity_rule = any("UNKNOWN NAMED ENTITIES" in p for p in prompt_list)
        assert has_named_entity_rule, "Large model should include the named-entity rule"
        has_auto_derive_rule = any("AUTO-DERIVE" in p for p in prompt_list)
        assert has_auto_derive_rule, "Large model should include the auto-derive rule"


class TestPromptLanguageAgnosticism:
    """Tests that prompts are language-agnostic."""

    def test_greeting_rule_is_language_agnostic(self):
        """Greeting handling must NOT list language-specific greeting tokens.

        CLAUDE.md forbids hardcoded language patterns — the assistant
        supports arbitrary languages, and listing 'hello' / 'ni hao' /
        'bonjour' both biases the model toward those languages and gives a
        false sense of coverage. The new rule describes the SEMANTIC
        category ("a greeting or casual social phrase, whatever language"),
        letting the model rely on its own multilingual understanding."""
        from jarvis.reply.prompts import get_system_prompts, ModelSize

        prompts = get_system_prompts(ModelSize.SMALL)
        constraints = prompts.tool_constraints.lower()

        # The section itself must be present.
        assert "greeting handling" in constraints

        # None of the old English-biased greeting tokens should be hard-coded
        # into the prompt any more.
        for token in ("ni hao", "bonjour", "hola", "merhaba", "ciao"):
            assert token not in constraints, (
                f"Stale language-specific token {token!r} is still hardcoded in "
                "the constraints — the rule should describe the category, not "
                "enumerate language-specific surface forms."
            )

        # The language-agnostic phrasing must be present.
        assert "whatever language" in constraints or "any language" in constraints

    def test_greeting_constraint_is_narrow(self):
        """Greeting constraint is narrowly scoped, not overly restrictive."""
        from jarvis.reply.prompts import get_system_prompts, ModelSize

        prompts = get_system_prompts(ModelSize.SMALL)
        constraints = prompts.tool_constraints.lower()

        # Should mention greetings specifically
        assert "greeting" in constraints
        # Should NOT have overly broad restrictions like "ONLY use tools when explicitly asked"
        # (This would hurt legitimate tool use for news, weather, etc.)
        assert "only when explicitly" not in constraints
