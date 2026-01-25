"""
Unit tests for the prompts module.

Tests model size detection and prompt component selection.
"""

import pytest


class TestModelSizeDetection:
    """Tests for detect_model_size function."""

    @pytest.mark.parametrize("model_name,expected_small", [
        # Small models (should return SMALL)
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
        """Small models get explicit tool constraints."""
        from jarvis.reply.prompts import get_system_prompts, ModelSize

        prompts = get_system_prompts(ModelSize.SMALL)

        assert prompts.tool_constraints is not None
        assert "greeting" in prompts.tool_constraints.lower()
        assert "ni hao" in prompts.tool_constraints.lower()

    def test_large_model_no_tool_constraints(self):
        """Large models don't have explicit tool constraints."""
        from jarvis.reply.prompts import get_system_prompts, ModelSize

        prompts = get_system_prompts(ModelSize.LARGE)

        assert prompts.tool_constraints is None

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

    def test_large_model_to_list_no_constraints(self):
        """Large model to_list() doesn't include explicit greeting constraints."""
        from jarvis.reply.prompts import get_system_prompts, ModelSize

        prompts = get_system_prompts(ModelSize.LARGE)
        prompt_list = prompts.to_list()

        # Should have fewer items (no tool_constraints)
        assert len(prompt_list) == 5

        # Explicit greeting constraints should NOT be in the list
        has_greeting_constraint = any("GREETING HANDLING" in p for p in prompt_list)
        assert not has_greeting_constraint, "Large model should not include explicit greeting constraints"


class TestPromptLanguageAgnosticism:
    """Tests that prompts are language-agnostic."""

    def test_greeting_examples_multilingual(self):
        """Tool constraints include greetings in multiple languages."""
        from jarvis.reply.prompts import get_system_prompts, ModelSize

        prompts = get_system_prompts(ModelSize.SMALL)
        constraints = prompts.tool_constraints.lower()

        # Should include examples in multiple languages
        languages_covered = [
            ("hello", "English"),
            ("ni hao", "Chinese"),
            ("bonjour", "French"),
            ("hola", "Spanish"),
            ("merhaba", "Turkish"),
            ("ciao", "Italian"),
        ]

        for greeting, language in languages_covered:
            assert greeting in constraints, \
                f"Missing {language} greeting '{greeting}' in constraints"

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
