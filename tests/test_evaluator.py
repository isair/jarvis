"""Unit tests for the agentic-loop turn evaluator."""

from unittest.mock import patch

import pytest

from jarvis.reply.evaluator import evaluate_turn, EvaluatorResult, _parse_result


class TestParseResult:
    def test_parses_satisfied(self):
        res = _parse_result('{"terminal": true, "reason": "satisfied"}')
        assert res.terminal is True
        assert res.reason == "satisfied"

    def test_parses_needs_user_input(self):
        res = _parse_result(
            '{"terminal": true, "reason": "needs_user_input", '
            '"clarification_question": "Which city?"}'
        )
        assert res.terminal is True
        assert res.reason == "needs_user_input"
        assert res.clarification_question == "Which city?"

    def test_parses_continue(self):
        res = _parse_result('{"terminal": false, "reason": "continue"}')
        assert res.terminal is False
        assert res.reason == "continue"

    def test_fails_open_on_garbage(self):
        res = _parse_result("not JSON at all")
        assert res.reason == "continue"
        assert res.terminal is False

    def test_strips_markdown_fences(self):
        res = _parse_result('```json\n{"terminal": true, "reason": "satisfied"}\n```')
        assert res.reason == "satisfied"

    def test_extracts_embedded_json(self):
        res = _parse_result('Here you go: {"terminal": true, "reason": "satisfied"} done')
        assert res.reason == "satisfied"

    def test_unknown_reason_falls_back_to_continue(self):
        res = _parse_result('{"terminal": true, "reason": "maybe"}')
        assert res.reason == "continue"
        assert res.terminal is False


class TestEvaluateTurn:
    def _cfg(self, **overrides):
        class _C:
            ollama_base_url = "http://x"
            ollama_chat_model = "m"
            llm_digest_timeout_sec = 5.0
            llm_thinking_enabled = False
        c = _C()
        for k, v in overrides.items():
            setattr(c, k, v)
        return c

    def test_satisfied_path(self):
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            return_value='{"terminal": true, "reason": "satisfied"}',
        ):
            res = evaluate_turn("what's 2+2?", "4.", 1, self._cfg())
        assert res.reason == "satisfied"
        assert res.terminal is True

    def test_needs_user_input_path(self):
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            return_value='{"terminal": true, "reason": "needs_user_input"}',
        ):
            res = evaluate_turn("book me a table", "For how many people?", 1, self._cfg())
        assert res.reason == "needs_user_input"
        assert res.terminal is True

    def test_continue_path(self):
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            return_value='{"terminal": false, "reason": "continue"}',
        ):
            res = evaluate_turn("compare iphones", "Let me check.", 1, self._cfg())
        assert res.reason == "continue"
        assert res.terminal is False

    def test_parse_failure_fails_open_to_continue(self):
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            return_value="not a valid response",
        ):
            res = evaluate_turn("q", "r", 1, self._cfg())
        assert res.reason == "continue"
        assert res.terminal is False

    def test_timeout_or_exception_fails_open(self):
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            side_effect=TimeoutError("slow"),
        ):
            res = evaluate_turn("q", "r", 1, self._cfg())
        assert res.reason == "continue"
        assert res.terminal is False

    def test_missing_config_fails_open(self):
        cfg = self._cfg(ollama_base_url="", ollama_chat_model="")
        res = evaluate_turn("q", "r", 1, cfg)
        assert res.reason == "continue"
        assert res.terminal is False

    def test_connection_error_fails_open(self):
        """F11: transport-level failure must collapse to 'continue'."""
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            side_effect=ConnectionError("ollama down"),
        ):
            res = evaluate_turn("q", "r", 1, self._cfg())
        assert res.reason == "continue"
        assert res.terminal is False

    def test_redacts_email_in_prompt(self):
        """Assistant response echoing an email is scrubbed before the LLM call."""
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return '{"terminal": true, "reason": "satisfied"}'

        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            side_effect=_capture,
        ):
            evaluate_turn(
                "who is alice?",
                "Her email is alice@example.com and she lives in London.",
                1,
                self._cfg(),
            )
        sent = captured.get("user_content", "")
        assert "alice@example.com" not in sent
        assert "[REDACTED_EMAIL]" in sent

    def test_clarification_question_carried_through(self):
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            return_value=(
                '{"terminal": true, "reason": "needs_user_input", '
                '"clarification_question": "Which city did you mean?"}'
            ),
        ):
            res = evaluate_turn("weather?", "Where?", 1, self._cfg())
        assert res.reason == "needs_user_input"
        assert res.clarification_question == "Which city did you mean?"

    def test_evaluator_model_override_used(self):
        """evaluator_model wins over intent_judge_model and ollama_chat_model."""
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return '{"terminal": true, "reason": "satisfied"}'

        cfg = self._cfg(
            evaluator_model="dedicated-evaluator",
            intent_judge_model="judge-model",
            ollama_chat_model="chat-model",
        )
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            side_effect=_capture,
        ):
            evaluate_turn("q", "r", 1, cfg)
        assert captured.get("chat_model") == "dedicated-evaluator"

    def test_evaluator_model_falls_back_to_intent_judge(self):
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return '{"terminal": true, "reason": "satisfied"}'

        cfg = self._cfg(
            evaluator_model="",
            intent_judge_model="judge-model",
            ollama_chat_model="chat-model",
        )
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            side_effect=_capture,
        ):
            evaluate_turn("q", "r", 1, cfg)
        assert captured.get("chat_model") == "judge-model"
