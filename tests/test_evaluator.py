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
