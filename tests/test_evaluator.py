"""Unit tests for the agentic-loop turn evaluator."""

from unittest.mock import patch

import pytest

from jarvis.reply.evaluator import evaluate_turn, EvaluatorResult, _parse_result


class TestParseResult:
    def test_parses_terminal_true(self):
        res = _parse_result('{"terminal": true, "nudge": "", "reason": "done"}')
        assert res.terminal is True
        assert res.nudge == ""

    def test_parses_continue_with_nudge(self):
        res = _parse_result(
            '{"terminal": false, "nudge": "Call openApp with target=YouTube", '
            '"reason": "agent offered instead of acting"}'
        )
        assert res.terminal is False
        assert res.nudge == "Call openApp with target=YouTube"
        assert "offered" in res.reason

    def test_fails_open_to_terminal_on_garbage(self):
        res = _parse_result("not JSON at all")
        assert res.terminal is True
        assert res.reason == "evaluator_failed_open"

    def test_strips_markdown_fences(self):
        res = _parse_result(
            '```json\n{"terminal": true, "nudge": "", "reason": "ok"}\n```'
        )
        assert res.terminal is True

    def test_extracts_embedded_json(self):
        res = _parse_result(
            'Here: {"terminal": false, "nudge": "use X", "reason": "r"} done'
        )
        assert res.terminal is False
        assert res.nudge == "use X"

    def test_missing_terminal_field_fails_open_to_terminal(self):
        res = _parse_result('{"nudge": "x", "reason": "y"}')
        assert res.terminal is True
        assert res.reason == "evaluator_failed_open"

    def test_non_bool_terminal_fails_open_to_terminal(self):
        res = _parse_result('{"terminal": "yes", "nudge": "", "reason": ""}')
        assert res.terminal is True


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

    def test_terminal_path(self):
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            return_value='{"terminal": true, "nudge": "", "reason": "done"}',
        ):
            res = evaluate_turn(
                "what's 2+2?", "4.", [("calc", "do maths")], 1, self._cfg()
            )
        assert res.terminal is True
        assert res.nudge == ""

    def test_continue_with_nudge(self):
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            return_value=(
                '{"terminal": false, "nudge": "Invoke openApp with '
                'target=YouTube", "reason": "offered instead of acted"}'
            ),
        ):
            res = evaluate_turn(
                "open youtube",
                "I can navigate you to YouTube homepage.",
                [("openApp", "Open an application"), ("stop", "stop sentinel")],
                1,
                self._cfg(),
            )
        assert res.terminal is False
        assert "openApp" in res.nudge

    def test_parse_failure_fails_open_to_terminal(self):
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            return_value="not a valid response",
        ):
            res = evaluate_turn("q", "r", [], 1, self._cfg())
        assert res.terminal is True
        assert res.reason == "evaluator_failed_open"

    def test_timeout_or_exception_fails_open_to_terminal(self):
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            side_effect=TimeoutError("slow"),
        ):
            res = evaluate_turn("q", "r", [], 1, self._cfg())
        assert res.terminal is True
        assert res.reason == "evaluator_failed_open"

    def test_missing_config_fails_open_to_terminal(self):
        cfg = self._cfg(ollama_base_url="", ollama_chat_model="")
        res = evaluate_turn("q", "r", [], 1, cfg)
        assert res.terminal is True
        assert res.reason == "evaluator_failed_open"

    def test_connection_error_fails_open_to_terminal(self):
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            side_effect=ConnectionError("ollama down"),
        ):
            res = evaluate_turn("q", "r", [], 1, self._cfg())
        assert res.terminal is True

    def test_redacts_email_in_prompt(self):
        """Assistant response echoing an email is scrubbed before the LLM call."""
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return '{"terminal": true, "nudge": "", "reason": ""}'

        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            side_effect=_capture,
        ):
            evaluate_turn(
                "who is alice?",
                "Her email is alice@example.com and she lives in London.",
                [],
                1,
                self._cfg(),
            )
        sent = captured.get("user_content", "")
        assert "alice@example.com" not in sent
        assert "[REDACTED_EMAIL]" in sent

    def test_available_tools_appear_in_prompt(self):
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return '{"terminal": true, "nudge": "", "reason": ""}'

        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            side_effect=_capture,
        ):
            evaluate_turn(
                "open youtube",
                "I can help you find YouTube.",
                [
                    ("openApp", "Open an application by name"),
                    ("webSearch", "Search the web"),
                ],
                1,
                self._cfg(),
            )
        sent = captured.get("user_content", "")
        assert "openApp" in sent
        assert "Open an application by name" in sent
        assert "webSearch" in sent

    def test_evaluator_model_override_used(self):
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return '{"terminal": true, "nudge": "", "reason": ""}'

        cfg = self._cfg(
            evaluator_model="dedicated-evaluator",
            intent_judge_model="judge-model",
            ollama_chat_model="chat-model",
        )
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            side_effect=_capture,
        ):
            evaluate_turn("q", "r", [], 1, cfg)
        assert captured.get("chat_model") == "dedicated-evaluator"

    def test_evaluator_model_falls_back_to_intent_judge(self):
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return '{"terminal": true, "nudge": "", "reason": ""}'

        cfg = self._cfg(
            evaluator_model="",
            intent_judge_model="judge-model",
            ollama_chat_model="chat-model",
        )
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            side_effect=_capture,
        ):
            evaluate_turn("q", "r", [], 1, cfg)
        assert captured.get("chat_model") == "judge-model"
