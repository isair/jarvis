"""
Tests for the LLM thinking mode feature.

Verifies that the ``llm_thinking_enabled`` config option correctly controls
the ``think`` parameter sent to Ollama across all call sites.
"""

import json
import threading
from unittest.mock import patch, MagicMock

import pytest

from jarvis.config import get_default_config


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

class TestThinkingConfig:
    """Config layer tests for thinking settings."""

    def test_default_config_has_chat_thinking_disabled(self):
        """llm_thinking_enabled should default to False."""
        config = get_default_config()
        assert "llm_thinking_enabled" in config
        assert config["llm_thinking_enabled"] is False

    def test_default_config_has_intent_judge_thinking_disabled(self):
        """intent_judge_thinking_enabled should default to False."""
        config = get_default_config()
        assert "intent_judge_thinking_enabled" in config
        assert config["intent_judge_thinking_enabled"] is False

    def test_default_config_has_dictation_thinking_disabled(self):
        """dictation_thinking_enabled should default to False."""
        config = get_default_config()
        assert "dictation_thinking_enabled" in config
        assert config["dictation_thinking_enabled"] is False


# ---------------------------------------------------------------------------
# llm.py — payload construction
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.usefixtures('no_network')
class TestLlmThinkingPayload:
    """Public local-model helpers forward reasoning and deliver usable content."""

    @pytest.mark.parametrize('operation', ['direct', 'streaming', 'chat'])
    @pytest.mark.parametrize('thinking', [None, False, True], ids=['default', 'disabled', 'enabled'])
    def test_reasoning_and_answer_delivery(self, monkeypatch, operation, thinking):
        from contextlib import nullcontext
        from inspect import signature
        from jarvis import llm
        from jarvis.llm import http

        helpers = {'direct': llm.call_llm_direct, 'streaming': llm.call_llm_streaming,
                   'chat': llm.chat_with_messages}
        helper = helpers[operation]
        kwargs = {} if thinking is None else {'thinking': thinking}
        expected = signature(helper).parameters['thinking'].default if thinking is None else thinking
        answer = 'Synthetic answer'
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {'message': {'content': answer}, 'done': True}
        response.iter_lines.return_value = [
            json.dumps({'message': {'thinking': 'Synthetic reasoning'}}).encode(),
            json.dumps({'message': {'content': answer}, 'done': True}).encode(),
        ]

        def post(url, **options):
            payload = options['json']
            assert url == 'http://127.0.0.1:1/api/chat'
            assert payload['model'] == 'synthetic-model'
            assert payload['messages'][-1]['content'] == 'Synthetic question'
            assert payload['think'] is expected
            return nullcontext(response)

        monkeypatch.setattr(http, 'post', post)
        seen = []
        if operation == 'chat':
            result = helper('http://127.0.0.1:1', 'synthetic-model',
                            [{'role': 'user', 'content': 'Synthetic question'}], **kwargs)
            assert result and result['message']['content'] == answer
        else:
            if operation == 'streaming':
                kwargs['on_token'] = seen.append
            result = helper('http://127.0.0.1:1', 'synthetic-model',
                            'Synthetic instructions', 'Synthetic question', **kwargs)
            assert result == answer
            if operation == 'streaming':
                assert seen == [answer]


# ---------------------------------------------------------------------------
# Intent judge
# ---------------------------------------------------------------------------

class TestIntentJudgeThinking:
    """Intent judge respects the thinking config."""

    def test_config_default_thinking_false(self):
        from jarvis.listening.intent_judge import IntentJudgeConfig
        config = IntentJudgeConfig()
        assert config.thinking is False

    def test_config_accepts_thinking_true(self):
        from jarvis.listening.intent_judge import IntentJudgeConfig
        config = IntentJudgeConfig(thinking=True)
        assert config.thinking is True

    def test_create_intent_judge_passes_thinking(self):
        """create_intent_judge should read intent_judge_thinking_enabled from cfg."""
        from jarvis.listening.intent_judge import create_intent_judge

        cfg = MagicMock()
        cfg.wake_word = "jarvis"
        cfg.wake_aliases = []
        cfg.fast_model = "gemma4:e2b"
        cfg.ollama_base_url = "http://localhost:11434"
        cfg.intent_judge_timeout_sec = 10.0
        cfg.intent_judge_thinking_enabled = True

        judge = create_intent_judge(cfg)
        assert judge is not None
        assert judge.config.thinking is True

    def test_create_intent_judge_independent_from_chat_thinking(self):
        """Intent judge thinking should be independent from chat thinking."""
        from jarvis.listening.intent_judge import create_intent_judge

        cfg = MagicMock()
        cfg.wake_word = "jarvis"
        cfg.wake_aliases = []
        cfg.fast_model = "gemma4:e2b"
        cfg.ollama_base_url = "http://localhost:11434"
        cfg.intent_judge_timeout_sec = 10.0
        cfg.llm_thinking_enabled = True
        cfg.intent_judge_thinking_enabled = False

        judge = create_intent_judge(cfg)
        assert judge.config.thinking is False


# ---------------------------------------------------------------------------
# Dictation engine
# ---------------------------------------------------------------------------

class TestDictationThinking:
    """Dictation engine respects the thinking config."""

    def test_engine_stores_thinking(self):
        from jarvis.dictation.dictation_engine import DictationEngine

        engine = DictationEngine(
            whisper_model_ref=lambda: MagicMock(),
            whisper_backend_ref=lambda: "faster-whisper",
            mlx_repo_ref=lambda: None,
            hotkey="ctrl+shift+d",
            sample_rate=16000,
            transcribe_lock=threading.Lock(),
            thinking=True,
        )
        assert engine._thinking is True


# ---------------------------------------------------------------------------
# Settings window metadata
# ---------------------------------------------------------------------------

class TestSettingsWindowThinking:
    """Settings window includes all three thinking fields."""

    def test_field_metadata_includes_chat_thinking(self):
        from desktop_app.settings_window import FIELD_METADATA
        keys = [fm.key for fm in FIELD_METADATA]
        assert "llm_thinking_enabled" in keys

    def test_chat_thinking_field_is_bool_in_llm_category(self):
        from desktop_app.settings_window import FIELD_METADATA
        field = next(fm for fm in FIELD_METADATA if fm.key == "llm_thinking_enabled")
        assert field.field_type == "bool"
        assert field.category == "llm"

    def test_field_metadata_includes_intent_judge_thinking(self):
        from desktop_app.settings_window import FIELD_METADATA
        keys = [fm.key for fm in FIELD_METADATA]
        assert "intent_judge_thinking_enabled" in keys

    def test_intent_judge_thinking_field_is_bool_in_llm_category(self):
        from desktop_app.settings_window import FIELD_METADATA
        field = next(fm for fm in FIELD_METADATA if fm.key == "intent_judge_thinking_enabled")
        assert field.field_type == "bool"
        assert field.category == "llm"

    def test_field_metadata_includes_dictation_thinking(self):
        from desktop_app.settings_window import FIELD_METADATA
        keys = [fm.key for fm in FIELD_METADATA]
        assert "dictation_thinking_enabled" in keys

    def test_dictation_thinking_field_is_bool_in_features_category(self):
        from desktop_app.settings_window import FIELD_METADATA
        field = next(fm for fm in FIELD_METADATA if fm.key == "dictation_thinking_enabled")
        assert field.field_type == "bool"
        assert field.category == "features"


# ---------------------------------------------------------------------------
# Timeout / error paths — regression test for missing debug_log import
# ---------------------------------------------------------------------------

class TestCallLlmDirectFailurePaths:
    """Exercises the exception branches in call_llm_direct.

    These branches call debug_log; a missing import would surface here as a
    NameError instead of the intended graceful None return.
    """

    def test_timeout_returns_none(self):
        import requests
        from jarvis.llm import call_llm_direct

        with patch('jarvis.llm.http.post', side_effect=requests.exceptions.Timeout):
            result = call_llm_direct(
                base_url="http://localhost:99999",
                chat_model="test-model",
                system_prompt="sys",
                user_content="hi",
                timeout_sec=0.1,
            )
        assert result is None

    def test_request_exception_returns_none(self):
        from jarvis.llm import call_llm_direct

        with patch('jarvis.llm.http.post', side_effect=ConnectionError("boom")):
            result = call_llm_direct(
                base_url="http://localhost:99999",
                chat_model="test-model",
                system_prompt="sys",
                user_content="hi",
                timeout_sec=0.1,
            )
        assert result is None
