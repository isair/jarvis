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

@pytest.mark.unit
@pytest.mark.usefixtures('no_network')
class TestSettingsWindowThinking:
    """Thinking switches remain visible, independent and persistent."""

    @pytest.mark.parametrize('key,category', [
        ('llm_thinking_enabled', 'llm'),
        ('intent_judge_thinking_enabled', 'llm'),
        ('dictation_thinking_enabled', 'features'),
    ], ids=['chat', 'intent', 'dictation'])
    @pytest.mark.parametrize('enabled', [False, True])
    def test_toggling_and_saving_preserves_other_values(
        self, settings_config_path, qapp, key, category, enabled,
    ):
        from desktop_app import settings_window as sw
        from jarvis.config import load_settings

        keys = ['llm_thinking_enabled', 'intent_judge_thinking_enabled', 'dictation_thinking_enabled']
        original = {name: not enabled for name in keys}
        original['opaque'] = {'text': 'İstanbul, 東京'}
        path = settings_config_path
        path.write_text(json.dumps(original), encoding='utf-8')
        window = sw.SettingsWindow()
        control = window._widgets[key]
        assert control.isChecked() is not enabled
        index = next(i for i, (name, _) in enumerate(sw.CATEGORIES) if name == category)
        window._sidebar.setCurrentRow(index)
        window.show()
        qapp.processEvents()
        assert control.isVisible()
        control.setChecked(enabled)
        window._on_save()

        saved = json.loads(path.read_text(encoding='utf-8'))
        cfg = load_settings()
        assert getattr(cfg, key) is enabled
        assert (key in saved) is (enabled != get_default_config()[key])
        assert all(getattr(cfg, name) is original[name] for name in keys if name != key)
        assert saved['opaque'] == original['opaque']


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
