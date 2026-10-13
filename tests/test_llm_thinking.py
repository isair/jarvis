"""
Tests for the LLM thinking mode feature.

Verifies that the ``llm_thinking_enabled`` config option correctly controls
the ``think`` parameter sent to Ollama across all call sites.
"""

import json
from unittest.mock import MagicMock

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


@pytest.mark.unit
@pytest.mark.usefixtures('no_network')
class TestCallLlmDirectFailurePaths:
    """Failed local requests stay private and permit a subsequent answer."""

    @pytest.mark.parametrize('entry', ['ollama', 'openai_compatible', 'helper'])
    @pytest.mark.parametrize('stage,error_type', [
        ('request', 'Timeout'), ('request', 'ConnectionError'),
        ('status', 'HTTPError'), ('decode', 'ValueError'),
    ])
    def test_failure_diagnostics_and_recovery(self, monkeypatch, capsys, entry, stage, error_type):
        from contextlib import nullcontext
        import requests
        from jarvis import llm

        monkeypatch.setenv('JARVIS_VOICE_DEBUG', '1')
        endpoint = 'http://127.0.0.1:1/private-endpoint?token=private-token'
        private_text = 'Synthetic private question'
        message = f'{endpoint}, {private_text}'
        error_class = ValueError if error_type == 'ValueError' else getattr(requests.exceptions, error_type)
        error = error_class(message)
        response = MagicMock()
        response.json.return_value = (
            {'choices': [{'message': {'content': 'Complete synthetic answer'}, 'finish_reason': 'stop'}]}
            if entry == 'openai_compatible' else
            {'message': {'content': 'Complete synthetic answer'}, 'done': True}
        )
        if stage == 'status':
            response.raise_for_status.side_effect = [error, None]
        elif stage == 'decode':
            response.json.side_effect = [error, response.json.return_value]
        failed = False

        def post(*args, **kwargs):
            nonlocal failed
            if stage == 'request' and not failed:
                failed = True
                raise error
            return nullcontext(response)

        monkeypatch.setattr(llm.http, 'post', post)
        if entry == 'helper':
            call = lambda: llm.call_llm_direct(endpoint, 'synthetic-model', 'Synthetic instructions',
                                             private_text, timeout_sec=0.25)
        else:
            backend_type = llm.OllamaBackend if entry == 'ollama' else llm.OpenAICompatibleBackend
            backend = backend_type(endpoint)
            call = lambda: backend.direct('synthetic-model', 'Synthetic instructions',
                                          private_text, timeout_sec=0.25)
        assert call() is None
        captured = capsys.readouterr()
        diagnostic = captured.out + captured.err
        assert endpoint not in diagnostic
        assert 'private-token' not in diagnostic
        assert private_text not in diagnostic
        if error_type == 'Timeout':
            assert 'timeout' in diagnostic and '0.25' in diagnostic
        else:
            assert error_type in diagnostic
        assert call() == 'Complete synthetic answer'
