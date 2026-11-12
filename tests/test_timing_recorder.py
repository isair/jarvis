"""Timing qualification observes the same backend calls as production."""

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from jarvis import llm
from jarvis.llm.factory import get_llm_backend
from jarvis.llm.ollama import OllamaBackend
from jarvis.llm.openai_compatible import OpenAICompatibleBackend
from tests.performance.timing_recorder import TimingRecorder

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]
PROVIDERS = [('ollama', OllamaBackend), ('openai_compatible', OpenAICompatibleBackend)]


@pytest.mark.parametrize('provider,backend_class', PROVIDERS)
@pytest.mark.parametrize('method', ['direct', 'streaming', 'chat'])
def test_records_existing_and_factory_created_backends(monkeypatch, mock_config, provider, backend_class, method):
    mock_config.llm_provider = provider
    model = 'owned-timing-model'
    messages = [{'role': 'user', 'content': 'owned prompt'}]
    result = {'message': {'content': 'owned response'}} if method == 'chat' else 'owned response'

    def respond(self, chat_model, *args, **kwargs):
        return result

    monkeypatch.setattr(backend_class, method, respond)
    existing = get_llm_backend(mock_config)
    with TimingRecorder() as recorder:
        for backend in (existing, get_llm_backend(mock_config)):
            if method == 'chat':
                received = backend.chat(chat_model=model, messages=messages)
            else:
                received = getattr(backend, method)(model, 'owned system', 'owned prompt')
            assert received is result
    assert len(recorder.calls) == 2
    assert {call.model for call in recorder.calls} == {model}
    assert {call.provider for call in recorder.calls} == {backend_class.__name__}
    assert {call.response_chars for call in recorder.calls} == {len('owned response')}
    assert all(call.prompt_chars > 0 and call.duration_sec >= 0 for call in recorder.calls)
    assert getattr(backend_class, method) is respond


@pytest.mark.parametrize('provider,backend_class', PROVIDERS)
@pytest.mark.parametrize('result,outcome', [('answer', 'response'), ('', 'empty'), ('   ', 'empty'), (None, 'empty')])
def test_reports_empty_and_returned_responses(monkeypatch, mock_config, provider, backend_class, result, outcome):
    mock_config.llm_provider = provider
    monkeypatch.setattr(backend_class, 'direct', lambda *args, **kwargs: result)
    with TimingRecorder() as recorder:
        assert get_llm_backend(mock_config).direct('owned model', 'system', 'user') is result
    assert [call.outcome for call in recorder.calls] == [outcome]
    summary = next(iter(recorder.to_dict()['contexts'].values()))
    assert summary['outcomes'] == {outcome: 1}


@pytest.mark.parametrize('provider,backend_class', PROVIDERS)
def test_records_errors_and_later_calls_without_changing_exception(monkeypatch, mock_config, provider, backend_class):
    mock_config.llm_provider = provider
    failure = RuntimeError('owned failure')

    def respond(self, chat_model, *args, **kwargs):
        if chat_model == 'failed request':
            raise failure
        return 'answer'

    monkeypatch.setattr(backend_class, 'direct', respond)
    with TimingRecorder() as recorder:
        with pytest.raises(RuntimeError) as raised:
            get_llm_backend(mock_config).direct('failed request', 'system', 'user')
        assert raised.value is failure
        assert get_llm_backend(mock_config).direct('later request', 'system', 'user') == 'answer'
    assert [call.outcome for call in recorder.calls] == ['error', 'response']


def test_helper_and_nested_backend_calls_are_counted_once(monkeypatch):
    def chat(self, chat_model, messages, **kwargs):
        return {'message': {'content': 'answer'}}

    def direct(self, chat_model, system_prompt, user_content, **kwargs):
        return self.chat(chat_model, [{'role': 'user', 'content': user_content}])['message']['content']

    monkeypatch.setattr(OllamaBackend, 'direct', direct)
    monkeypatch.setattr(OllamaBackend, 'chat', chat)
    with TimingRecorder() as recorder:
        assert llm.call_llm_direct('http://owned.invalid', 'owned model', 'system', 'user') == 'answer'
    assert len(recorder.calls) == 1
    assert recorder.calls[0].model == 'owned model'


def test_concurrent_calls_remain_distinct(monkeypatch):
    ready = threading.Barrier(2)

    def direct(self, chat_model, system_prompt, user_content, **kwargs):
        ready.wait(timeout=5)
        return chat_model

    monkeypatch.setattr(OllamaBackend, 'direct', direct)
    with TimingRecorder() as recorder:
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures = [workers.submit(OllamaBackend('http://owned.invalid').direct, model, 'system', 'user')
                       for model in ('first', 'second')]
            assert [future.result(timeout=10) for future in futures] == ['first', 'second']
    assert len(recorder.calls) == 2
    assert {call.model for call in recorder.calls} == {'first', 'second'}


def test_native_tool_response_is_recorded_without_text(monkeypatch):
    result = {'message': {'content': '', 'tool_calls': [{'function': {'name': 'owned', 'arguments': {}}}]}}
    monkeypatch.setattr(OllamaBackend, 'chat', lambda *args, **kwargs: result)
    with TimingRecorder() as recorder:
        assert OllamaBackend('http://owned.invalid').chat('owned model', []) is result
    assert [(call.outcome, call.response_chars) for call in recorder.calls] == [('response', 0)]


def test_actual_engine_chat_has_a_main_turn_context(monkeypatch, mock_config):
    from jarvis.reply.engine import chat_with_messages

    mock_config.llm_provider = 'ollama'
    result = {'message': {'content': 'answer'}}
    monkeypatch.setattr(OllamaBackend, 'chat', lambda *args, **kwargs: result)
    with TimingRecorder() as recorder:
        assert chat_with_messages(mock_config, [{'role': 'user', 'content': 'question'}]) is result
    assert set(recorder.by_context()) == {'main_chat_turn'}


def test_nested_recorders_restore_outer_measurement(monkeypatch):
    respond = lambda *args, **kwargs: 'answer'
    monkeypatch.setattr(OllamaBackend, 'direct', respond)
    backend = OllamaBackend('http://owned.invalid')
    with TimingRecorder() as outer:
        with TimingRecorder() as inner:
            assert backend.direct('inner', 'system', 'user') == 'answer'
        assert backend.direct('outer', 'system', 'user') == 'answer'
    assert [call.model for call in outer.calls] == ['inner', 'outer']
    assert [call.model for call in inner.calls] == ['inner']
    assert OllamaBackend.direct is respond
