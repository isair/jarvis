"""Streaming budgets include transport, ignored events and token callbacks."""

import json
from unittest.mock import patch

import pytest

from jarvis.llm import OllamaBackend, OpenAICompatibleBackend

pytestmark = pytest.mark.unit


@pytest.fixture(params=[OllamaBackend, OpenAICompatibleBackend], ids=['ollama', 'openai'])
def transport(request):
    backend_type = request.param

    def content(text):
        if backend_type is OllamaBackend:
            return json.dumps({'message': {'content': text}}).encode()
        return b'data: ' + json.dumps({'choices': [{'delta': {'content': text}}]}).encode()

    ignored = (b'{"message":{"thinking":"still reasoning"}}'
               if backend_type is OllamaBackend else b': ping')
    return backend_type('http://localhost:1234'), content, ignored


class TimedResponse:
    def __init__(self, clock, events, finished_at=None):
        self.clock = clock
        self.events = events
        self.finished_at = finished_at
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def raise_for_status(self):
        pass

    def iter_lines(self):
        for elapsed, line in self.events:
            self.clock[0] = elapsed
            yield line
        if self.finished_at is not None:
            self.clock[0] = self.finished_at


@pytest.mark.parametrize('expiry', ['content', 'ignored', 'blank', 'completion', 'callback', 'headers'])
def test_expired_stream_is_not_a_successful_response(transport, monkeypatch, expiry):
    backend, content, ignored = transport
    clock = [0.0]
    monkeypatch.setattr('time.monotonic', lambda: clock[0])
    timeout = 5.0
    events = [(1.0, content('early'))]
    if expiry in ('content', 'ignored', 'blank'):
        events += [(timeout, {'content': content('late'), 'ignored': ignored, 'blank': b''}[expiry]),
                   (timeout + 1, content('later'))]
    response = TimedResponse(clock, events, timeout if expiry == 'completion' else None)
    tokens = []

    def receive(text):
        tokens.append(text)
        if expiry == 'callback':
            clock[0] = timeout

    def post(*args, **kwargs):
        if expiry == 'headers':
            clock[0] = timeout
            response.events = [(timeout, content('late'))]
        return response

    with patch('requests.post', side_effect=post):
        result = backend.streaming('model', 'system', 'user', on_token=receive, timeout_sec=timeout)

    assert result is None, 'Expired output must not become a completed model response'
    assert tokens == ([] if expiry == 'headers' else ['early'])
    assert response.closed, 'Aborted streams must release the connection'


def test_stream_within_budget_preserves_content_and_callback(transport, monkeypatch):
    backend, content, ignored = transport
    clock = [0.0]
    monkeypatch.setattr('time.monotonic', lambda: clock[0])
    response = TimedResponse(clock, [(1, ignored), (2, content('hello ')), (3, b''),
                                     (4, content('world'))], finished_at=4.5)
    tokens = []
    with patch('requests.post', return_value=response):
        result = backend.streaming('model', 'system', 'user', on_token=tokens.append, timeout_sec=5)
    assert result == 'hello world'
    assert tokens == ['hello ', 'world']
    assert response.closed
