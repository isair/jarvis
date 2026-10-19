"""Streaming budgets include transport, ignored events and token callbacks."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from threading import Event
from unittest.mock import MagicMock, patch

import pytest

from jarvis.llm import OllamaBackend, OpenAICompatibleBackend, streaming

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_network")]


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


@pytest.fixture
def clock(monkeypatch, transport):
    clock = [0.0]
    module_clock = MagicMock(wraps=time)
    module_clock.monotonic.side_effect = lambda: clock[0]
    backend, _, _ = transport
    monkeypatch.setattr(import_module(type(backend).__module__), 'time', module_clock)
    monkeypatch.setattr(streaming, 'time', module_clock)
    return clock


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
def test_expired_stream_is_not_a_successful_response(transport, clock, expiry):
    backend, content, ignored = transport
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

    with patch('jarvis.llm.http.post', side_effect=post):
        result = backend.streaming('model', 'system', 'user', on_token=receive, timeout_sec=timeout)

    assert result is None, 'Expired output must not become a completed model response'
    assert tokens == ([] if expiry == 'headers' else ['early'])
    assert response.closed, 'Aborted streams must release the connection'


@pytest.mark.parametrize("background_wait", [False, True])
def test_stream_within_budget_preserves_content_and_callback(transport, clock, background_wait):
    backend, content, ignored = transport
    response = TimedResponse(clock, [(1, ignored), (2, content('hello ')), (3, b''),
                                     (4, content('world'))], finished_at=4.5)
    tokens = []
    observed = []

    def background_clock():
        before = time.monotonic()
        Event().wait(0.001)
        return before, time.monotonic()

    def post(*args, **kwargs):
        if background_wait:
            with ThreadPoolExecutor(max_workers=1) as pool:
                observed.append(pool.submit(background_clock).result(timeout=5))
        return response

    with patch('jarvis.llm.http.post', side_effect=post):
        result = backend.streaming('model', 'system', 'user', on_token=tokens.append, timeout_sec=5)
    assert result == 'hello world'
    assert tokens == ['hello ', 'world']
    assert response.closed
    if background_wait:
        before, after = observed[0]
        assert after > before, 'The synthetic streaming clock froze an unrelated worker'
