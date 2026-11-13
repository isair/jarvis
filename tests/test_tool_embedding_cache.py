"""Owned HTTP controls for embedding routing across factory lookups."""
from contextlib import contextmanager
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from types import SimpleNamespace

import numpy as np
import pytest

from jarvis.llm import get_embedding_backend
from jarvis.tools.registry import ToolSpec
from jarvis.tools.selection import select_tools, ToolSelectionStrategy

pytestmark = pytest.mark.unit


def vector(text):
    return [value - 128 for value in sha256(text.encode()).digest()]


@contextmanager
def embedding_endpoint():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            text = data.get('prompt', data.get('input'))
            requests.append((self.path, data['model'], text, self.headers.get('Authorization')))
            body = json.dumps({'embedding': vector(text), 'data': [{'embedding': vector(text)}]}).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = Thread(target=server.serve_forever)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)
        assert not worker.is_alive()


def catalogue():
    return {**{f'tool{i}': ToolSpec(f'tool{i}', f'Capability {i}') for i in range(12)},
            'stop': ToolSpec('stop', 'Stop the conversation')}


def route(cfg, query, tools):
    return select_tools(query, tools, {}, strategy=ToolSelectionStrategy.EMBEDDING,
                        embedding_backend=get_embedding_backend(cfg), embed_model=cfg.embedding_model)


@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
def test_repeated_routes_only_send_new_queries(provider):
    with embedding_endpoint() as (url, requests):
        cfg = SimpleNamespace(embedding_provider=provider, embedding_base_url=url,
                              embedding_api_key='owned-token', embedding_model='owned-model')
        tools = catalogue()
        route(cfg, 'first request', tools)
        requests.clear()
        query = 'a different request'
        actual = route(cfg, query, tools)
        assert [request[2] for request in requests] == [query]
        # An independent uncached ranking uses the identical float32 arithmetic.
        def normalised(text):
            arr = np.array(vector(text), dtype=np.float32)
            return arr / np.linalg.norm(arr)
        q = normalised(query)
        scores = sorted(((name, float(np.dot(q, normalised(f'{name}: {tool.description}'))))
                         for name, tool in tools.items() if name != 'stop'),
                        key=lambda pair: pair[1], reverse=True)
        from jarvis.tools.selection import _MIN_SELECTED, _RELATIVE_THRESHOLD
        expected = [name for name, score in scores if score >= scores[0][1] * _RELATIVE_THRESHOLD]
        if len(expected) < _MIN_SELECTED:
            expected = [name for name, score in scores[:_MIN_SELECTED]]
        assert actual == expected + ['stop']


@pytest.mark.parametrize('change', ['endpoint', 'credential', 'provider', 'model'])
def test_effective_configuration_changes_rebuild_catalogue(change):
    with embedding_endpoint() as (url, requests), embedding_endpoint() as (other_url, other_requests):
        cfg = SimpleNamespace(embedding_provider='openai_compatible', embedding_base_url=url,
                              embedding_api_key='first-token', embedding_model='first-model')
        tools = catalogue()
        route(cfg, 'first query', tools)
        requests.clear()
        if change == 'endpoint':
            cfg.embedding_base_url = other_url
        elif change == 'credential':
            cfg.embedding_api_key = 'second-token'
        elif change == 'provider':
            cfg.embedding_provider = 'ollama'
        else:
            cfg.embedding_model = 'second-model'
        result = route(cfg, 'second query', tools)
        observed = other_requests if change == 'endpoint' else requests
        assert len(observed) == len(tools)
        assert observed[0][2] == 'second query'
        assert {entry[1] for entry in observed} == {cfg.embedding_model}
        if cfg.embedding_provider == 'openai_compatible':
            assert {entry[3] for entry in observed} == {f'Bearer {cfg.embedding_api_key}'}
        assert 'stop' in result
