"""Search page responses release unread bodies and validate redirect hops."""
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
import requests

from jarvis.tools.builtin import web_search

pytestmark = pytest.mark.unit


@pytest.fixture
def search_page_server(monkeypatch):
    state = SimpleNamespace(status=302, destination='/target', paths=[], responses=[],
                            followed=threading.Event(), finished=threading.Event(),
                            observed=threading.Event(), handled_before_body=None,
                            large=False, error=False)
    text = b'<p>Kyoto station reports 19.5 degrees Celsius.</p>'

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state.paths.append(self.path)
            if self.path == '/redirect':
                self.send_response(state.status)
                self.send_header('Location', state.destination)
                self.send_header('Content-Length', str(2 * web_search._MAX_FETCH_BYTES))
                self.end_headers()
                self.wfile.flush()
                event = state.followed if state.destination == '/target' else state.finished
                state.handled_before_body = event.wait(timeout=2)
                state.observed.set()
                body = b' ' * (2 * web_search._MAX_FETCH_BYTES)
            elif self.path == '/loop':
                self.send_response(307)
                self.send_header('Location', '/loop')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            else:
                if self.path == '/target':
                    state.followed.set()
                body = text + b' ' * (2 * web_search._MAX_FETCH_BYTES) if state.large else text
                self.send_response(500 if state.error else 200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f'http://127.0.0.1:{server.server_port}'
    monkeypatch.setattr(web_search, '_is_public_url', lambda url: url.startswith(base + '/')
                        and not url.endswith('/private'))
    original_get = requests.get

    def retain_response(*args, **kwargs):
        response = original_get(*args, **kwargs)
        state.responses.append(response)
        return response

    monkeypatch.setattr(requests, 'get', retain_response)
    yield state, base
    for response in state.responses:
        response.close()
    server.shutdown()
    server.server_close()
    worker.join(timeout=2)


@pytest.mark.parametrize('status', [301, 302, 303, 307, 308])
@pytest.mark.parametrize('destination', ['/target', '/private'])
def test_redirect_headers_are_handled_before_the_body(search_page_server, status, destination):
    state, base = search_page_server
    state.status = status
    state.destination = destination
    result = web_search._fetch_page_content(base + '/redirect')
    state.finished.set()
    assert state.observed.wait(timeout=2)
    assert state.handled_before_body, '🌐 Redirect bodies must not delay following or rejecting a hop'
    if destination == '/target':
        assert result and '19.5' in result
    else:
        assert result is None and '/private' not in state.paths
    assert all(response.raw.closed for response in state.responses)


@pytest.mark.parametrize('large', [False, True])
@pytest.mark.parametrize('parser_error', [False, True])
def test_page_responses_close_after_extraction_or_parser_failure(monkeypatch, search_page_server, large, parser_error):
    state, base = search_page_server
    state.large = large
    if parser_error:
        import bs4

        def broken_parser(*args, **kwargs):
            raise ValueError('Private parser failure')

        monkeypatch.setattr(bs4, 'BeautifulSoup', broken_parser)
    result = web_search._fetch_page_content(base + '/page')
    assert result is None if parser_error else result and '19.5' in result
    assert state.responses[0].raw.closed, '🌐 Unread page bytes must be released before returning'


def test_http_failure_closes_the_unread_response(search_page_server):
    state, base = search_page_server
    state.error = True
    assert web_search._fetch_page_content(base + '/page') is None
    assert state.responses[0].raw.closed


def test_redirect_loops_stop_at_the_configured_hop_limit(search_page_server):
    state, base = search_page_server
    assert web_search._fetch_page_content(base + '/loop') is None
    assert len(state.paths) == web_search._MAX_REDIRECTS + 1
    assert all(response.raw.closed for response in state.responses)
