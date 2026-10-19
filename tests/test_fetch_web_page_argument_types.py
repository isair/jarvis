"""Malformed JSON fields do not become web requests or truthy link flags."""
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from jarvis.tools.base import ToolContext
from jarvis.tools.builtin.fetch_web_page import FetchWebPageTool

pytestmark = pytest.mark.unit


@pytest.fixture
def page_argument_server(monkeypatch, mock_config):
    paths = []
    body = b'<p>Private page facts.</p><a href="/details">Extra page details</a>'

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            paths.append(self.path)
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f'http://127.0.0.1:{server.server_port}/page'
    original_get = requests.get

    def private_get(requested_url, *args, **kwargs):
        if requested_url != url:
            raise requests.RequestException('Private fixture blocks non-local requests')
        return original_get(requested_url, *args, **kwargs)

    monkeypatch.setattr(requests, 'get', private_get)
    context = ToolContext(None, mock_config, '', '', '', 0, lambda message: None)
    yield FetchWebPageTool(), context, url, paths
    server.shutdown()
    server.server_close()
    worker.join(timeout=2)


@pytest.mark.parametrize('value', [None, False, True, 0, 12, 0.5, [], {}, ['https://fixture.example']])
def test_non_string_url_is_a_correctable_argument_failure(page_argument_server, value):
    tool, context, url, paths = page_argument_server
    result = tool.run({'url': value}, context)
    assert not result.success and 'string' in result.reply_text.casefold()
    assert 'url' in result.reply_text.casefold()
    assert not paths, '🌐 Malformed URLs must not issue HTTP requests'


@pytest.mark.parametrize('value', [None, 0, 1, 0.5, 'false', 'true', [], {}])
def test_non_boolean_link_flag_is_rejected_before_fetching(page_argument_server, value):
    tool, context, url, paths = page_argument_server
    result = tool.run({'url': url, 'include_links': value}, context)
    assert not result.success and 'boolean' in result.reply_text.casefold()
    assert 'include_links' in result.reply_text
    assert not paths, '🌐 Malformed link flags must not issue HTTP requests'


@pytest.mark.parametrize('flag', [None, False, True])
def test_declared_arguments_retain_page_and_link_selection(page_argument_server, flag):
    tool, context, url, paths = page_argument_server
    args = {'url': url}
    if flag is not None:
        args['include_links'] = flag
    result = tool.run(args, context)
    assert result.success and 'Private page facts.' in result.reply_text
    assert ('**Links found on page:**' in result.reply_text) == (flag is True)
    assert paths == ['/page']
