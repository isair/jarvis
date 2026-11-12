"""Supplied pages retain local access and both page paths retain proxies."""

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from urllib3.util import connection

from jarvis.tools.builtin.fetch_web_page import FetchWebPageTool
from jarvis.tools.builtin.web_search import _fetch_page_content


pytestmark = pytest.mark.unit


@pytest.fixture
def owned_page_network(monkeypatch):
    paths, headers = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            paths.append(self.path)
            headers.append(dict(self.headers))
            body = b'<p>Owned dashboard content.</p>'
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    resolve = socket.getaddrinfo
    connect = connection.create_connection

    def owned_resolve(host, port, *args, **kwargs):
        assert host in {'127.0.0.1', 'dashboard.test', 'proxy.test', 'public.test'}
        address = '1.1.1.1' if host == 'public.test' else '127.0.0.1'
        return resolve(address, port, *args, **kwargs)

    def owned_connect(address, *args, **kwargs):
        assert address[0] in {'127.0.0.1', 'dashboard.test', 'proxy.test'}
        assert address[1] == server.server_port
        return connect(('127.0.0.1', server.server_port), *args, **kwargs)

    monkeypatch.setattr(socket, 'getaddrinfo', owned_resolve)
    monkeypatch.setattr(connection, 'create_connection', owned_connect)
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy',
                'all_proxy', 'NO_PROXY', 'no_proxy'):
        monkeypatch.delenv(key, raising=False)
    yield SimpleNamespace(port=server.server_port, paths=paths, headers=headers)
    server.shutdown()
    server.server_close()
    worker.join(timeout=2)
    assert not worker.is_alive()


@pytest.mark.parametrize('host', ['127.0.0.1', 'dashboard.test'])
def test_supplied_page_can_read_local_dashboard(owned_page_network, host):
    url = f'http://{host}:{owned_page_network.port}/dashboard'
    result = FetchWebPageTool().run({'url': url}, SimpleNamespace(user_print=lambda _: None))
    assert result.success, result.reply_text
    assert 'Owned dashboard content.' in result.reply_text
    assert owned_page_network.paths == ['/dashboard']


def test_supplied_local_page_honours_proxy_bypass(owned_page_network, monkeypatch):
    monkeypatch.setenv('HTTP_PROXY', 'http://unused-proxy.invalid:1')
    monkeypatch.setenv('NO_PROXY', 'dashboard.test')
    url = f'http://dashboard.test:{owned_page_network.port}/dashboard'
    result = FetchWebPageTool().run({'url': url}, SimpleNamespace(user_print=lambda _: None))
    assert result.success, result.reply_text
    assert 'Owned dashboard content.' in result.reply_text
    assert owned_page_network.paths == ['/dashboard']
    assert 'Proxy-Authorization' not in owned_page_network.headers[0]


@pytest.mark.parametrize('search_result', [False, True])
def test_page_fetch_uses_configured_proxy(owned_page_network, monkeypatch, tmp_path, search_result):
    monkeypatch.setenv('HTTP_PROXY', f'http://proxy-user:proxy-password@proxy.test:{owned_page_network.port}')
    netrc = tmp_path / 'netrc'
    netrc.write_text('machine public.test login origin-user password origin-password\n')
    monkeypatch.setenv('NETRC', str(netrc))
    url = 'http://public.test/article'
    if search_result:
        content = _fetch_page_content(url)
        assert content is not None
        assert 'Owned dashboard content.' in content
        assert owned_page_network.headers[0]['Authorization'].startswith('Basic ')
    else:
        result = FetchWebPageTool().run({'url': url}, SimpleNamespace(user_print=lambda _: None))
        assert result.success, result.reply_text
        assert 'Owned dashboard content.' in result.reply_text
    assert owned_page_network.paths == [url]
    assert owned_page_network.headers[0]['Proxy-Authorization'].startswith('Basic ')
