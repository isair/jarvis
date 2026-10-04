"""Web pages are bounded before HTML parsing or raw-text extraction."""
import gzip
import builtins
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
import pytest
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin import fetch_web_page
pytestmark = pytest.mark.unit

@pytest.fixture
def bounded_page_server(monkeypatch, mock_config):
    limit = 4096
    monkeypatch.setattr(fetch_web_page, '_MAX_FETCH_BYTES', limit, raising=False)
    html = b'<html><title>Private fixture</title><p>Bounded page content</p></html>'
    state = SimpleNamespace(body=html, encoding=None, length=True, charset='utf-8',
                            followed=threading.Event(), observed=threading.Event(),
                            followed_before_body=None, cookie=False)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/redirect':
                redirect_body = b' ' * (limit * 4)
                self.send_response(302)
                self.send_header('Location', '/target')
                self.send_header('Content-Length', str(len(redirect_body)))
                if state.cookie:
                    self.send_header('Set-Cookie', 'fixture=passed; Path=/')
                self.end_headers()
                self.wfile.flush()
                state.followed_before_body = state.followed.wait(timeout=2)
                state.observed.set()
                try:
                    self.wfile.write(redirect_body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if self.path == '/target':
                state.followed.set()
                if state.cookie and 'fixture=passed' not in self.headers.get('Cookie', ''):
                    self.send_error(403)
                    return
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=' + state.charset
                             if state.charset else 'application/octet-stream')
            if state.encoding: self.send_header('Content-Encoding', state.encoding)
            if state.length: self.send_header('Content-Length', str(len(state.body)))
            self.end_headers()
            try: self.wfile.write(state.body)
            except (BrokenPipeError, ConnectionResetError): pass
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    context = ToolContext(None, mock_config, '', '', '', 0, lambda message: None)
    yield fetch_web_page.FetchWebPageTool(), context, state, f'http://127.0.0.1:{server.server_port}', limit, html
    server.shutdown(); server.server_close(); worker.join(timeout=2)

@pytest.mark.parametrize('transport', ['length', 'unknown_length', 'gzip'])
def test_oversized_pages_return_failure_without_partial_content(bounded_page_server, transport):
    tool, context, state, url, limit, html = bounded_page_server
    body = html + b' ' * (limit + 1 - len(html))
    if transport == 'gzip':
        state.body = gzip.compress(body); state.encoding = 'gzip'
        assert len(state.body) < limit
    else:
        state.body = body; state.length = transport != 'unknown_length'
    result = tool.run({'url': url}, context)
    assert not result.success, '🌐 Oversized decoded pages need an honest fetch failure'
    assert 'Bounded page content' not in result.reply_text
    assert 'limit' in result.reply_text.casefold() or 'large' in result.reply_text.casefold()

@pytest.mark.parametrize('padding', [False, True])
def test_small_and_exact_limit_pages_keep_content(bounded_page_server, padding):
    tool, context, state, url, limit, html = bounded_page_server
    state.body = html + b' ' * (limit - len(html)) if padding else html
    result = tool.run({'url': url}, context)
    assert result.success, result.reply_text
    assert 'Bounded page content' in result.reply_text


@pytest.mark.parametrize('parser_available', [False, True])
@pytest.mark.parametrize('charset, text', [('utf-8', 'İstanbul ve İzmir'), ('iso-8859-1', 'Café déjà vu')])
def test_bounded_pages_preserve_decoded_text(monkeypatch, bounded_page_server, parser_available, charset, text):
    tool, context, state, url, limit, html = bounded_page_server
    state.charset = charset
    state.body = f'<html><meta charset="{charset}"><p>{text}</p></html>'.encode(charset)
    if not parser_available:
        original_import = builtins.__import__
        def import_without_parser(name, *args, **kwargs):
            if name == 'bs4':
                raise ImportError('Private missing-parser fixture')
            return original_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, '__import__', import_without_parser)
    result = tool.run({'url': url}, context)
    assert result.success, result.reply_text
    assert text in result.reply_text


@pytest.mark.parametrize('cookie', [False, True])
def test_redirects_follow_headers_without_downloading_the_redirect_body(bounded_page_server, cookie):
    tool, context, state, url, limit, html = bounded_page_server
    state.cookie = cookie
    result = tool.run({'url': url + '/redirect'}, context)
    assert result.success, result.reply_text
    assert state.observed.wait(timeout=1), '🌐 Redirect fixture must finish observing the request order'
    assert state.followed_before_body, '🌐 Redirect bodies must not be downloaded before following their Location'
    assert 'Bounded page content' in result.reply_text


@pytest.mark.parametrize('charset', [None, 'unsupported-fixture-encoding'])
def test_raw_text_keeps_utf8_with_missing_or_unsupported_charset(monkeypatch, bounded_page_server, charset):
    tool, context, state, url, limit, html = bounded_page_server
    text = 'İstanbul ve İzmir, kıyı şehirleri hakkında bir metin.'
    state.charset = charset
    state.body = text.encode('utf-8')
    original_import = builtins.__import__
    def import_without_parser(name, *args, **kwargs):
        if name == 'bs4':
            raise ImportError('Private missing-parser fixture')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', import_without_parser)
    result = tool.run({'url': url}, context)
    assert result.success, result.reply_text
    assert text in result.reply_text
