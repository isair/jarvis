"""Web pages cannot turn public requests into reads from private services."""

import ipaddress
import socket
import ssl
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from urllib3.util import connection

from jarvis.tools.builtin.fetch_web_page import FetchWebPageTool
from jarvis.tools import public_http


pytestmark = pytest.mark.unit


@pytest.fixture
def page_network(monkeypatch, tmp_path, request):
    state = SimpleNamespace(paths=[], headers=[], dialled=[], redirect=None, mixed=False, answers=None)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state.paths.append(self.path)
            state.headers.append(dict(self.headers))
            if self.path == '/redirect':
                self.send_response(302)
                self.send_header('Location', state.redirect)
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            body = b'<p>Owned public article.</p>' if self.path == '/article' else b'<p>Private fixture secret.</p>'
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    certificate_name = getattr(request, 'param', None)
    if certificate_name:
        state.certificate = tmp_path / 'certificate.pem'
        key = tmp_path / 'key.pem'
        subprocess.run([
            'openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
            '-keyout', str(key), '-out', str(state.certificate), '-days', '1',
            '-subj', f'/CN={certificate_name}',
            '-addext', f'subjectAltName=DNS:{certificate_name}',
        ], check=True, capture_output=True, timeout=15)
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(state.certificate, key)
        server.socket = tls.wrap_socket(server.socket, server_side=True)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    original_resolve = socket.getaddrinfo
    original_connect = connection.create_connection

    def resolve(host, port, *args, **kwargs):
        if host == 'public.test':
            addresses = state.answers or (['1.1.1.1', '127.0.0.1'] if state.mixed else ['1.1.1.1'])
            return [(socket.AF_INET6 if ':' in address else socket.AF_INET, socket.SOCK_STREAM,
                     socket.IPPROTO_TCP, '', (address, port or 80, 0, 0) if ':' in address else (address, port or 80))
                    for address in addresses]
        # All other names in this synthetic network resolve privately.
        return original_resolve('127.0.0.1', port, *args, **kwargs)

    def dial(address, *args, **kwargs):
        state.dialled.append(address[0])
        return original_connect(('127.0.0.1', server.server_port), *args, **kwargs)

    monkeypatch.setattr(socket, 'getaddrinfo', resolve)
    monkeypatch.setattr(connection, 'create_connection', dial)
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        monkeypatch.delenv(key, raising=False)
    state.context = SimpleNamespace(user_print=lambda message: None)
    yield state
    server.shutdown()
    server.server_close()
    worker.join(timeout=2)
    assert not worker.is_alive()


@pytest.mark.parametrize('url', [
    'http://127.0.0.1/private', 'http://169.254.169.254/private',
    'http://10.0.0.1/private', 'http://[::1]/private', 'http://private.test/private',
    'http://100.64.0.1/private',
    'http://[::ffff:127.0.0.1]/private', 'http://2130706433/private',
    'http://[fec0::1]/private', 'http://[::127.0.0.1]/private',
])
def test_private_destinations_return_no_content(page_network, url):
    result = FetchWebPageTool().run({'url': url}, page_network.context)
    assert not result.success
    assert 'Private fixture secret' not in result.reply_text
    assert not page_network.paths, 'Private destination must not receive an HTTP request'


@pytest.mark.parametrize('destination', ['http://127.0.0.1/private', 'http://private.test/private', 'http://[::1]/private'])
def test_public_redirect_cannot_reach_a_private_destination(page_network, destination):
    page_network.redirect = destination
    result = FetchWebPageTool().run({'url': 'http://public.test/redirect'}, page_network.context)
    assert not result.success
    assert page_network.paths == ['/redirect']
    assert 'Private fixture secret' not in result.reply_text


def test_mixed_public_and_private_dns_is_rejected(page_network):
    page_network.mixed = True
    result = FetchWebPageTool().run({'url': 'http://public.test/article'}, page_network.context)
    assert not result.success
    assert not page_network.paths


@pytest.mark.parametrize('address', ['fec0::1', '::127.0.0.1'])
def test_non_public_ipv6_dns_answers_are_rejected(page_network, address):
    page_network.answers = ['1.1.1.1', address]
    result = FetchWebPageTool().run({'url': 'http://public.test/article'}, page_network.context)
    assert not result.success
    assert not page_network.paths


@pytest.mark.parametrize('redirect', [False, True])
def test_public_pages_use_validated_numeric_destinations(page_network, redirect):
    page_network.redirect = '/article'
    url = 'http://public.test/' + ('redirect' if redirect else 'article')
    result = FetchWebPageTool().run({'url': url}, page_network.context)
    assert result.success, result.reply_text
    assert 'Owned public article.' in result.reply_text
    assert page_network.dialled
    assert all(ipaddress.ip_address(host).is_global for host in page_network.dialled)


def test_public_page_does_not_receive_ambient_credentials_or_proxy_requests(page_network, monkeypatch, tmp_path):
    netrc = tmp_path / 'netrc'
    netrc.write_text('machine public.test login owned-login password owned-password\n')
    monkeypatch.setenv('NETRC', str(netrc))
    monkeypatch.setenv('HTTP_PROXY', 'http://private.test:8080')
    monkeypatch.delenv('NO_PROXY', raising=False)
    result = FetchWebPageTool().run({'url': 'http://public.test/article'}, page_network.context)
    assert result.success, result.reply_text
    assert 'Authorization' not in page_network.headers[0]
    assert page_network.paths == ['/article']
    assert all(ipaddress.ip_address(host).is_global for host in page_network.dialled)


@pytest.mark.parametrize('page_network', ['public.test'], indirect=True)
def test_https_retains_original_hostname_for_certificate_verification(page_network):
    with public_http.get('https://public.test/article', verify=str(page_network.certificate), timeout=3) as response:
        assert 'Owned public article.' in response.text
    assert all(ipaddress.ip_address(host).is_global for host in page_network.dialled)


@pytest.mark.parametrize('page_network', ['different.test'], indirect=True)
def test_https_rejects_a_certificate_for_another_hostname(page_network):
    import requests
    with pytest.raises(requests.exceptions.SSLError):
        public_http.get('https://public.test/article', verify=str(page_network.certificate), timeout=3)
    assert not page_network.paths
