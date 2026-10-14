"""Real native search downloads obey budgets and verify TLS, source and frozen."""
import gzip
import json
import os
from pathlib import Path
import ssl
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(params=['source', 'frozen'])
def download(request):
    executable = os.environ.get('JARVIS_SEARCH_PROBE_EXE')
    if request.param == 'frozen' and not executable:
        pytest.skip('Set JARVIS_SEARCH_PROBE_EXE to qualify the frozen native transport')

    def run(url, timeout=2, limit=1024):
        if request.param == 'source':
            from jarvis.tools.search_http import download_search_html
            try:
                status, body = download_search_html(url, timeout, limit)
                return {'status': status, 'body': body.decode()}
            except Exception as error:
                return {'error': type(error).__name__, 'message': str(error)}
        result = subprocess.run([executable, url, str(timeout), str(limit)],
                                capture_output=True, text=True, timeout=10)
        assert result.returncode in (0, 1), result.stderr
        return json.loads(result.stdout)
    return run


@pytest.fixture
def servers(tmp_path, monkeypatch):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    # Native libcurl bypasses Python socket patches. These tests only pass
    # freshly owned loopback endpoints and remove inherited proxy/CA settings.
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy',
                'all_proxy', 'CURL_CA_BUNDLE', 'REQUESTS_CA_BUNDLE'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('NO_PROXY', '127.0.0.1,localhost')
    monkeypatch.setenv('no_proxy', '127.0.0.1,localhost')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = b'Owned fixture'
            self.send_response(302 if self.path == '/redirect' else 200)
            if self.path == '/redirect':
                self.send_header('Location', '/target')
            if self.path == '/gzip':
                body = gzip.compress(b'x' * (1024 * 1024))
                self.send_header('Content-Encoding', 'gzip')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            if self.path == '/slow':
                time.sleep(.5)
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                pass

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'Owned test CA')])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ip_address('127.0.0.1'))]), False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), True)
            .sign(key, hashes.SHA256()))
    cert_path = tmp_path / 'owned-ca.pem'
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path = tmp_path / 'owned-key.pem'
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                         serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    plain = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    secure = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_path, key_path)
    secure.socket = ctx.wrap_socket(secure.socket, server_side=True)
    threads = [threading.Thread(target=s.serve_forever, kwargs={'poll_interval': .02}, daemon=True)
               for s in (plain, secure)]
    for thread in threads:
        thread.start()
    try:
        yield (f'http://127.0.0.1:{plain.server_port}',
               f'https://127.0.0.1:{secure.server_port}', cert_path)
    finally:
        for server in (plain, secure):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)
            assert not thread.is_alive()


def test_native_plain_and_exact_byte_budget(download, servers):
    assert download(servers[0], limit=len(b'Owned fixture')) == {'status': 200, 'body': 'Owned fixture'}


def test_native_redirect_is_not_followed(download, servers):
    assert download(servers[0] + '/redirect') == {'status': 302, 'body': 'Owned fixture'}


@pytest.mark.parametrize('path', ['/', '/gzip'])
def test_native_cap_applies_to_decoded_bytes(download, servers, path):
    result = download(servers[0] + path, limit=8)
    assert result['error'] == 'ValueError'
    assert 'decoded byte budget' in result['message']
    assert 'body' not in result


def test_native_timeout_does_not_return_partial_success(download, servers):
    result = download(servers[0] + '/slow', timeout=.05)
    assert result['error'] == 'Timeout'
    assert 'body' not in result


def test_native_tls_rejects_untrusted_certificate(download, servers):
    result = download(servers[1])
    assert result['error'] == 'CertificateVerifyError'
    assert 'body' not in result


def test_native_tls_accepts_explicit_owned_ca(download, servers, monkeypatch):
    monkeypatch.setenv('CURL_CA_BUNDLE', str(servers[2]))
    assert download(servers[1]) == {'status': 200, 'body': 'Owned fixture'}


@pytest.mark.parametrize('timeout,limit', [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_invalid_budget_is_rejected(download, servers, timeout, limit):
    result = download(servers[0], timeout=timeout, limit=limit)
    assert result['error'] == 'ValueError'
    assert 'positive' in result['message']
