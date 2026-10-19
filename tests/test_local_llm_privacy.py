"""Local model traffic uses the chosen endpoint and explicit credentials."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from jarvis.llm import OllamaBackend, OpenAICompatibleBackend

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("owned_loopback_network")]


@pytest.fixture
def ambient_environment(tmp_path, monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                 "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "netrc"
    path.write_text("", encoding="utf-8")
    path.chmod(0o600)
    monkeypatch.setenv("NETRC", str(path))
    return path


@contextmanager
def server(*, redirect=False):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            calls.append((self.path, dict(self.headers), body))
            if redirect and self.path != '/final':
                self.send_response(307)
                self.send_header('Location', '/final')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            data = json.dumps({"message": {"content": "Synthetic reply"},
                               "choices": [{"message": {"content": "Synthetic reply"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", calls
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join(3)
        assert not worker.is_alive()


def backend(provider, base):
    return OllamaBackend(base) if provider == "ollama" else OpenAICompatibleBackend(base, api_key="Explicit-key")


@pytest.mark.parametrize("provider", ["ollama", "openai"])
@pytest.mark.parametrize("host", ["127.0.0.1", "127.1", "2130706433", "0x7f000001"])
def test_local_private_text_bypasses_ambient_proxy(provider, ambient_environment, monkeypatch, host):
    with server() as (model_url, model_calls), server() as (proxy_url, proxy_calls):
        monkeypatch.setenv("HTTP_PROXY", proxy_url)
        client = backend(provider, model_url.replace("127.0.0.1", host))
        assert client.direct("synthetic-model", "Synthetic private context", "Synthetic private input", timeout_sec=2) == "Synthetic reply"
        assert proxy_calls == []
        assert len(model_calls) == 1
        assert "Synthetic private input" in json.dumps(model_calls[0][2])


@pytest.mark.parametrize("provider", ["ollama", "openai"])
def test_local_requests_do_not_borrow_netrc_credentials(provider, ambient_environment):
    ambient_environment.write_text("machine 127.0.0.1 login Synthetic-user password Synthetic-password\n", encoding="utf-8")
    with server() as (base, calls):
        client = backend(provider, base)
        assert client.direct("synthetic-model", "Synthetic context", "Synthetic input", timeout_sec=2) == "Synthetic reply"
        expected = None if provider == "ollama" else "Bearer Explicit-key"
        assert calls[0][1].get("Authorization") == expected


@pytest.mark.parametrize("provider", ["ollama", "openai"])
def test_local_redirect_retains_only_explicit_authentication(provider, ambient_environment):
    ambient_environment.write_text("machine 127.0.0.1 login Synthetic-user password Synthetic-password\n", encoding="utf-8")
    with server(redirect=True) as (base, calls):
        assert backend(provider, base).direct("synthetic-model", "Synthetic context", "Synthetic input", timeout_sec=2) == "Synthetic reply"
        expected = None if provider == "ollama" else "Bearer Explicit-key"
        assert len(calls) == 2
        assert all(headers.get("Authorization") == expected for _, headers, _ in calls)
        assert calls[-1][0] == '/final'


def test_explicit_url_credentials_remain_available(ambient_environment):
    import base64

    ambient_environment.write_text("machine 127.0.0.1 login Ambient-user password Ambient-password\n", encoding="utf-8")
    with server() as (base, calls):
        selected = base.replace('http://', 'http://Selected-user:Selected-password@')
        assert backend('ollama', selected).direct("synthetic-model", "Synthetic context", "Synthetic input", timeout_sec=2) == "Synthetic reply"
        expected = base64.b64encode(b'Selected-user:Selected-password').decode()
        assert calls[0][1]['Authorization'] == 'Basic ' + expected
