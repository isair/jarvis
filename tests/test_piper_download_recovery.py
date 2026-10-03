"""Piper downloads recover transport failures without publishing partial files."""
from unittest.mock import Mock

import pytest
import requests

import src.jarvis.output.tts as tts

pytestmark = pytest.mark.unit


class Response:
    """Small streamed HTTP boundary with an observable closed-resource state."""
    def __init__(self, body):
        self.headers = {'content-length': str(len(body))}
        self.body = body
        self.closed = False

    def raise_for_status(self):
        pass

    def iter_content(self, **kwargs):
        yield self.body

    def close(self):
        self.closed = True


def response(body=b'complete'):
    return Response(body)


@pytest.fixture
def download(monkeypatch, tmp_path):
    monkeypatch.setattr(tts, '_get_piper_models_dir', lambda: tmp_path)
    monkeypatch.setattr(tts.time, 'sleep', lambda seconds: None)
    return lambda: tts._download_piper_voice('en_GB-alan-medium')


@pytest.mark.parametrize('failure', [requests.ConnectionError('reset'), requests.Timeout('timed out')])
def test_connection_failure_recovers_without_publishing_partial_model(monkeypatch, tmp_path, download, failure):
    replies = iter([failure, response(b'voice'), response(b'{}')])

    def get(*args, **kwargs):
        reply = next(replies)
        if isinstance(reply, Exception):
            assert not list(tmp_path.glob('*.onnx'))
            raise reply
        return reply

    monkeypatch.setattr(requests, 'get', get)
    assert download() == str(tmp_path / 'en_GB-alan-medium.onnx')
    assert (tmp_path / 'en_GB-alan-medium.onnx').read_bytes() == b'voice'
    assert (tmp_path / 'en_GB-alan-medium.onnx.json').read_bytes() == b'{}'


def test_mid_stream_reset_restarts_file_and_closes_failed_response(monkeypatch, tmp_path, download):
    failed = response()
    def interrupted(**kwargs):
        yield b'partial'
        raise requests.exceptions.ChunkedEncodingError('connection reset')
    failed.iter_content = interrupted
    replies = iter([failed, response(b'complete'), response(b'{}')])
    monkeypatch.setattr(requests, 'get', lambda *args, **kwargs: next(replies))
    assert download() == str(tmp_path / 'en_GB-alan-medium.onnx')
    assert (tmp_path / 'en_GB-alan-medium.onnx').read_bytes() == b'complete'
    assert failed.closed
    assert not list(tmp_path.glob('*.tmp'))


def test_persistent_transport_failure_is_bounded_and_cleans_partial_files(monkeypatch, tmp_path, download):
    replies = []
    def get(*args, **kwargs):
        reply = response()
        def interrupted(**kwargs):
            yield b'partial'
            raise requests.ConnectionError('reset')
        reply.iter_content = interrupted
        replies.append(reply)
        return reply
    monkeypatch.setattr(requests, 'get', get)
    assert download() is None
    assert len(replies) == tts._PIPER_DOWNLOAD_MAX_RETRIES + 1
    assert all(reply.closed for reply in replies)
    assert not list(tmp_path.iterdir())


def test_certificate_failure_is_not_retried(monkeypatch, download):
    get = Mock(side_effect=requests.exceptions.SSLError('untrusted certificate'))
    monkeypatch.setattr(requests, 'get', get)
    assert download() is None
    get.assert_called_once()


def test_completed_model_is_retained_when_configuration_download_fails(monkeypatch, tmp_path, download):
    def get(url, **kwargs):
        if url.endswith('.json'):
            raise requests.ConnectionError('reset')
        return response(b'complete voice')
    monkeypatch.setattr(requests, 'get', get)
    assert download() is None
    assert (tmp_path / 'en_GB-alan-medium.onnx').read_bytes() == b'complete voice'
    assert not (tmp_path / 'en_GB-alan-medium.onnx.json').exists()
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.integration
def test_real_http_disconnect_recovers_complete_voice_files(monkeypatch, tmp_path, download):
    """A truncated localhost response must not become the cached voice file."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    voice = b'synthetic local voice bytes' * 1000
    class Handler(BaseHTTPRequestHandler):
        interrupted = False
        def do_GET(self):
            body = b'{}' if self.path.endswith('.json') else voice
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            if not self.path.endswith('.json') and not type(self).interrupted:
                type(self).interrupted = True
                self.wfile.write(body[:8192])
                self.wfile.flush()
                self.close_connection = True
            else:
                self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    worker.start()
    monkeypatch.setattr(tts, 'PIPER_VOICE_BASE_URL', f'http://127.0.0.1:{server.server_port}')
    try:
        assert download() == str(tmp_path / 'en_GB-alan-medium.onnx')
        assert (tmp_path / 'en_GB-alan-medium.onnx').read_bytes() == voice
        assert (tmp_path / 'en_GB-alan-medium.onnx.json').read_bytes() == b'{}'
        assert not list(tmp_path.glob('*.tmp'))
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_locked_partial_file_reports_failure_without_crashing(monkeypatch, tmp_path, download):
    from pathlib import Path
    partial = tmp_path / 'en_GB-alan-medium.tmp'
    partial.write_bytes(b'partial')
    unlink = Path.unlink
    def locked(path, *args, **kwargs):
        if path == partial:
            raise PermissionError('file is locked')
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', locked)
    monkeypatch.setattr(requests, 'get', Mock(side_effect=requests.ConnectionError('reset')))
    assert download() is None
    assert not (tmp_path / 'en_GB-alan-medium.onnx').exists()
