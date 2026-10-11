"""Concurrent voice downloads publish complete files and own their cleanup."""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
import src.jarvis.output.tts as tts

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('first_fails', [False, True])
def test_overlapping_voice_downloads_keep_partial_files_private(monkeypatch, tmp_path, first_fails):
    written = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    voices = [b'A' * 16384, b'B' * 8192]
    local = threading.local()

    class Response:
        def __init__(self, body, owner, voice):
            self.headers = {'content-length': str(len(body))}
            self.body, self.owner, self.voice = body, owner, voice
            self.closed = False

        def raise_for_status(self):
            pass

        def iter_content(self, **kwargs):
            if self.voice:
                boundary = len(self.body) // 2
                yield self.body[:boundary]
                written[self.owner].set()
                assert release[self.owner].wait(5), 'Download was not released'
                if self.owner == 0 and first_fails:
                    raise requests.ConnectionError('interrupted response')
                yield self.body[boundary:]
            else:
                yield self.body

        def close(self):
            self.closed = True

    responses = []
    def get(url, **kwargs):
        voice = not url.endswith('.json')
        reply = Response(voices[local.owner] if voice else b'{}', local.owner, voice)
        responses.append(reply)
        return reply

    monkeypatch.setattr(tts, '_get_piper_models_dir', lambda: tmp_path)
    monkeypatch.setattr(tts, '_PIPER_DOWNLOAD_MAX_RETRIES', 0)
    monkeypatch.setattr(requests, 'get', get)
    def download(owner):
        local.owner = owner
        return tts._download_piper_voice('en_GB-alan-medium')

    model = tmp_path / 'en_GB-alan-medium.onnx'
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(download, 0)
        try:
            assert written[0].wait(5), 'First download did not write a partial file'
            second = pool.submit(download, 1)
            assert written[1].wait(5), 'Second download did not write a partial file'
            release[0].set()
            assert first.result(timeout=5) == (None if first_fails else str(model))
            if not first_fails:
                complete = model.read_bytes() == voices[0]
                assert complete, 'A published model must contain its complete response body'
            release[1].set()
            assert second.result(timeout=5) == str(model)
        finally:
            for event in release:
                event.set()

    assert model.read_bytes() == voices[1]
    assert (tmp_path / 'en_GB-alan-medium.onnx.json').read_bytes() == b'{}'
    assert all(reply.closed for reply in responses)
    assert not list(tmp_path.glob('*.tmp'))
