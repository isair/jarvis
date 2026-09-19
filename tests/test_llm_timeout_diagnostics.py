"""Timeout classification and honest startup diagnostics."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests
from urllib3.exceptions import ReadTimeoutError

from jarvis.llm import OpenAICompatibleBackend, OllamaBackend

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('backend_type', [OpenAICompatibleBackend, OllamaBackend])
@pytest.mark.parametrize('wrapped', [False, True])
def test_read_timeout_is_not_reported_as_connection_failure(backend_type, wrapped, capsys):
    error = requests.ReadTimeout('secret-host?token=secret')
    if wrapped:
        error = requests.ConnectionError(ReadTimeoutError(None, '/?token=secret', 'read timed out'))
    with patch('requests.post', side_effect=error):
        result = backend_type('http://unused').chat('test', [], timeout_sec=6)
    assert result is None
    output = capsys.readouterr().out
    assert 'timed out' in output
    assert '6s' in output
    assert 'connection error' not in output
    assert 'secret' not in output


@pytest.mark.parametrize('backend_type', [OpenAICompatibleBackend, OllamaBackend])
def test_real_connection_failure_still_reaches_caller(backend_type, capsys):
    with patch('requests.post', side_effect=requests.ConnectionError('token=secret')):
        with pytest.raises(requests.ConnectionError):
            backend_type('http://unused').chat('test', [])
    assert 'connection error' in capsys.readouterr().out


def test_warmup_reports_one_shared_probe_and_separate_intent_deadline(capsys):
    from jarvis.listening.listener import VoiceListener
    listener = VoiceListener.__new__(VoiceListener)
    listener.cfg = SimpleNamespace(intent_judge_timeout_sec=7.5)
    listener._llm_warmup_results = {
        'chat': ('shared-model', True), 'judge': ('shared-model', True),
        'router': ('shared-model', True), 'embed': ('missing-embed', False),
    }
    listener._report_llm_warmup()
    output = capsys.readouterr().out
    assert output.count('shared-model') == 1
    assert 'warmup probe passed' in output
    assert 'chat, intent judge, tool router' in output
    assert '7.5s' in output
    assert 'full intent request not tested' in output
    assert 'Embedding probe failed' in output
    assert 'ready' not in output
    assert 'will load' not in output


def test_failed_warmup_does_not_claim_ready(capsys):
    from jarvis.listening.listener import VoiceListener
    listener = VoiceListener.__new__(VoiceListener)
    listener.cfg = SimpleNamespace(intent_judge_timeout_sec=6)
    listener._llm_warmup_results = {'chat': ('offline-model', False)}
    listener._report_llm_warmup()
    output = capsys.readouterr().out
    assert 'warmup probe failed' in output
    assert 'passed' not in output


def test_connection_refused_is_not_a_timeout():
    from jarvis.llm.errors import is_timeout_error
    from urllib3.exceptions import NewConnectionError
    assert not is_timeout_error(requests.ConnectionError(NewConnectionError(None, 'refused')))


def test_chat_probe_cannot_prove_embedding_support():
    from jarvis.listening.listener import VoiceListener
    listener = VoiceListener.__new__(VoiceListener)
    listener.cfg = SimpleNamespace(llm_chat_model='same-name', embedding_model='same-name',
                                   fast_model='', tool_selection_strategy='heuristic', low_power_mode=False)
    listener._intent_judge = None
    with patch('jarvis.listening.listener.warm_up_chat_model', return_value=True), patch(
        'jarvis.listening.listener.get_embedding_backend',
        return_value=SimpleNamespace(embed=lambda *args, **kwargs: None),
    ):
        for thread in listener._start_llm_warmup():
            thread.join(timeout=2)
    assert listener._llm_warmup_results['chat'] == ('same-name', True)
    assert listener._llm_warmup_results['embed'] == ('same-name', False)


def test_wrapped_timeout_keeps_intent_judge_available():
    from jarvis.listening.intent_judge import IntentJudge
    from jarvis.listening.transcript_buffer import TranscriptSegment
    judge = IntentJudge()
    error = requests.ConnectionError(ReadTimeoutError(None, '/', 'slow body'))
    backend = OpenAICompatibleBackend('http://unused')
    with patch('jarvis.listening.intent_judge.get_llm_backend', return_value=backend), patch(
        'requests.post', side_effect=error,
    ):
        assert judge.judge([TranscriptSegment('Jarvis, hello', 1000.0, 1001.0)]) is None
    assert judge.available is True


@pytest.mark.parametrize('backend_type', [OpenAICompatibleBackend, OllamaBackend])
def test_actual_response_body_timeout_is_classified(backend_type, capsys):
    """Headers arrive, then the body stalls: Requests wraps this read timeout."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    finished = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.send_response(200)
            self.send_header('Content-Length', '100')
            self.end_headers()
            self.wfile.flush()
            finished.wait(timeout=2)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        backend = backend_type(f'http://127.0.0.1:{server.server_port}')
        assert backend.chat('test', [], timeout_sec=0.1) is None
        output = capsys.readouterr().out
        assert 'timed out' in output
        assert 'connection error' not in output
    finally:
        finished.set()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
