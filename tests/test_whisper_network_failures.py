"""Network recovery retains trust checks, cached files and private error data."""
import errno
import ssl
from types import SimpleNamespace

import pytest
import requests
from urllib3.exceptions import MaxRetryError, ProtocolError, SSLError

from jarvis.listening import model_download

pytestmark = pytest.mark.unit


def nested_certificate_failure():
    certificate = ssl.SSLCertVerificationError(1, 'ιδιωτικό μήνυμα /private/fixture-cache')
    retry = MaxRetryError(None, '/private/fixture-url', SSLError(certificate))
    return requests.exceptions.SSLError(retry)


@pytest.mark.parametrize('error,category', [
    (nested_certificate_failure(), 'certificate'),
    (ssl.SSLCertVerificationError(1, 'μη έμπιστο'), 'certificate'),
    (requests.exceptions.ReadTimeout('ιδιωτικό'), 'network'),
    (requests.exceptions.ConnectTimeout('ιδιωτικό'), 'network'),
    (requests.exceptions.ConnectionError(ProtocolError('ιδιωτικό', ConnectionResetError())), 'network'),
    (ssl.SSLError(1, 'handshake interrupted'), 'network'),
    (TimeoutError(), 'network'),
    (OSError(errno.ENETUNREACH, 'ιδιωτικό'), 'network'),
    (requests.exceptions.HTTPError('404'), 'download'),
    (RuntimeError('CERTIFICATE_VERIFY_FAILED: connection timed out'), 'download'),
])
def test_worker_preserves_network_failure_category_without_private_details(monkeypatch, error, category):
    wrapped = FileNotFoundError('cache unavailable')
    wrapped.__cause__ = error
    def fail_download(*args, **kwargs):
        raise wrapped
    monkeypatch.setattr('faster_whisper.utils.download_model', fail_download)
    messages = []
    connection = SimpleNamespace(send=messages.append, close=lambda: None)
    model_download._download_worker(connection, 'small')
    assert messages == [('error', category, 'FileNotFoundError')]


def test_network_wrapper_keeps_rate_limit_priority_and_terminates_with_cycles():
    limited = RuntimeError('remote quota')
    limited.response = SimpleNamespace(status_code=429)
    error = requests.exceptions.ConnectionError(limited)
    limited.__context__ = error
    assert model_download._download_error_category(error) == 'rate_limit'


@pytest.mark.parametrize('category,actions', [
    ('certificate', ('certificate', 'proxy', 'trust', 'restart jarvis')),
    ('network', ('connection', 'proxy', 'restart jarvis', 'resume')),
])
def test_network_failure_explains_safe_recovery_and_retains_cache(monkeypatch, tmp_path, capsys, category, actions):
    cached = tmp_path / 'model.bin'
    cached.write_bytes(b'partial weights')
    monkeypatch.setattr('faster_whisper.utils.download_model', lambda *args, **kwargs: str(tmp_path))
    def fail_worker(*args):
        raise model_download.ModelDownloadError(category, 'FileNotFoundError')
    monkeypatch.setattr(model_download, '_run_download_worker', fail_worker)
    with pytest.raises(model_download.ModelDownloadError) as failure:
        model_download.prepare_faster_whisper_model('small')
    assert failure.value.category == category
    assert cached.read_bytes() == b'partial weights'
    output = capsys.readouterr().out.lower()
    assert all(action in output for action in actions)
    assert 'cached files' in output and 'kept' in output
    assert 'disable' not in output
