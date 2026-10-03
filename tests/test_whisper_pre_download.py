"""Downloads cannot take down the listener or admit incomplete model files."""
import os
import time
from pathlib import Path

import pytest

from jarvis.listening import model_download

pytestmark = pytest.mark.unit


def complete_model(path):
    path.mkdir(parents=True, exist_ok=True)
    for name in ('model.bin', 'config.json', 'tokenizer.json', 'vocabulary.json'):
        (path / name).write_bytes(b'{}')
    return path


def successful_worker(connection, model_name):
    connection.send(('ready', os.environ.get('JARVIS_TEST_WHISPER_PATH', model_name), ''))
    connection.close()


def failed_worker(connection, model_name):
    connection.send(('error', 'rate_limit', 'HTTP 429'))
    connection.close()


def aborted_worker(connection, model_name):
    # A non-zero native exit has the same IPC outcome as a native abort.
    os._exit(6)


def stalled_worker(connection, model_name):
    time.sleep(30)


@pytest.mark.parametrize('missing', ['model.bin', 'config.json', 'tokenizer.json', 'vocabulary.json'])
def test_partial_cache_is_completed_by_worker_before_loading(tmp_path, monkeypatch, missing):
    partial = complete_model(tmp_path / 'partial')
    (partial / missing).unlink()
    ready = complete_model(tmp_path / 'ready')
    import faster_whisper.utils
    monkeypatch.setattr(faster_whisper.utils, 'download_model', lambda *a, **k: str(partial))
    monkeypatch.setattr(model_download, '_download_worker', successful_worker, raising=False)
    monkeypatch.setenv('JARVIS_TEST_WHISPER_PATH', str(ready))
    assert model_download.prepare_faster_whisper_model('small') == str(ready)


def test_complete_cache_loads_offline_without_starting_worker(tmp_path, monkeypatch):
    ready = complete_model(tmp_path / 'ready')
    import faster_whisper.utils
    def cached(name, **kwargs):
        assert kwargs['local_files_only']
        return str(ready)
    monkeypatch.setattr(faster_whisper.utils, 'download_model', cached)
    def unexpected_worker(*args):
        pytest.fail('Complete cached models must stay offline')
    monkeypatch.setattr(model_download, '_run_download_worker', unexpected_worker, raising=False)
    assert model_download.prepare_faster_whisper_model('small') == str(ready)


@pytest.mark.parametrize('worker,category', [(aborted_worker, 'worker_exit'),
                                            (stalled_worker, 'timeout'),
                                            (failed_worker, 'rate_limit')])
def test_child_failures_are_bounded_and_report_their_category(monkeypatch, worker, category):
    monkeypatch.setattr(model_download, '_download_worker', worker, raising=False)
    start = time.monotonic()
    with pytest.raises(model_download.ModelDownloadError) as failure:
        model_download._run_download_worker('small', timeout_sec=0.8)
    assert failure.value.category == category
    assert time.monotonic() - start < 5


def test_successful_child_cannot_supply_incomplete_model(tmp_path, monkeypatch):
    monkeypatch.setattr(model_download, '_download_worker', successful_worker, raising=False)
    with pytest.raises(model_download.ModelDownloadError) as failure:
        model_download._run_download_worker(str(tmp_path), timeout_sec=5)
    assert failure.value.category == 'incomplete'


def test_rate_limit_status_survives_wrapped_cache_error():
    from types import SimpleNamespace
    original = RuntimeError('remote request rejected')
    original.response = SimpleNamespace(status_code=429)
    wrapped = FileNotFoundError('cache unavailable')
    wrapped.__cause__ = original
    assert model_download._download_error_category(wrapped) == 'rate_limit'


def listener_for_loading(monkeypatch):
    from unittest.mock import MagicMock
    from test_voice_listener import _create_mock_config, _mock_input_devices
    from jarvis.listening import listener as module

    monkeypatch.setattr(module, 'FASTER_WHISPER_AVAILABLE', True)
    monkeypatch.setattr(module, 'MLX_WHISPER_AVAILABLE', False)
    audio = MagicMock()
    _mock_input_devices(audio, [{'name': 'Test microphone', 'max_input_channels': 1}])
    audio.InputStream.side_effect = RuntimeError('No physical microphone in test')
    monkeypatch.setattr(module, 'sd', audio)
    listener = module.VoiceListener(MagicMock(), _create_mock_config(whisper_device='cpu'),
                                    None, MagicMock())
    listener.cfg.whisper_min_confidence = 0.3
    listener.cfg.whisper_no_speech_threshold = 0.5
    monkeypatch.setattr(listener, '_start_llm_warmup', lambda: [])
    return listener, module


def test_failed_preparation_never_enters_model_or_audio_loading(monkeypatch, capsys):
    listener, module = listener_for_loading(monkeypatch)
    def fail_preparation(name):
        raise model_download.ModelDownloadError('worker_exit', 'download child aborted')
    monkeypatch.setattr(model_download, 'prepare_faster_whisper_model', fail_preparation)
    def unsafe_model(*args, **kwargs):
        pytest.fail('A failed isolated download must never reach WhisperModel')
    monkeypatch.setattr(module, 'WhisperModel', unsafe_model)
    listener.run()
    output = capsys.readouterr().out
    assert 'download child aborted' in output
    assert 'Listening!' not in output


def test_listener_passes_complete_path_and_forbids_network_loading(tmp_path, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    listener, module = listener_for_loading(monkeypatch)
    ready = str(complete_model(tmp_path / 'ready'))
    monkeypatch.setattr(model_download, 'prepare_faster_whisper_model', lambda name: ready)
    def load(path, **options):
        assert path == ready
        assert options['local_files_only']
        def transcribe(audio, **kwargs):
            segment = SimpleNamespace(text='hello Jarvis', avg_logprob=0.0, no_speech_prob=0.0)
            return iter([segment]), SimpleNamespace(language='en')
        return SimpleNamespace(transcribe=transcribe, model=SimpleNamespace(device='cpu'))
    monkeypatch.setattr(module, 'WhisperModel', load)
    listener.run()
    assert listener._transcribe_audio(np.zeros(16000)) == ('hello Jarvis', 'en', ())


def test_runtime_cpu_recovery_uses_isolated_local_model(tmp_path, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    listener, module = listener_for_loading(monkeypatch)
    listener._whisper_device = 'cuda'
    def broken_decode(*args, **kwargs):
        raise RuntimeError('CUDA runtime failed')
    listener.model = SimpleNamespace(transcribe=broken_decode)
    ready = str(complete_model(tmp_path / 'ready'))
    monkeypatch.setattr(model_download, 'prepare_faster_whisper_model', lambda name: ready)
    def load(path, **options):
        assert path == ready
        assert options['local_files_only']
        assert options['device'] == 'cpu'
        def transcribe(*args, **kwargs):
            segment = SimpleNamespace(text='recovered speech', avg_logprob=0.0, no_speech_prob=0.0)
            return iter([segment]), SimpleNamespace(language='en')
        return SimpleNamespace(transcribe=transcribe, model=SimpleNamespace(device='cpu'))
    monkeypatch.setattr(module, 'WhisperModel', load)
    assert listener._transcribe_audio(np.zeros(16000)) == ('recovered speech', 'en', ())


def test_local_directory_cannot_trigger_tokeniser_download(tmp_path, monkeypatch):
    model = complete_model(tmp_path / 'local')
    (model / 'tokenizer.json').unlink()
    import faster_whisper.utils
    def forbidden_download(*args, **kwargs):
        pytest.fail('An incomplete local directory must not access the Hub')
    monkeypatch.setattr(faster_whisper.utils, 'download_model', forbidden_download)
    with pytest.raises(model_download.ModelDownloadError) as failure:
        model_download.prepare_faster_whisper_model(str(model))
    assert failure.value.category == 'incomplete'


def test_local_directory_is_available_with_no_worker(tmp_path, monkeypatch):
    model = complete_model(tmp_path / 'local')
    def forbidden_worker(*args, **kwargs):
        pytest.fail('A complete local directory must not start a worker')
    monkeypatch.setattr(model_download, '_run_download_worker', forbidden_worker)
    assert model_download.prepare_faster_whisper_model(str(model)) == str(model)


def test_rate_limited_preparation_recovers_before_listener_can_decode(tmp_path, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    listener, module = listener_for_loading(monkeypatch)
    ready = str(complete_model(tmp_path / 'ready'))
    results = iter([model_download.ModelDownloadError('rate_limit', 'HTTP 429'), ready])
    def prepare(name):
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(model_download, 'prepare_faster_whisper_model', prepare)
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: None)
    def load(path, **options):
        assert path == ready
        assert options['local_files_only']
        def transcribe(*args, **kwargs):
            segment = SimpleNamespace(text='speech after retry', avg_logprob=0.0, no_speech_prob=0.0)
            return iter([segment]), SimpleNamespace(language='en')
        return SimpleNamespace(transcribe=transcribe, model=SimpleNamespace(device='cpu'))
    monkeypatch.setattr(module, 'WhisperModel', load)
    listener.run()
    assert listener._transcribe_audio(np.zeros(16000)) == ('speech after retry', 'en', ())


def test_partial_cache_cannot_hide_remote_rate_limit(tmp_path, monkeypatch):
    import requests
    import huggingface_hub
    from huggingface_hub.utils import HfHubHTTPError

    partial = tmp_path / 'partial'
    partial.mkdir()
    (partial / 'model.bin').write_bytes(b'partial weights')
    response = requests.Response()
    response.status_code = 429
    def snapshot(repo_id, **kwargs):
        if kwargs.get('local_files_only'):
            return str(partial)
        raise HfHubHTTPError('remote request rejected', response=response)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', snapshot)
    class Result:
        value = None
        def send(self, value):
            self.value = value
        def close(self):
            pass
    connection = Result()
    model_download._download_worker(connection, 'small')
    assert connection.value == ('error', 'rate_limit', 'HTTP 429')
