"""Speech remains usable when CUDA fails during lazy decoding."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from jarvis.listening.listener import VoiceListener

pytestmark = pytest.mark.unit


def listener_with_model(monkeypatch, device, transcribe):
    monkeypatch.setattr('jarvis.listening.listener.MLX_WHISPER_AVAILABLE', False)
    cfg = SimpleNamespace(sample_rate=16000, vad_enabled=False, echo_tolerance=0.3,
                          echo_energy_threshold=2.0, hot_window_seconds=3.0,
                          tune_enabled=False, whisper_model='small',
                          whisper_compute_type='float16', whisper_min_confidence=0.3,
                          whisper_no_speech_threshold=0.5)
    listener = VoiceListener(MagicMock(), cfg, None, MagicMock())
    listener._whisper_backend = 'faster-whisper'
    listener._whisper_device = device
    listener.model = SimpleNamespace(transcribe=transcribe)
    return listener


def successful_decode(audio, **kwargs):
    segment = SimpleNamespace(text='hello Jarvis', avg_logprob=0.0, no_speech_prob=0.0)
    return iter([segment]), SimpleNamespace(language='en')


@pytest.mark.parametrize('lazy', [False, True])
def test_cuda_failure_retries_utterance_and_future_speech_on_cpu(monkeypatch, lazy):
    def broken_decode(audio, **kwargs):
        def segments():
            raise RuntimeError('CUDA failed with error invalid device ordinal')
            yield
        if lazy:
            return segments(), SimpleNamespace(language='en')
        raise RuntimeError('CUDA failed with error invalid device ordinal')

    listener = listener_with_model(monkeypatch, 'cuda', broken_decode)
    def load_cpu(name, **kwargs):
        assert kwargs['device'] == 'cpu'
        assert kwargs['compute_type'] != 'float16'
        def decode(audio, **options):
            assert options['without_timestamps']
            assert not options['condition_on_previous_text']
            return successful_decode(audio, **options)
        return SimpleNamespace(transcribe=decode, model=SimpleNamespace(device='cpu'))
    monkeypatch.setattr('jarvis.listening.listener.WhisperModel', load_cpu)
    audio = np.zeros(16000, dtype=np.float32)
    assert listener._transcribe_audio(audio) == ('hello Jarvis', 'en', ())
    assert listener._transcribe_audio(audio) == ('hello Jarvis', 'en', ())


@pytest.mark.parametrize('device,error', [('cpu', 'CUDA allocation failed'),
                                         ('cuda', 'invalid audio input')])
def test_other_errors_do_not_replace_the_model(monkeypatch, device, error):
    def broken_decode(audio, **kwargs):
        raise RuntimeError(error)
    listener = listener_with_model(monkeypatch, device, broken_decode)
    def unexpected_load(*args, **kwargs):
        pytest.fail('Unrelated decode error must not reload the model')
    monkeypatch.setattr('jarvis.listening.listener.WhisperModel', unexpected_load)
    assert listener._transcribe_audio(np.zeros(16000)) == ('', None, ())


def test_failed_cpu_recovery_is_not_repeated_for_every_utterance(monkeypatch):
    def broken_decode(audio, **kwargs):
        raise RuntimeError('cuDNN runtime failure')
    listener = listener_with_model(monkeypatch, 'cuda', broken_decode)
    attempted = False
    def failed_load(*args, **kwargs):
        nonlocal attempted
        if attempted:
            pytest.fail('CPU recovery must be bounded')
        attempted = True
        raise RuntimeError('CPU model unavailable')
    monkeypatch.setattr('jarvis.listening.listener.WhisperModel', failed_load)
    for _ in range(3):
        assert listener._transcribe_audio(np.zeros(16000)) == ('', None, ())
    assert attempted


def test_dictation_uses_replacement_published_while_waiting_for_lock(monkeypatch):
    from jarvis.dictation.dictation_engine import DictationEngine
    monkeypatch.setattr("jarvis.dictation.dictation_engine.parse_hotkey", lambda hotkey: (set(), None))
    current = SimpleNamespace(model=SimpleNamespace(transcribe=lambda *a, **k: ([], None)))
    class PublishingLock:
        def __enter__(self):
            current.model = SimpleNamespace(transcribe=successful_decode)
        def __exit__(self, *args):
            pass
    engine = DictationEngine(whisper_model_ref=lambda: current.model,
                             whisper_backend_ref=lambda: 'faster-whisper',
                             mlx_repo_ref=lambda: None,
                             transcribe_lock=PublishingLock())
    assert engine._transcribe(np.zeros(16000)) == 'hello Jarvis'


def test_startup_warmup_recovers_cuda_before_listening(monkeypatch):
    def broken_decode(audio, **kwargs):
        raise RuntimeError('CUDA failed with error invalid device ordinal')
    listener = listener_with_model(monkeypatch, 'cuda', broken_decode)
    listener.cfg.whisper_device = 'cuda'
    listener.cfg.whisper_backend = 'faster-whisper'
    listener.cfg.voice_device = None
    listener.cfg.voice_debug = False
    listener.cfg.vad_frame_ms = 20
    gpu = SimpleNamespace(transcribe=broken_decode, model=SimpleNamespace(device='cuda'))
    cpu = SimpleNamespace(transcribe=successful_decode, model=SimpleNamespace(device='cpu'))
    def load(name, **kwargs):
        return cpu if kwargs['device'] == 'cpu' else gpu
    monkeypatch.setattr('jarvis.listening.listener.WhisperModel', load)
    monkeypatch.setattr('jarvis.listening.listener.FASTER_WHISPER_AVAILABLE', True)
    monkeypatch.setattr(listener, '_start_llm_warmup', lambda: [])
    sd = MagicMock()
    sd.query_devices.return_value = [{'name': 'Test Mic', 'max_input_channels': 1}]
    sd.InputStream.side_effect = RuntimeError('No physical microphone in test')
    monkeypatch.setattr('jarvis.listening.listener.sd', sd)
    listener.run()
    assert listener._transcribe_audio(np.zeros(16000)) == ('hello Jarvis', 'en', ())

@pytest.fixture(autouse=True)
def cached_whisper_files(monkeypatch):
    """Model-loading tests use synthetic local files and never access the Hub."""
    monkeypatch.setattr("jarvis.listening.model_download.prepare_faster_whisper_model",
                        lambda name: name)
