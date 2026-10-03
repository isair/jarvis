"""Speech output availability reflects whether the local model can load."""
import sys

import pytest

import src.jarvis.output.tts as speech

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('backend', ['piper', 'chatterbox'])
@pytest.mark.parametrize('entrypoint', ['start', 'speak'])
def test_failed_model_load_disables_runtime_speech_output(monkeypatch, tmp_path, backend, entrypoint, capsys):
    if backend == 'piper':
        engine = speech.PiperTTS(model_path=str(tmp_path / 'missing-voice.onnx'))
        monkeypatch.setattr(speech, '_download_piper_voice', lambda *args, **kwargs: None)
    else:
        engine = speech.ChatterboxTTS()
        monkeypatch.setitem(sys.modules, 'torch', None)
    try:
        if entrypoint == 'start':
            engine.start()
        else:
            engine.speak('A reply that cannot be synthesised')
        assert not engine.enabled
        assert not engine.is_speaking()
        engine.speak('A reply that cannot be synthesised')
        output = capsys.readouterr()
        assert 'unavailable' in (output.out + output.err).lower()
    finally:
        engine.stop()


def test_daemon_reports_failed_speech_output_without_a_success_message(monkeypatch, tmp_path, capsys):
    from src.jarvis import daemon
    engine = speech.PiperTTS(model_path=str(tmp_path / 'missing-voice.onnx'))
    monkeypatch.setattr(speech, '_download_piper_voice', lambda *args, **kwargs: None)
    try:
        daemon._start_tts_engine(engine)
        output = capsys.readouterr().out
        assert 'speech output unavailable' in output.lower()
        assert 'TTS engine started' not in output
    finally:
        engine.stop()


@pytest.mark.parametrize('backend', ['piper', 'chatterbox'])
def test_available_backend_starts_and_delivers_speech(monkeypatch, backend, capsys):
    import threading
    from src.jarvis import daemon
    engine = speech.PiperTTS() if backend == 'piper' else speech.ChatterboxTTS()
    loader = '_ensure_initialized' if backend == 'piper' else '_ensure_model'
    monkeypatch.setattr(engine, loader, lambda: True)
    delivered = []
    done = threading.Event()
    def synthesise(text):
        delivered.append(text)
        done.set()
    monkeypatch.setattr(engine, '_speak_once', synthesise)
    try:
        daemon._start_tts_engine(engine)
        engine.speak('Test reply')
        assert done.wait(timeout=1)
        assert delivered == ['Test reply']
        assert 'TTS engine started' in capsys.readouterr().out
    finally:
        engine.stop()


def test_configured_disabled_speech_output_is_reported_without_model_loading(monkeypatch, capsys):
    from src.jarvis import daemon
    engine = speech.PiperTTS(enabled=False)
    monkeypatch.setattr(engine, '_ensure_initialized', lambda: pytest.fail('unexpected model load'))
    daemon._start_tts_engine(engine)
    assert 'TTS disabled' in capsys.readouterr().out
