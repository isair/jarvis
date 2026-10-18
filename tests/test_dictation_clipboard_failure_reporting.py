"""Failed clipboard delivery is visible while saved dictation remains available."""
from types import SimpleNamespace
from unittest.mock import MagicMock
import time

import numpy as np
import pytest

from jarvis.dictation import dictation_engine
from jarvis.dictation.history import DictationHistory

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.mark.parametrize('system,writer', [
    ('Windows', '_clipboard_windows'),
    ('Darwin', '_clipboard_macos'),
    ('Linux', '_clipboard_linux'),
])
@pytest.mark.parametrize('debug', ['0', '1'])
def test_clipboard_failure_is_safe_and_preserves_saved_result(
        monkeypatch, tmp_path, capsys, system, writer, debug):
    monkeypatch.setenv('JARVIS_VOICE_DEBUG', debug)
    monkeypatch.setattr(dictation_engine.platform, 'system', lambda: system)
    monkeypatch.setattr(dictation_engine, 'pynput_keyboard', None)
    monkeypatch.setattr(dictation_engine, 'parse_hotkey', lambda _: (frozenset(), None))
    monkeypatch.setattr(dictation_engine, '_play_beep', lambda *_: None)
    text = 'Complete synthetic dictation.'
    error_marker = 'private-clipboard-error-marker'

    def failed_copy(_):
        raise OSError(error_marker)

    def forbidden_paste(*_):
        pytest.fail('Clipboard failure must not attempt automatic paste')

    monkeypatch.setattr(dictation_engine, writer, failed_copy)
    monkeypatch.setattr(dictation_engine, '_check_macos_accessibility', forbidden_paste)
    monkeypatch.setattr(dictation_engine, '_paste_cgevent', forbidden_paste)
    monkeypatch.setattr(dictation_engine, 'pynput_keyboard', SimpleNamespace(Controller=forbidden_paste))
    path = tmp_path / 'history.json'
    history = DictationHistory(path=path)
    history.add('Earlier synthetic record.', duration=1)
    model = MagicMock()
    model.transcribe.return_value = ([SimpleNamespace(text=text)], SimpleNamespace())
    saved, phases = [], []
    engine = dictation_engine.DictationEngine(
        whisper_model_ref=lambda: model, whisper_backend_ref=lambda: 'faster-whisper',
        mlx_repo_ref=lambda: None, history=history, on_dictation_result=saved.append,
        on_dictation_processing_start=lambda: phases.append('processing'),
        on_dictation_end=lambda: phases.append('idle'),
    )
    frames = [np.zeros(int(engine._target_sample_rate * 0.5), dtype=np.float32)]
    engine._finalise_and_transcribe(None, frames, time.monotonic() - 0.5,
                                   engine._target_sample_rate, engine._session)
    output = capsys.readouterr()
    assert '⚠️ Dictation could not be copied to the clipboard.' in output.out
    assert error_marker not in output.out + output.err
    assert 'The text is on your clipboard' not in output.out
    assert phases == ['processing', 'idle']
    assert [entry['text'] for entry in saved] == [text]
    assert [entry['text'] for entry in DictationHistory(path=path).get_all()] == [text, 'Earlier synthetic record.']
