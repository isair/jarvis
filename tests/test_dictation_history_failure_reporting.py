"""Dictation storage failures are visible without losing successful delivery."""
from types import SimpleNamespace
from unittest.mock import MagicMock
import time

import numpy as np
import pytest

from jarvis.dictation import dictation_engine
from jarvis.dictation.history import DictationHistory

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.mark.parametrize('failure', ['corrupt', 'read', 'write'])
def test_history_failure_reports_safely_and_can_recover(monkeypatch, tmp_path, capsys, failure):
    monkeypatch.setenv('JARVIS_VOICE_DEBUG', '0')
    monkeypatch.setattr(dictation_engine, 'pynput_keyboard', None)
    monkeypatch.setattr(dictation_engine, 'parse_hotkey', lambda _: (frozenset(), None))
    monkeypatch.setattr(dictation_engine, '_play_beep', lambda *args: None)
    path = tmp_path / 'history.json'
    history = DictationHistory(path=path)
    history.add('Earlier synthetic record.', duration=1)
    valid = path.read_bytes()
    text = 'Complete synthetic dictation.'
    model = MagicMock()
    model.transcribe.return_value = ([SimpleNamespace(text=text)], SimpleNamespace())
    pasted, delivered, phases = [], [], []
    monkeypatch.setattr(dictation_engine, '_clipboard_paste', pasted.append)
    engine = dictation_engine.DictationEngine(
        whisper_model_ref=lambda: model, whisper_backend_ref=lambda: 'faster-whisper',
        mlx_repo_ref=lambda: None, history=history, on_dictation_result=delivered.append,
        on_dictation_processing_start=lambda: phases.append('processing'),
        on_dictation_end=lambda: phases.append('idle'),
    )
    frames = [np.zeros(int(engine._target_sample_rate * 0.5), dtype=np.float32)]

    with monkeypatch.context() as fault:
        if failure == 'corrupt':
            path.write_bytes(b'Synthetic broken history')
        else:
            def unavailable():
                raise OSError('private-storage-error-marker')
            fault.setattr(history, '_read_entries' if failure == 'read' else '_save', unavailable)
        protected = path.read_bytes()
        engine._finalise_and_transcribe(None, frames, time.time() - 0.5,
                                       engine._target_sample_rate, engine._session)
        output = capsys.readouterr()
        assert 'Dictation could not be saved to history' in output.out
        assert '⚠️' in output.out
        assert 'private-storage-error-marker' not in output.out + output.err
        assert path.read_bytes() == protected
        assert pasted == [text]
        assert delivered == []
        assert phases == ['processing', 'idle']

    if failure == 'corrupt':
        path.write_bytes(valid)
    engine._finalise_and_transcribe(None, frames, time.time() - 0.5,
                                   engine._target_sample_rate, engine._session)
    assert pasted == [text, text]
    assert [entry['text'] for entry in delivered] == [text]
    assert phases == ['processing', 'idle', 'processing', 'idle']
    assert [entry['text'] for entry in DictationHistory(path=path).get_all()] == [text, 'Earlier synthetic record.']
    assert 'Dictation could not be saved to history' not in capsys.readouterr().out
