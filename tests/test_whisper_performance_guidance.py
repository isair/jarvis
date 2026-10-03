"""Slow speech decoding gives useful guidance without changing user settings."""
from types import SimpleNamespace
from unittest.mock import MagicMock
import numpy as np
import pytest
from jarvis.listening.listener import VoiceListener, _TranscriptionJob

pytestmark = pytest.mark.unit


def run_speech(monkeypatch, durations, model='medium', audio_seconds=2, transcript='hello'):
    cfg = SimpleNamespace(sample_rate=16000, vad_enabled=False, echo_tolerance=0.3,
                          echo_energy_threshold=2, hot_window_seconds=3,
                          tune_enabled=False, whisper_model=model)
    listener = VoiceListener(MagicMock(), cfg, None, MagicMock())
    listener._whisper_model_name = model
    listener._transcribe_audio = lambda audio: (transcript, 'en', ())
    clock = iter(t for index, elapsed in enumerate(durations) for t in (index*20, index*20+elapsed))
    monkeypatch.setattr('jarvis.listening.listener.time.monotonic', lambda: next(clock))
    for _ in durations:
        listener._transcription_jobs_q.put(_TranscriptionJob(
            np.zeros(int(audio_seconds*16000)), 1, 3, 0.1, 0, False, 0))
    listener._transcription_jobs_q.put(None)
    listener._run_transcription_worker()
    results=[]
    while not listener._transcription_results_q.empty():
        results.append(listener._transcription_results_q.get().text)
    assert results == [transcript]*len(durations)
    assert cfg.whisper_model == model


@pytest.mark.parametrize('model,recommended', [('medium','small'),('small.en','base.en')])
def test_sustained_slow_decode_recommends_smaller_model_once(monkeypatch,capsys,model,recommended):
    run_speech(monkeypatch,[5,5,5,5,5,5],model)
    output=capsys.readouterr().out
    assert output.count('speech recognition is slower than real time') == 1
    assert f"'{recommended}'" in output
    assert 'Setup Wizard' in output


@pytest.mark.parametrize('durations,audio_seconds', [([5,0.5,5,0.5,5],2),([0.5]*4,2),([5]*4,0.2)])
def test_fast_intermittent_or_too_short_samples_do_not_warn(monkeypatch,capsys,durations,audio_seconds):
    run_speech(monkeypatch,durations,audio_seconds=audio_seconds)
    assert 'slower than real time' not in capsys.readouterr().out


def test_smallest_model_does_not_recommend_another_model(monkeypatch,capsys):
    run_speech(monkeypatch,[5]*3,model='tiny')
    output=capsys.readouterr().out
    assert 'slower than real time' in output
    assert 'smaller model' not in output
    assert 'accelerated' in output


def test_empty_decodes_do_not_recommend_model_changes(monkeypatch,capsys):
    run_speech(monkeypatch,[5]*4,transcript='')
    assert 'slower than real time' not in capsys.readouterr().out
