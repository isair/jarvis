"""Noise cannot silently bypass the listener's speech decoding guard."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from jarvis.listening.listener import VoiceListener

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


def listener_with_decoder(monkeypatch, device, decoder, threshold=0.37):
    monkeypatch.setattr('jarvis.listening.listener.create_intent_judge', lambda cfg: None)
    cfg = SimpleNamespace(sample_rate=16000, vad_enabled=False, hot_window_seconds=3,
                          voice_collect_seconds=2, voice_max_collect_seconds=60,
                          tune_enabled=False, whisper_min_confidence=0.3,
                          whisper_no_speech_threshold=threshold)
    listener = VoiceListener(MagicMock(), cfg, None, MagicMock())
    listener._whisper_backend = 'faster-whisper'
    listener._whisper_device = device
    listener.model = SimpleNamespace(transcribe=decoder)
    return listener


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('spoken', ['', 'Thank you.', 'Merci Jarvis.', 'ありがとう'])
def test_only_audio_with_speech_reaches_transcripts(monkeypatch, device, spoken):
    observed = []
    def decode(audio, **options):
        observed.append(options)
        # The backend can otherwise return confident caption artefacts for noise.
        guarded = options.get('vad_filter') and not options.get('condition_on_previous_text', True)
        text = spoken if guarded else 'Unspoken caption artefact'
        segments = ([SimpleNamespace(text=text, avg_logprob=0.0, no_speech_prob=0.0)]
                    if text else [])
        return iter(segments), SimpleNamespace(language='en')
    listener = listener_with_decoder(monkeypatch, device, decode)
    text, _, events = listener._transcribe_audio(np.zeros(16000, dtype=np.float32))
    assert observed and all(options['no_speech_threshold'] == 0.37 for options in observed)
    assert text == spoken
    assert events == ()


def test_invalid_decoder_options_cannot_trigger_an_unguarded_retry(monkeypatch):
    def decode(audio, **options):
        if 'vad_filter' in options:
            raise TypeError('Invalid speech detector options')
        return iter([SimpleNamespace(text='Unspoken caption', avg_logprob=0.0,
                                     no_speech_prob=0.0)]), SimpleNamespace(language='en')
    listener = listener_with_decoder(monkeypatch, 'cuda', decode)
    assert listener._transcribe_audio(np.zeros(16000, dtype=np.float32)) == ('', None, ())
