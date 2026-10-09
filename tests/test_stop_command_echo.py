"""Spoken control commands do not turn the assistant's own echo into an interruption."""
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from jarvis.listening.listener import VoiceListener

pytestmark=pytest.mark.unit


class SpeakingTTS:
    enabled=True
    def __init__(self):
        self.speaking=True
    def is_speaking(self):
        return self.speaking
    def interrupt(self):
        self.speaking=False


def process(text, spoken, commands):
    cfg=SimpleNamespace(sample_rate=16000,vad_enabled=False,echo_tolerance=0.3,
                        echo_energy_threshold=2,hot_window_seconds=3,tune_enabled=False,
                        whisper_model='small',stop_commands=commands,tts_rate=200,
                        voice_debug=False,hot_window_enabled=False)
    tts=SpeakingTTS()
    listener=VoiceListener(MagicMock(),cfg,tts,MagicMock())
    listener.echo_detector.track_tts_start(spoken)
    start=listener.echo_detector._tts_start_time
    listener._process_transcript(text,utterance_energy=0.005,
                                 utterance_start_time=start+0.2,utterance_end_time=start+1,
                                 captured_during_tts=True,captured_tts_start_time=start, generation=listener._capture_generation)
    listener.state_manager.stop()
    return tts.speaking


@pytest.mark.parametrize('phrase,command',[
    ('You can say stop to interrupt my answer.','stop'),
    ('You can say SHUT UP to interrupt my answer!','shut up'),
    ('Sie können ruhig sagen, um mich zu unterbrechen.','ruhig'),
    ('您可以说停止让我结束回答。','停止'),
])
def test_literal_spoken_echo_keeps_the_reply_playing(phrase,command):
    assert process(phrase,phrase,[command])


@pytest.mark.parametrize('command',['stop','shut up','ruhig','停止'])
def test_standalone_command_interrupts_even_when_present_in_tts(command):
    assert not process(command,f'The configured command is {command}.',[command])


def test_mixed_echo_and_appended_stop_interrupts():
    spoken='The weather today is sunny and warm.'
    assert not process(spoken+' stop',spoken,['stop'])


def test_fuzzy_standalone_command_keeps_interrupt_priority():
    assert not process('stopp','You can say stop to interrupt my answer.',['stop'])


@pytest.mark.parametrize('spoken,heard,command',[
    ('You can say “stop” any time.','you can say stop any time','stop'),
    ('You can say stop to interrupt my answer.','say stop to interrupt','stop'),
    ('يمكنك أن تقول توقف لإيقاف جوابي.','يمكنك أن تقول توقف لإيقاف جوابي.','توقف'),
])
def test_normalised_and_partial_echo_does_not_interrupt(spoken,heard,command):
    assert process(heard,spoken,[command])


@pytest.mark.parametrize('spoken,command',[
    ('You can say stop to interrupt my answer.','stop'),
    ('您可以说停止让我结束回答。','停止'),
])
def test_appended_control_is_not_lost_when_echo_contains_the_same_command(spoken,command):
    assert not process(spoken+' '+command,spoken,[command])
