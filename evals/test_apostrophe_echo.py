"""Local intent inference remains behind the literal-echo admission boundary."""
from dataclasses import replace
from pathlib import Path
import runpy
import time

import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import isolated_voice_settings
from jarvis.listening.intent_judge import create_intent_judge
from jarvis.listening.listener import _TranscriptionResult

_helpers = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tests' / 'test_hot_window_input.py'))
pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize('processing_delay', [0, 0.06])
@pytest.mark.parametrize('heard,is_echo', [
    ("We'll see sunny weather.", True),
    ('Well?', False),
    ("We'll see sunny weather tomorrow. Tell me more please.", False),
])
def test_local_judge_receives_user_speech_and_not_literal_echo(heard, is_echo, processing_delay, capsys):
    cfg = replace(isolated_voice_settings(), hot_window_seconds=30, tune_enabled=False,
                  vad_enabled=False, echo_tolerance=0.02)
    listener, _ = _helpers['_create_listener'](hot_window_seconds=30, echo_tolerance=0.02)
    listener.cfg = cfg
    judge = create_intent_judge(cfg)
    original_judge = judge.judge
    decisions = []

    def judge_user_speech(*args, **kwargs):
        assert not is_echo, 'Literal TTS reached local inference'
        decision = original_judge(*args, **kwargs)
        decisions.append(decision)
        return decision

    judge.judge = judge_user_speech
    listener._intent_judge = judge
    try:
        listener.echo_detector.track_tts_start('We’ll see sunny weather tomorrow.')
        _helpers['_simulate_tts_finish'](listener)
        capture_finish = listener.echo_detector._last_tts_finish_time
        _helpers['_wait_for_hot_window_state'](listener)
        time.sleep(processing_delay)
        listener._handle_transcription_result(_TranscriptionResult(
            text=heard, language='en', low_confidence_events=(),
            start_time=capture_finish + 0.005, end_time=capture_finish + 0.01, energy=0.01,
            capture_generation=listener._capture_generation, captured_during_tts=False,
            captured_tts_start_time=listener.echo_detector._tts_start_time,
        ))
        if is_echo:
            assert _helpers['_accepted_query'](listener) == ''
            assert not decisions
        else:
            assert decisions and all(decision is not None for decision in decisions)
            assert 'Heard (echo)' not in capsys.readouterr().out
            if 'Tell me more please.' in heard:
                assert _helpers['_accepted_query'](listener) == 'Tell me more please.'
    finally:
        listener.state_manager.stop()
