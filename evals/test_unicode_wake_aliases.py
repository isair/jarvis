"""Canonical wake aliases reach the native judge without changing captured speech."""
import unicodedata
import pytest
from evals.conftest import requires_judge_llm
from evals.helpers import voice_config
from jarvis.listening.intent_judge import IntentJudge, IntentJudgeConfig
from jarvis.listening.transcript_buffer import TranscriptSegment
from jarvis.listening.wake_detection import is_wake_word_detected

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize('alias', ['Frédérique', 'Αθηνά', 'Müller'])
@pytest.mark.parametrize('heard_form,alias_form', [('NFD', 'NFC'), ('NFC', 'NFD'), ('NFC', 'NFC')])
def test_canonical_alias_reaches_native_intent_judge(alias, heard_form, alias_form):
    configured = unicodedata.normalize(alias_form, alias)
    heard = unicodedata.normalize(heard_form, alias) + ', what time is it?'
    assert is_wake_word_detected(heard, 'Jarvis', [configured], fuzzy_ratio=1.0)
    cfg = voice_config()
    judge = IntentJudge(IntentJudgeConfig(assistant_name='Jarvis', aliases=[configured],
                                        cfg=cfg, model=cfg.fast_model, timeout_sec=10.0))
    segment = TranscriptSegment(heard, 1000.0, 1002.0)
    result = judge.judge(segments=[segment], wake_timestamp=1000.5,
                         last_tts_text='', last_tts_finish_time=0.0,
                         in_hot_window=False, current_text=heard)
    assert result is not None, 'Native judge returned no usable decision'
    assert result.directed is True, result
    assert result.stop is False, result
    assert segment.text == heard, 'Prompt normalisation must preserve captured speech'
