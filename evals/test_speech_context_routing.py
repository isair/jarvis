"""Speech identifies requested subjects without supplying unknown answer facts."""
import pytest
from evals.helpers import isolated_voice_settings
from evals.conftest import requires_judge_llm

pytestmark = [pytest.mark.eval, requires_judge_llm]
from evals.tool_routing import route_tools
from jarvis.listening.speech_context import SpeechContext
from jarvis.listening.transcript_buffer import TranscriptSegment


@pytest.mark.eval
@pytest.mark.parametrize('earlier,current,tool', [
    ('The new iPhone looks cool', 'Jarvis how much does that cost', 'webSearch'),
    ('How tall is Mount Everest', 'Jarvis answer that', 'webSearch'),
    ('Los AirPods suenan muy bien', 'Jarvis cuánto cuestan', 'webSearch'),
    ('', 'Jarvis, who is Jarvis Cocker?', 'webSearch'),
    ('Brandon Cronenberg directed Possessor', 'What other films has he made', 'webSearch'),
    ('How tall is Mount Everest? Charlie sands to that.', 'Jarvis answer that', 'webSearch'),
    ('The parcel reference is ZX-4821', 'Jarvis what is the parcel reference', None),
])
def test_router_resolves_speech_reference_before_selecting_capabilities(earlier, current, tool):
    cfg = isolated_voice_settings()
    reference = SpeechContext.capture([TranscriptSegment(earlier, 1, 2),
                                       TranscriptSegment(current, 3, 4)],
                                      current_text=current, assistant_names=(cfg.wake_word, *cfg.wake_aliases)).render()
    from jarvis.reply.engine import _build_enrichment_context_hint
    hint = _build_enrichment_context_hint(cfg, [])
    selected, model_reply = route_tools(cfg, current, timeout_sec=cfg.llm_tools_timeout_sec,
                                       context_hint=hint, transcript_context=reference)
    if tool:
        assert tool in selected, model_reply
    else:
        assert set(selected) <= {'stop'}, model_reply
