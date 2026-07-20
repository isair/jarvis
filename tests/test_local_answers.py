"""Deterministic answers for date, clock and arithmetic.

These are facts the process already holds. Routing them through gemma4:e2b
turned certainties into guesses — it invented weekdays and the non-word
"vigileoptitudine". The matchers must be narrow: a false positive hijacks a
question the model should have answered.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.jarvis.reply.local_answers import try_local_answer

NOW = datetime(2026, 7, 20, 18, 59)      # a Monday
NOON = datetime(2026, 7, 20, 12, 0)


# ------------------------------------------------------------------- date

@pytest.mark.unit
@pytest.mark.parametrize("q", [
    "ce zi este astăzi", "ce zi este azi", "ce zi e azi",
    "în ce zi suntem", "in ce zi suntem",          # diacritics dropped by ASR
    "data de astăzi", "azi ce zi e",
])
def test_date_questions_answered_locally(q):
    a = try_local_answer(q, NOW)
    assert a is not None, f"{q!r} must be answered locally"
    assert "luni" in a and "20" in a and "iulie" in a and "2026" in a


@pytest.mark.unit
def test_date_uses_romanian_names_not_english():
    a = try_local_answer("ce zi este astăzi", NOW)
    for english in ("Monday", "July", "Mon", "Jul"):
        assert english not in a, "must not leak English calendar names"


# ------------------------------------------------------------------- time

@pytest.mark.unit
@pytest.mark.parametrize("q", [
    "cât este ceasul", "cat este ceasul",
    "câte este ceasul", "cate este ceasul",  # Whisper ASR live FAIL
    "cât e ceasul", "câte e ceasul",
    "cat e ceasul", "cate e ceasul",
    "ce oră este", "spune-mi ora exactă", "cât e ora",
])
def test_time_questions_answered_locally(q):
    a = try_local_answer(q, NOW)
    assert a is not None, f"{q!r} must be answered locally"
    assert "18" in a and "59" in a


@pytest.mark.unit
def test_live_asr_cate_este_ceasul_answered_locally():
    """Real Whisper transcript that previously fell through to Gemma."""
    a = try_local_answer("câte este ceasul", NOW)
    assert a is not None
    assert "18" in a and "59" in a
    assert "de minute" in a  # 59 ≥ 20 → Romanian "de"


@pytest.mark.unit
def test_whole_hour_phrasing():
    assert try_local_answer("cât este ceasul", NOON) == "Este ora 12 fix."
    assert try_local_answer("câte este ceasul", NOON) == "Este ora 12 fix."


@pytest.mark.unit
@pytest.mark.parametrize("minute,expected", [
    (0,  "Este ora 19 fix."),
    (1,  "Este ora 19 și un minut."),
    (2,  "Este ora 19 și 2 minute."),
    (9,  "Este ora 19 și 9 minute."),
    (19, "Este ora 19 și 19 minute."),
    (20, "Este ora 19 și 20 de minute."),
    (21, "Este ora 19 și 21 de minute."),
    (59, "Este ora 19 și 59 de minute."),
])
def test_minute_agreement_follows_romanian_not_english(minute, expected):
    """Romanian inserts "de" from 20 upward. An English one/many split
    produced "9 de minute", heard twice in live testing."""
    assert try_local_answer("cât este ceasul",
                            datetime(2026, 7, 20, 19, minute)) == expected


@pytest.mark.unit
def test_no_de_minute_below_twenty():
    for m in range(1, 20):
        a = try_local_answer("cât este ceasul", datetime(2026, 7, 20, 19, m))
        assert "de minute" not in a, f"minute={m} must not use 'de'"


@pytest.mark.unit
def test_de_minute_from_twenty_up():
    for m in range(20, 60):
        a = try_local_answer("cât este ceasul", datetime(2026, 7, 20, 19, m))
        assert "de minute" in a, f"minute={m} must use 'de'"


# ------------------------------------------------------------- arithmetic

@pytest.mark.unit
@pytest.mark.parametrize("q,expected", [
    ("cât fac zece plus cincisprezece", 25),
    ("cât fac 10 plus 15", 25),                     # ASR normalises to digits
    ("cat fac douazeci minus trei", 17),
    ("cât fac cinci ori patru", 20),
    ("cât fac zece împărțit la doi", 5),
    ("cât face doi plus doi", 4),
])
def test_arithmetic_is_computed_not_guessed(q, expected):
    a = try_local_answer(q, NOW)
    assert a is not None, f"{q!r} must be computed locally"
    assert str(expected) in a


@pytest.mark.unit
def test_division_by_zero_is_refused_not_crashed():
    a = try_local_answer("cât fac zece împărțit la zero", NOW)
    assert a is not None and "zero" in a.lower()


# --------------------------------------------------- must NOT be hijacked

@pytest.mark.unit
@pytest.mark.parametrize("q", [
    "explică pe scurt ce este un SSD",
    "cum e vremea în Cluj-Napoca",
    "spune-mi o poveste",
    "ce părere ai despre muzica clasică",
    "cine a scris Luceafărul",
    "câte ceasuri sunt",          # plural clocks — not "what time is it"
    "cate ceasuri sunt",
    "ceasul biologic",            # contains "ceasul" but not the clock query
    "",
    "   ",
])
def test_general_questions_fall_through_to_the_model(q):
    assert try_local_answer(q, NOW) is None, f"{q!r} must reach the LLM"


@pytest.mark.unit
def test_arithmetic_with_unknown_words_falls_through():
    """Better to let the model try than to answer with a wrong number."""
    assert try_local_answer("cât fac elefanți plus girafe", NOW) is None


# ------------------------------------------------------- engine integration

def _cfg():
    """Minimal cfg for the short-circuit test."""
    cfg = Mock()
    cfg.local_answers_enabled = True
    return cfg


def _full_cfg():
    """cfg complete enough to run the pipeline past the short-circuit.

    The engine coerces several settings with float()/int(), which a bare Mock
    cannot satisfy — these must be real numbers.
    """
    cfg = Mock()
    cfg.local_answers_enabled = True
    cfg.ollama_base_url = "http://localhost:11434"
    cfg.ollama_chat_model = "test-large"
    cfg.ollama_embed_model = "test-embed"
    cfg.voice_debug = False
    cfg.llm_chat_timeout_sec = 45.0
    cfg.llm_tools_timeout_sec = 8.0
    cfg.llm_embed_timeout_sec = 10.0
    cfg.llm_digest_timeout_sec = 8.0
    cfg.memory_enrichment_max_results = 5
    cfg.memory_enrichment_source = "diary"
    cfg.memory_digest_enabled = False
    cfg.tool_result_digest_enabled = False
    cfg.location_ip_address = None
    cfg.location_auto_detect = False
    cfg.location_enabled = False
    cfg.agentic_max_turns = 8
    cfg.tool_search_max_calls = 3
    cfg.tool_selection_strategy = "all"
    cfg.tool_carryover_max_turns = 2
    cfg.tool_carryover_per_entry_chars = 1200
    cfg.mcps = {}
    cfg.llm_thinking_enabled = False
    cfg.tts_engine = "none"
    cfg.response_language = "ro"
    cfg.assistant_style = "professional"
    cfg.wake_word = "jarvis"
    return cfg


@pytest.mark.unit
@pytest.mark.parametrize("text", [
    "ce zi este astăzi",
    "câte este ceasul",  # live ASR FAIL — must bypass planner/LLM/memory/tools
])
def test_engine_short_circuits_before_any_llm_or_tool_call(text):
    """The whole point: no web, no memory, no model turn."""
    from src.jarvis.reply.engine import run_reply_engine
    from src.jarvis.memory.conversation import DialogueMemory

    with patch("src.jarvis.reply.engine.chat_with_messages") as chat, \
         patch("src.jarvis.reply.engine.run_tool_with_retries") as tool, \
         patch("src.jarvis.reply.engine.plan_query") as plan, \
         patch("src.jarvis.reply.local_answers.datetime") as dt:
        dt.now.return_value = NOW
        reply = run_reply_engine(db=Mock(), cfg=_cfg(), tts=None,
                                 text=text,
                                 dialogue_memory=DialogueMemory())

    assert reply
    if "ceasul" in text:
        assert "18" in reply and "59" in reply
    else:
        assert "luni" in reply
    assert chat.call_count == 0, "no LLM turn may happen"
    assert tool.call_count == 0, "no tool may run"
    assert plan.call_count == 0, "no planning may happen"


@pytest.mark.unit
def test_engine_still_uses_the_model_for_general_questions():
    from src.jarvis.reply.engine import run_reply_engine
    from src.jarvis.memory.conversation import DialogueMemory

    cfg = _full_cfg()

    with patch("src.jarvis.reply.engine.plan_query", return_value=[]), \
         patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={}), \
         patch("src.jarvis.reply.engine.extract_text_from_response", return_value="raspuns"), \
         patch("src.jarvis.reply.engine.chat_with_messages") as chat:
        chat.return_value = {"message": {"content": "raspuns"}}
        run_reply_engine(db=Mock(), cfg=cfg, tts=None,
                         text="explică pe scurt ce este un SSD",
                         dialogue_memory=DialogueMemory())

    assert chat.call_count >= 1, "general questions must still reach the model"


@pytest.mark.unit
def test_reply_header_uses_configured_wake_word_cora(capsys):
    """Visible log label must follow wake_word, not a hardcoded Jarvis."""
    from src.jarvis.reply.engine import run_reply_engine
    from src.jarvis.memory.conversation import DialogueMemory

    cfg = _full_cfg()
    cfg.wake_word = "cora"
    cfg.local_answers_enabled = False  # force LLM print path

    with patch("src.jarvis.reply.engine.plan_query", return_value=[]), \
         patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={}), \
         patch("src.jarvis.reply.engine.extract_text_from_response", return_value="ok"), \
         patch("src.jarvis.reply.engine.chat_with_messages") as chat:
        chat.return_value = {"message": {"content": "ok"}}
        run_reply_engine(db=Mock(), cfg=cfg, tts=None,
                         text="salut",
                         dialogue_memory=DialogueMemory())

    out = capsys.readouterr().out
    assert "🤖 Cora" in out
    assert "🤖 Jarvis" not in out


@pytest.mark.unit
def test_local_answers_can_be_disabled():
    from src.jarvis.reply.engine import run_reply_engine
    from src.jarvis.memory.conversation import DialogueMemory

    cfg = _full_cfg()
    cfg.local_answers_enabled = False

    with patch("src.jarvis.reply.engine.plan_query", return_value=[]), \
         patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={}), \
         patch("src.jarvis.reply.engine.extract_text_from_response", return_value="x"), \
         patch("src.jarvis.reply.engine.chat_with_messages") as chat:
        chat.return_value = {"message": {"content": "x"}}
        run_reply_engine(db=Mock(), cfg=cfg, tts=None, text="ce zi este astăzi",
                         dialogue_memory=DialogueMemory())

    assert chat.call_count >= 1, "disabling must restore upstream behaviour"
