"""Actual native models consume complete, paired batch results."""
from contextlib import closing
import json
from unittest.mock import patch

import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import assert_not_fallback_reply, judge_pass_fail, voice_config
from jarvis.memory.conversation import DialogueMemory
from jarvis.memory.db import Database
from jarvis.reply import engine
from jarvis.tools.types import ToolExecutionResult

pytestmark = [pytest.mark.eval, requires_judge_llm]

_READING_CRITERIA = (
    'Judge the recorded assistant answer, do not answer the original request. '
    'PASS requires explicitly reporting London with its actual measured temperature of 12 C '
    'and Paris with its actual measured temperature of 18 C. '
    'Never accept missing, swapped or invented measured readings. '
    'Side-by-side readings count as a comparison. Correct arithmetic differences are allowed. '
    'Judge the grounding of measured readings; humour and qualitative weather comments '
    'are outside this check unless they contradict those numeric readings.'
)


@pytest.mark.parametrize('reply,expected', [
    ('London: 12 C. Paris: 18 C.', 'PASS'),
    ('London is at 12 °C, Paris 18 °C, a 6 °C gap that makes Paris feel a touch more inviting.', 'PASS'),
    ('London: 12 C.', 'FAIL'),
    ('London: 18 C. Paris: 12 C.', 'FAIL'),
])
def test_batch_reading_judge_requires_complete_correct_readings(reply, expected):
    assert judge_pass_fail(_READING_CRITERIA, 'Recorded assistant reply: ' + reply) == expected



@pytest.mark.parametrize("seed_batch", [True, False])
def test_native_city_comparison_uses_every_result(tmp_path, seed_batch):
    cfg = voice_config()
    if engine.detect_model_size(cfg.llm_chat_model) == engine.ModelSize.SMALL:
        pytest.skip("Native-tool qualification requires a native-tool model")
    cfg.location_enabled = False
    cfg.planner_enabled = False
    cfg.memory_digest_enabled = False
    cfg.tool_result_digest_enabled = False
    cfg.agentic_max_turns = 5
    cfg.db_path = str(tmp_path / "owned.db")
    readings = {"London": 12, "Paris": 18}
    executed, responses = [], []
    live_chat = engine.chat_with_messages
    seeded = False

    def chat(**kwargs):
        nonlocal seeded
        if seed_batch and not seeded:
            seeded = True
            return {"message": {"content": "", "tool_calls": [
                {"id": city, "function": {"name": "getWeather", "arguments": {"location": city}}}
                for city in readings
            ]}}
        if seed_batch and not responses:
            # Both results must exist before the native model resumes.
            assert executed == list(readings)
            history = kwargs["messages"]
            outcomes = [m for m in history if m.get("role") == "tool"]
            assert [m["tool_call_id"] for m in outcomes] == list(readings)
        response = live_chat(**kwargs)
        responses.append(response)
        return response

    def weather(db, cfg, tool_name, tool_args, **kwargs):
        city = tool_args.get("location")
        assert tool_name == "getWeather" and city in readings, (tool_name, tool_args)
        executed.append(city)
        return ToolExecutionResult(success=True, reply_text=json.dumps({city: readings[city]}))

    with closing(Database(cfg.db_path, sqlite_vss_path=None)) as db, \
         patch.object(engine, "select_tools", return_value=["getWeather", "stop"]), \
         patch.object(engine, "plan_query", return_value=[]), \
         patch.object(engine, "extract_search_params_for_memory", return_value={}), \
         patch.object(engine, "run_tool_with_retries", side_effect=weather), \
         patch.object(engine, "chat_with_messages", side_effect=chat):
        reply = engine.run_reply_engine(db, cfg, None,
            'Get the weather separately for London and Paris, then compare their measured temperatures.', DialogueMemory())
    assert_not_fallback_reply(reply)
    assert set(executed) == set(readings)
    assert responses, "No actual native model response"
    assert all(city in reply for city in readings), reply
    assert all(str(value) in reply for value in readings.values()), reply
    verdict = judge_pass_fail(_READING_CRITERIA, 'Recorded assistant reply: ' + reply)
    assert verdict == "PASS", (reply, verdict)
