"""Live follow-up replies recover exact arguments from a completed text-tool request."""
from contextlib import closing
import json
from unittest.mock import patch

import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import voice_config
from jarvis.memory.conversation import DialogueMemory
from jarvis.memory.db import Database
from jarvis.reply import engine
from jarvis.tools.types import ToolExecutionResult

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize("location", ["Port Aster-Q7", "Gümüş-K4"])
def test_followup_recovers_exact_text_tool_arguments(tmp_path, location):
    cfg = voice_config()
    cfg.db_path = str(tmp_path / "memory.db")
    cfg.location_enabled = False
    cfg.memory_enrichment_source = "none"
    cfg.planner_enabled = False
    cfg.memory_digest_enabled = False
    cfg.tool_result_digest_enabled = False
    cfg.agentic_max_turns = 4
    memory = DialogueMemory()
    arguments = {"location": location}
    scripted = iter([
        {"message": {"role": "assistant", "content": "tool_calls: " + json.dumps([
            {"id": "prior-reading", "function": {"name": "getWeather", "arguments": arguments}},
        ], ensure_ascii=False)}},
        {"message": {"role": "assistant", "content": "The recorded reading is 14 C."}},
    ])
    live_chat = engine.chat_with_messages
    executed = []
    native_responses = []

    def chat(**kwargs):
        response = next(scripted, None)
        if response is not None:
            return response
        response = live_chat(**kwargs)
        native_responses.append(response)
        return response

    def weather(db, cfg, tool_name, tool_args, **kwargs):
        executed.append((tool_name, tool_args))
        return ToolExecutionResult(success=True, reply_text="Recorded reading: 14 C.")

    with closing(Database(cfg.db_path, sqlite_vss_path=None)) as db, \
         patch.object(engine, "select_tools", return_value=["getWeather", "stop"]), \
         patch.object(engine, "plan_query", return_value=[]), \
         patch.object(engine, "extract_search_params_for_memory", return_value={}), \
         patch.object(engine, "run_tool_with_retries", side_effect=weather), \
         patch.object(engine, "chat_with_messages", side_effect=chat):
        assert engine.run_reply_engine(db, cfg, None, "Fetch the supplied weather reading.", memory)
        answer = engine.run_reply_engine(
            db, cfg, None,
            'Which exact location argument did the earlier getWeather request use? '
            'Read the earlier request, do not run a fresh lookup. '
            'Return only a JSON object with "location", or null if that argument is unavailable.',
            memory,
        )
    assert answer, "No follow-up reply"
    assert native_responses, "No live follow-up response"
    raw = engine.extract_text_from_response(native_responses[-1]) or ""
    start = raw.find("{")
    assert start >= 0, raw
    result, _ = json.JSONDecoder().raw_decode(raw[start:])
    assert result == {"location": location}, raw
    assert answer == location, "Delivered reply must retain the exact recorded argument"
    assert executed == [("getWeather", arguments)], "Follow-up must reuse the recorded request"
