"""A local compatible server consumes tool history without rewriting its owner."""

from copy import deepcopy
import json

import pytest

from conftest import requires_judge_llm
from helpers import voice_config
from jarvis.llm.openai_compatible import OpenAICompatibleBackend, _normalise_response

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.fixture(autouse=True)
def private_runtime_paths(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(tmp_path / "config.json"))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, str(tmp_path))


def test_live_tool_result_retains_shared_decoded_history():
    cfg = voice_config()
    base = cfg.llm_base_url.rstrip('/')
    if not base.endswith('/v1'):
        base += '/v1'
    backend = OpenAICompatibleBackend(base)
    value = "synthetic-fixture"
    decoded = _normalise_response({"choices": [{"message": {
        "role": "assistant", "content": None, "tool_calls": [{
            "id": "fixture", "type": "function", "function": {
                "name": "fixtureEcho", "arguments": json.dumps({"value": value}),
            },
        }],
    }}]})
    saved_response = deepcopy(decoded)
    history = [
        {"role": "system", "content": "Report completed tool results as JSON. Do not call another tool."},
        {"role": "user", "content": "Read the synthetic fixture."},
        decoded["message"],
        {"role": "tool", "tool_call_id": "fixture", "content": json.dumps({"value": value})},
        {"role": "user", "content": 'Return only a JSON object with the tool result in the "value" field.'},
    ]
    saved_history = deepcopy(history)
    result = backend.chat(cfg.llm_chat_model, history, timeout_sec=60,
                          extra_options={"temperature": 0, "max_tokens": 128})
    assert result and isinstance(result.get("message", {}).get("content"), str), result
    content = result["message"]["content"]
    start = content.find('{')
    assert start >= 0, result
    reported, _ = json.JSONDecoder().raw_decode(content[start:])
    assert reported == {"value": value}, result
    assert decoded == saved_response
    assert history == saved_history
