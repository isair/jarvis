import pytest

from jarvis.coach import ask_coach_with_tools


class DummyResp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise RuntimeError("http error")

    def json(self):
        return self._payload


@pytest.mark.unit
def test_json_directive_internal_tool(monkeypatch):
    # Mock requests.post to return a single message content that is the JSON directive
    def _fake_post(url, json=None, timeout=None):
        msg = {"message": {"content": '{"tool": {"name": "FETCH_MEALS", "args": {"since_utc": "2025-01-01T00:00:00Z"}}}'}}
        return DummyResp(msg)

    monkeypatch.setattr("jarvis.coach.requests.post", _fake_post)

    final_text, tool_req, tool_args = ask_coach_with_tools(
        base_url="http://localhost:11434",
        chat_model="test",
        system_prompt="sys",
        user_content="user",
        tools_desc="desc",
        timeout_sec=1.0,
        additional_messages=None,
        include_location=False,
        config_ip=None,
        auto_detect=False,
        voice_debug=False,
    )
    assert final_text is None
    assert tool_req == "FETCH_MEALS"
    assert isinstance(tool_args, dict) and tool_args.get("since_utc") == "2025-01-01T00:00:00Z"


@pytest.mark.unit
def test_json_directive_mcp_tool(monkeypatch):
    def _fake_post(url, json=None, timeout=None):
        msg = {"message": {"content": '{"tool": {"server": "fs", "name": "list", "args": {"path": "~"}}}'}}
        return DummyResp(msg)

    monkeypatch.setattr("jarvis.coach.requests.post", _fake_post)

    final_text, tool_req, tool_args = ask_coach_with_tools(
        base_url="http://localhost:11434",
        chat_model="test",
        system_prompt="sys",
        user_content="user",
        tools_desc="desc",
        timeout_sec=1.0,
        additional_messages=None,
        include_location=False,
        config_ip=None,
        auto_detect=False,
        voice_debug=False,
    )
    assert final_text is None
    assert tool_req == "MCP"
    assert isinstance(tool_args, dict) and tool_args.get("server") == "fs"


@pytest.mark.unit
def test_structured_tool_calls_function_path(monkeypatch):
    # Simulate a response with structured tool_calls array
    def _fake_post(url, json=None, timeout=None):
        payload = {
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "TOOL:DELETE_MEAL", "arguments": {"id": 5}}}
                ],
            }
        }
        return DummyResp(payload)

    monkeypatch.setattr("jarvis.coach.requests.post", _fake_post)

    final_text, tool_req, tool_args = ask_coach_with_tools(
        base_url="http://localhost:11434",
        chat_model="test",
        system_prompt="sys",
        user_content="user",
        tools_desc="desc",
        timeout_sec=1.0,
        additional_messages=None,
        include_location=False,
        config_ip=None,
        auto_detect=False,
        voice_debug=False,
    )
    assert final_text is None
    assert tool_req == "DELETE_MEAL"
    assert tool_args == {"id": 5}

