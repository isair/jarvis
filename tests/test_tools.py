import types
import pytest

from jarvis.tools import run_tool_with_retries, ToolExecutionResult


class DummyCfg:
    def __init__(self):
        self.voice_debug = False
        self.ollama_base_url = "http://localhost"
        self.ollama_chat_model = "test"
        self.llm_chat_timeout_sec = 5.0
        self.location_enabled = False
        self.location_ip_address = None
        self.location_auto_detect = False
        self.use_stdin = True
        self.web_search_enabled = False
        self.mcps = {}


class DummyDB:
    def get_meals_between(self, since, until):
        return []

    def delete_meal(self, mid: int) -> bool:
        return mid == 1


@pytest.mark.unit
def test_delete_meal_success(monkeypatch):
    db = DummyDB()
    cfg = DummyCfg()
    res = run_tool_with_retries(
        db=db,
        cfg=cfg,
        tool_name="DELETE_MEAL",
        tool_args={"id": 1},
        system_prompt="",
        original_prompt="",
        redacted_text="",
        max_retries=0,
    )
    assert isinstance(res, ToolExecutionResult)
    assert res.success is True
    assert "deleted" in (res.reply_text or "").lower()


@pytest.mark.unit
def test_delete_meal_failure(monkeypatch):
    db = DummyDB()
    cfg = DummyCfg()
    res = run_tool_with_retries(
        db=db,
        cfg=cfg,
        tool_name="DELETE_MEAL",
        tool_args={"id": 2},
        system_prompt="",
        original_prompt="",
        redacted_text="",
        max_retries=0,
    )
    assert res.success is False


@pytest.mark.unit
def test_mcp_invocation_json(monkeypatch):
    db = DummyDB()
    cfg = DummyCfg()
    cfg.mcps = {"fake": {"transport": "stdio", "command": "x", "args": []}}

    # Mock MCPClient usage inside tools
    class FakeClient:
        def __init__(self, conf):
            self.conf = conf

        def invoke_tool(self, server_name: str, tool_name: str, arguments=None):
            assert server_name == "fake"
            assert tool_name == "alpha"
            assert arguments == {"p": 1}
            return {"text": "ok", "isError": False}

    # Patch the symbol used by the tools module (observable seam)
    import jarvis.tools as tools_mod
    monkeypatch.setattr(tools_mod, "MCPClient", FakeClient)

    res = run_tool_with_retries(
        db=db,
        cfg=cfg,
        tool_name="MCP",
        tool_args={"server": "fake", "name": "alpha", "args": {"p": 1}},
        system_prompt="",
        original_prompt="",
        redacted_text="",
        max_retries=0,
    )
    assert res.success is True
    assert res.reply_text == "ok"

@pytest.mark.unit
def test_mcp_missing_required_fields_returns_error(monkeypatch):
    db = DummyDB()
    cfg = DummyCfg()
    res = run_tool_with_retries(
        db=db,
        cfg=cfg,
        tool_name="MCP",
        tool_args={"server": "fake"},  # missing name
        system_prompt="",
        original_prompt="",
        redacted_text="",
        max_retries=0,
    )
    assert res.success is False
    assert isinstance(res.error_message, str)
    assert "server" in res.error_message.lower() or "name" in res.error_message.lower()


@pytest.mark.unit
def test_mcp_client_error_propagates_to_reply_text(monkeypatch):
    db = DummyDB()
    cfg = DummyCfg()
    cfg.mcps = {"fake": {"transport": "stdio", "command": "x", "args": []}}

    class FakeClient:
        def __init__(self, conf):
            self.conf = conf

        def invoke_tool(self, server_name: str, tool_name: str, arguments=None):
            return {"text": "boom", "isError": True}

    import jarvis.tools as tools_mod
    monkeypatch.setattr(tools_mod, "MCPClient", FakeClient)

    res = run_tool_with_retries(
        db=db,
        cfg=cfg,
        tool_name="MCP",
        tool_args={"server": "fake", "name": "alpha", "args": {}},
        system_prompt="",
        original_prompt="",
        redacted_text="",
        max_retries=0,
    )
    assert res.success is False
    assert res.reply_text == "boom"
    assert res.error_message == "boom"
