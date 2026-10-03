"""Observable routing behaviour against a local typed-decision HTTP service."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from jarvis.config import load_settings
from jarvis.tools.selection import ToolSelectionStrategy, select_tools


@pytest.fixture
def decision_server():
    state = {"requests": [], "scores": {}, "status": 200}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append((self.path, body))
            answers = {
                key: {"type": "noul", "noul": state["scores"].get(key, 0.01)}
                for key in body["questions"]
            }
            payload = state.get("payload", {"answers": answers})
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()
    server.server_close()
    thread.join()


def catalogue():
    return {
        "getWeather": SimpleNamespace(description="Fetch weather forecasts for a location."),
        "webSearch": SimpleNamespace(description="Search the web for information."),
        "logMeal": SimpleNamespace(description="Record a meal in the diary."),
        "stop": SimpleNamespace(description="Dismiss the assistant."),
    }


def route(server, **kwargs):
    return select_tools(
        query="Compare the forecast with the latest transport news",
        builtin_tools=catalogue(),
        mcp_tools={},
        strategy=ToolSelectionStrategy.DECISION,
        decision_base_url=server["url"],
        decision_model="multilingual",
        **kwargs,
    )


def test_ranks_multiple_relevant_tools_and_keeps_stop(decision_server):
    decision_server["scores"] = {"getWeather": 0.91, "webSearch": 0.97}
    assert route(decision_server) == ["webSearch", "getWeather", "stop"]


def test_confident_no_tool_decision_returns_only_stop(decision_server):
    decision_server["scores"] = {"no_tools": 0.98}
    assert route(decision_server) == ["stop"]


@pytest.mark.parametrize("payload", [
    {"answers": {}},
    {"answers": {"getWeather": {"type": "noul", "noul": 1.5}}},
    {"answers": {"getWeather": {"type": "noul", "noul": True}}},
    {"answers": {"getWeather": {"type": "noul", "noul": float("nan")}}},
    {"answers": {"getWeather": {"type": "choice", "choice": "getWeather"}}},
    {"answers": []},
    {"truncated": True, "answers": {}},
    {"usage": {"truncated": True}, "answers": {}},
    {"usage": {"state_tokens_dropped": 4}, "answers": {}},
    {"usage": {"truncated_questions": ["getWeather"]}, "answers": {}},
    [],
])
def test_incomplete_or_invalid_results_fall_back_to_router(decision_server, payload):
    decision_server["payload"] = payload
    backend = SimpleNamespace(direct=lambda *a, **kw: "webSearch")
    assert route(decision_server, llm_backend=backend) == ["webSearch", "stop"]


def test_uncertain_decision_falls_back_to_router(decision_server):
    decision_server["scores"] = {"getWeather": 0.52, "no_tools": 0.55}
    backend = SimpleNamespace(direct=lambda *a, **kw: "logMeal")
    assert route(decision_server, llm_backend=backend) == ["logMeal", "stop"]


def test_http_failure_falls_back_to_keyword_without_chat_backend(decision_server):
    decision_server["status"] = 503
    assert "webSearch" in route(decision_server)


def test_sends_full_descriptions_and_context_as_data(decision_server):
    tools = catalogue()
    tools["webSearch"].description = "x" * 150 + " Search the web."
    hint = "Recent dialogue (short-term memory):\nuser: check the weather"
    select_tools(
        "Londra'da yarın hava nasıl?", tools, {}, ToolSelectionStrategy.DECISION,
        decision_base_url=decision_server["url"], decision_model="multilingual",
        context_hint=hint,
    )
    path, body = decision_server["requests"][0]
    assert path == "/v1/systemone"
    assert body["model"] == "multilingual"
    assert body["state"]["query"] == "Londra'da yarın hava nasıl?"
    assert body["state"]["context"] == hint
    assert tools["webSearch"].description in body["questions"]["webSearch"]["instructions"]
    assert "stop" not in body["questions"]


def test_caps_catalogue_selection_and_preserves_mcp_names(decision_server):
    mcp = {f"server_tool_{i}": SimpleNamespace(description="Read a document") for i in range(9)}
    decision_server["scores"] = {name: 0.8 + i * 0.01 for i, name in enumerate(mcp)}
    result = select_tools(
        "Read documents", catalogue(), mcp, ToolSelectionStrategy.DECISION,
        decision_base_url=decision_server["url"],
    )
    assert result == list(reversed(list(mcp)))[0:5] + ["stop"]


def test_conflicting_no_tools_result_falls_back(decision_server):
    decision_server["scores"] = {"getWeather": 0.95, "no_tools": 0.96}
    backend = SimpleNamespace(direct=lambda *a, **kw: "webSearch")
    assert route(decision_server, llm_backend=backend) == ["webSearch", "stop"]


def test_mcp_no_tools_name_does_not_hide_the_tool(decision_server):
    decision_server["scores"] = {"no_tools": 0.96, "_no_tools": 0.01}
    result = select_tools(
        "Run the no_tools utility", catalogue(),
        {"no_tools": SimpleNamespace(description="Run a diagnostic utility")},
        ToolSelectionStrategy.DECISION, decision_base_url=decision_server["url"],
    )
    assert result == ["no_tools", "stop"]


def test_configured_threshold_controls_selection(decision_server):
    decision_server["scores"] = {"getWeather": 0.79, "webSearch": 0.85}
    assert route(decision_server, decision_threshold=0.8) == ["webSearch", "stop"]


def test_request_redirect_is_not_followed(decision_server):
    decision_server["status"] = 302
    decision_server["scores"] = {"getWeather": 0.95}
    backend = SimpleNamespace(direct=lambda *a, **kw: "webSearch")
    assert route(decision_server, llm_backend=backend) == ["webSearch", "stop"]


def test_environment_proxy_cannot_receive_classifier_data(decision_server, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    decision_server["scores"] = {"getWeather": 0.95}
    assert route(decision_server) == ["getWeather", "stop"]


def test_decision_configuration_round_trips(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "tool_selection_strategy": "decision",
        "tool_decision_base_url": "http://127.0.0.1:8001/",
        "tool_decision_model": "multilingual",
        "tool_decision_threshold": 0.8,
    }))
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))
    cfg = load_settings()
    assert cfg.tool_selection_strategy == "decision"
    assert cfg.tool_decision_base_url == "http://127.0.0.1:8001"
    assert cfg.tool_decision_model == "multilingual"
    assert cfg.tool_decision_threshold == 0.8


def test_tool_search_uses_configured_classifier(decision_server, tmp_path, monkeypatch):
    from jarvis.tools.base import ToolContext
    from jarvis.tools.builtin.tool_search import ToolSearchTool

    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "tool_selection_strategy": "decision",
        "tool_decision_base_url": decision_server["url"],
        "tool_decision_threshold": 0.9,
    }))
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))
    decision_server["scores"] = {"getWeather": 0.95, "webSearch": 0.8}
    context = ToolContext(None, load_settings(), "", "", "", 1, lambda text: None)
    result = ToolSearchTool().run({"query": "Check the forecast in London"}, context)
    assert result.success
    assert result.reply_text.startswith("getWeather:")
    assert "webSearch:" not in result.reply_text
