"""Plan recognition preserves exact registered tool names and boundaries."""

import pytest

from jarvis.reply.planner import (
    plan_has_unresolved_tool_steps,
    resolve_next_tool_call,
    tool_names_in_plan,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("name", ["local.files__read", "outil-écriture__ouvrir", "tool+name__run"])
def test_registered_names_remain_executable_plan_steps(mock_config, monkeypatch, name):
    from jarvis.reply import planner

    plan = [f"{name} path='/tmp/notes.txt'", "Reply to the user."]
    schema = [{"type": "function", "function": {
        "name": name, "description": "Read the requested local file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]},
    }}]
    def unexpected_inference(**kwargs):
        pytest.fail("A concrete registered step needs no model inference")
    monkeypatch.setattr(planner, "call_llm_direct", unexpected_inference)

    assert tool_names_in_plan(plan, [name]) == [name]
    assert not plan_has_unresolved_tool_steps(plan, [name])
    assert resolve_next_tool_call(mock_config, plan[0], [], schema) == (
        name, {"path": "/tmp/notes.txt"}
    )


@pytest.mark.parametrize("step", [
    "local.files__read path='x'",
    "local+files__read path='x'",
    "local-files__read path='x'",
])
def test_plan_does_not_authorise_a_registered_prefix_of_another_name(step):
    plan = [step, "Reply to the user."]
    assert tool_names_in_plan(plan, ["local"]) == []
    assert plan_has_unresolved_tool_steps(plan, ["local"])


@pytest.mark.parametrize("suffix", [" path='x'", ": path='x'", '({"path": "x"})', ""])
def test_exact_plan_name_preserves_supported_delimiters(suffix):
    assert tool_names_in_plan(["local.files__read" + suffix], ["local.files__read"]) == [
        "local.files__read"
    ]
