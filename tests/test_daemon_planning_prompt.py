import pytest


@pytest.mark.unit
def test_planning_prompt_no_formatting_error(monkeypatch):
    # Import here to ensure module load after monkeypatch if needed
    import jarvis.daemon as daemon

    # Provide a minimal plan with only a finalResponse step
    plan_json = (
        '{\n'
        '  "steps": [\n'
        '    {\n'
        '      "step": 1,\n'
        '      "action": "finalResponse",\n'
        '      "description": "Synthesize and respond to user"\n'
        '    }\n'
        '  ]\n'
        '}'
    )

    def _fake_ask_coach_with_tools(base_url, chat_model, system_prompt, user_content, tools_desc, **kwargs):
        # Return the plan; no immediate tool request
        return plan_json, None, None

    def _fake_ask_coach(base_url, chat_model, system_prompt, user_content, **kwargs):
        return "ok"

    monkeypatch.setattr(daemon, "ask_coach_with_tools", _fake_ask_coach_with_tools)
    monkeypatch.setattr(daemon, "ask_coach", _fake_ask_coach)

    class _Cfg:
        # Minimal attributes used in the code path
        voice_debug = False
        ollama_base_url = "http://localhost:11434"
        ollama_chat_model = "test-model"
        llm_multi_step_timeout_sec = 1.0
        llm_chat_timeout_sec = 1.0
        location_enabled = False
        location_ip_address = None
        location_auto_detect = False

    cfg = _Cfg()

    # Call the function under test; should not raise and should return "ok"
    result = daemon._execute_multi_step_plan(
        db=None,
        cfg=cfg,
        system_prompt="sys",
        initial_prompt="user",
        initial_tool_req=None,
        initial_tool_args=None,
        initial_reply=None,
        redacted_text="what do you think my interests are?",
        recent_messages=[],
        allowed_tools=["fetchMeals"],
        tools_desc="",
        conversation_context="",
    )

    assert result == "ok"


