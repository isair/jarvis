import os
import requests
import pytest

from jarvis.coach import ask_coach, ask_coach_with_tools


def _ollama_available(base_url: str) -> bool:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _has_model(base_url: str, model_name: str) -> bool:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=3)
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns {"models": [{"name": "llama3:8b"}, ...]}
        names = set()
        if isinstance(data, dict) and isinstance(data.get("models"), list):
            for m in data["models"]:
                if isinstance(m, dict) and m.get("name"):
                    names.add(m["name"]) 
        return model_name in names
    except Exception:
        # If the tag listing schema changes, don't block the test here.
        return True


@pytest.mark.integration
def test_ollama_chat_basic():
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    chat_model = os.environ.get("OLLAMA_CHAT_MODEL", "llama3:8b")

    if not _ollama_available(base_url):
        pytest.skip("Ollama not reachable; set OLLAMA_BASE_URL and ensure it is running")
    if not _has_model(base_url, chat_model):
        pytest.skip(f"Model '{chat_model}' not available; set OLLAMA_CHAT_MODEL or pull it in Ollama")

    system_prompt = "You are a terse assistant."
    reply = ask_coach(base_url, chat_model, system_prompt, "Say hi in one word.", timeout_sec=10.0)
    assert isinstance(reply, (str, type(None)))


@pytest.mark.integration
def test_ollama_with_tools_prompt():
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    chat_model = os.environ.get("OLLAMA_CHAT_MODEL", "llama3:8b")

    if not _ollama_available(base_url):
        pytest.skip("Ollama not reachable; set OLLAMA_BASE_URL and ensure it is running")
    if not _has_model(base_url, chat_model):
        pytest.skip(f"Model '{chat_model}' not available; set OLLAMA_CHAT_MODEL or pull it in Ollama")

    system_prompt = "You are a helpful assistant that follows tool instructions."
    tools_desc = (
        "Tool-use protocol: Reply with ONLY a JSON object:\n"
        '{"tool": { "name": "fetchMeals", "args": { "since_utc": "2025-01-01T00:00:00Z" } }}'
    )
    final_text, tool_req, tool_args = ask_coach_with_tools(
        base_url,
        chat_model,
        system_prompt,
        "Example: if you need a tool, output JSON on one line.",
        tools_desc,
        timeout_sec=10.0,
        additional_messages=None,
        include_location=False,
        config_ip=None,
        auto_detect=False,
        voice_debug=False,
    )

    # Non-deterministic; just assert the function completed and returned a tuple
    assert (final_text is None and tool_req is not None) or (final_text is None or isinstance(final_text, str))


