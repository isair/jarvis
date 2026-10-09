"""Selected-backend routing evaluations retain the actual model answer."""
import json
from unittest.mock import patch

import pytest

from evals.helpers import JUDGE_BASE_URL, JUDGE_MODEL, MockConfig, is_judge_llm_available
from jarvis.llm import get_llm_backend
from jarvis.tools.registry import BUILTIN_TOOLS
from jarvis.tools.selection import select_tools, ToolSelectionStrategy

requires_judge_llm = pytest.mark.skipif(
    not is_judge_llm_available(), reason="🧰 Selected routing model is unavailable",
)


def routing_config():
    return MockConfig(
        llm_provider="ollama", llm_base_url=JUDGE_BASE_URL,
        ollama_base_url=JUDGE_BASE_URL, llm_chat_model=JUDGE_MODEL,
    )


def route_tools(cfg, query, *, timeout_sec=15.0, context_hint=None, mcp_tools=None):
    """Return selected tools and a genuine router answer, rejecting fallback."""
    backend = get_llm_backend(cfg)
    model_reply = None
    direct = backend.direct
    def record_reply(*args, **kwargs):
        nonlocal model_reply
        model_reply = direct(*args, **kwargs)
        return model_reply

    with patch.object(backend, 'direct', side_effect=record_reply):
        selected = select_tools(
            query=query, builtin_tools=BUILTIN_TOOLS, mcp_tools=mcp_tools or {},
            strategy=ToolSelectionStrategy.LLM, llm_backend=backend,
            llm_model=cfg.llm_chat_model or cfg.ollama_chat_model,
            llm_timeout_sec=timeout_sec, context_hint=context_hint,
        )
    assert isinstance(model_reply, str) and model_reply.strip(), (
        "The router returned no model response; keyword fallback is not LLM accuracy"
    )
    try:
        payload = json.loads(model_reply)
    except ValueError as error:
        raise AssertionError(f"The router response was not JSON: {model_reply}") from error
    assert isinstance(payload, dict), f"The router response was not an object: {model_reply}"
    operation, names = payload.get('requested_operation'), payload.get('tools')
    assert isinstance(operation, str) and operation.strip(), f"The router response has no operation: {model_reply}"
    assert isinstance(names, list) and all(isinstance(name, str) for name in names), f"The router response has invalid tool names: {model_reply}"
    assert not names or any(name in BUILTIN_TOOLS or name in (mcp_tools or {}) for name in names), (
        f"The router did not select an available tool: {model_reply}"
    )
    return selected, model_reply
