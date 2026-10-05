"""Selected-backend routing evaluations retain the actual model answer."""
import re
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


def route_tools(cfg, query, *, timeout_sec=15.0, context_hint=None):
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
            query=query, builtin_tools=BUILTIN_TOOLS, mcp_tools={},
            strategy=ToolSelectionStrategy.LLM, llm_backend=backend,
            llm_model=cfg.llm_chat_model or cfg.ollama_chat_model,
            llm_timeout_sec=timeout_sec, context_hint=context_hint,
        )
    assert isinstance(model_reply, str) and model_reply.strip(), (
        "The router returned no model response; keyword fallback is not LLM accuracy"
    )
    assert model_reply.strip().lower() == 'none' or any(
        re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", model_reply)
        for name in BUILTIN_TOOLS if name != 'stop'
    ), f"The router response did not select any available tool: {model_reply}"
    return selected, model_reply
