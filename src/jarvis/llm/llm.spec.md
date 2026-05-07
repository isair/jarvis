# LLM Backend Specification

The `jarvis.llm` package owns every LLM HTTP call Jarvis makes. It exists to keep Jarvis runtime-agnostic: the same reply engine, planner, intent judge, evaluator, memory pipeline, and tools work whether the user runs Ollama, an OpenAI-compatible server (LM Studio, oMLX, llama.cpp's `llama-server`, vLLM, LocalAI), or an Anthropic-compatible server.

This spec covers the abstraction shape, the supported providers, and the legacy free functions that bridge the existing call sites.

## Goals

1. **Pluggable.** New backends drop in by subclassing `LLMBackend` and being registered in `factory.get_llm_backend`. No call sites change.
2. **Behaviour-preserving.** PR 1 changes no observable behaviour: the Ollama backend has identical wire shape, defaults, retry semantics, and error handling as the previous flat `llm.py`.
3. **Privacy-first.** Backends never send data anywhere unless the user has explicitly configured the URL. Defaults remain `127.0.0.1:11434`.

## Phases

| Phase | Scope | Status |
|-------|-------|--------|
| PR 1 | Extract `LLMBackend` ABC, `OllamaBackend`, factory, re-exports. Ollama-only. Zero behaviour change. | Done |
| PR 2 | Add `OpenAICompatibleBackend`, `llm_provider` config key + migration, embedding routing override. | Pending |
| PR 3 | Setup wizard provider page, settings UI provider group, README update. | Pending |
| PR 4 | Add `AnthropicCompatibleBackend` (optional, demand-driven). | Pending |

## Public surface

```python
from jarvis.llm import (
    LLMBackend,            # provider-agnostic ABC
    OllamaBackend,         # only implementation in PR 1
    ToolsNotSupportedError,
    get_llm_backend,       # factory: settings → backend
    extract_text_from_response,  # response-shape helper, used by reply engine
)
```

New code uses `get_llm_backend(settings)`; legacy callers continue to use the free functions `call_llm_direct`, `call_llm_streaming`, `chat_with_messages` (re-exported for backwards compatibility, both paths route through `OllamaBackend` today).

## `LLMBackend` interface

Each method maps onto a well-defined wire shape. Implementations translate the generic kwargs into their native protocol, drop or map ones the runtime does not support, and normalise responses.

| Method | Returns | Contract |
|--------|---------|----------|
| `direct(model, system, user, *, timeout_sec, thinking, num_ctx, temperature)` | `Optional[str]` | Single-shot system+user. Returns assistant text, or `None` on timeout / error / empty content. |
| `streaming(model, system, user, *, on_token, timeout_sec, thinking)` | `Optional[str]` | Streams tokens via `on_token`; returns the concatenated full text or `None` if no content was produced. |
| `chat(model, messages, *, timeout_sec, extra_options, tools, thinking)` | `Optional[Dict]` | Arbitrary messages array. Returns the raw response dict so callers (today: the reply engine) can inspect `content` and `tool_calls`. Raises `ToolsNotSupportedError` when the model rejects native tools. |
| `embed(text, model, *, timeout_sec)` | `Optional[List[float]]` | Vector embedding. Returns `None` on error or when the runtime does not expose embeddings. |
| `list_models(*, timeout_sec)` | `List[str]` | Names of locally available models. Returns `[]` on error. |

`direct()` and `streaming()` are convenience methods over `chat()`: they construct the `[system, user]` messages array internally so callers running classification-shaped passes (planner, intent judge, evaluator, enrichment extractor) do not have to. `chat()` is the low-level primitive for arbitrary message arrays — multi-turn dialogue, native tool calls, and anything that needs custom roles. Use the convenience methods when you have a single system + single user; reach for `chat()` whenever the message array is non-trivial.

### Tool calling

The `tools` parameter accepts the OpenAI-compatible JSON-schema format produced by `jarvis.tools.registry.generate_tools_json_schema()`. Ollama 0.4+ adopts that exact format, so no translation layer is needed for the Ollama backend; future OpenAI-compatible and Anthropic-compatible backends translate inside their `chat()` methods so the reply engine sees a single shape.

When a model rejects the `tools` parameter (Ollama returns HTTP 400 in that case), the backend raises `ToolsNotSupportedError`. The reply engine catches it and falls back to text-based tool calling for the rest of the session — the existing small-model fallback path in `src/jarvis/reply/engine.py`.

### Streaming

Each backend parses its own stream format internally (Ollama JSONL, OpenAI SSE, Anthropic SSE event blocks). The public `on_token(str)` contract is identical across backends.

### Embeddings

`embed()` is part of the same backend interface so the same provider can serve both chat and embeddings when capable. PR 2 introduces an `embedding_provider` setting so users on backends without embeddings (e.g. some oMLX builds) can route only embeddings through Ollama while keeping chat on their preferred runtime. Until that lands, `src/jarvis/memory/embeddings.py` keeps its own POST against `/api/embeddings` for compatibility.

## Backwards compatibility

The legacy free functions are thin wrappers that construct a fresh `OllamaBackend(base_url)` and delegate. Their signatures match the previous `llm.py` exactly so the existing ~10 call sites under `src/jarvis/`, `tests/`, and `evals/` continue to import them unchanged.

`import requests` is re-exported from the package `__init__.py` so existing tests that patch `jarvis.llm.requests.post` keep working after the package split.

## File layout

```
src/jarvis/llm/
├── __init__.py        # re-exports + legacy free functions
├── backend.py         # LLMBackend ABC + ToolsNotSupportedError
├── ollama.py          # OllamaBackend + extract_text_from_response
├── factory.py         # get_llm_backend(settings)
└── llm.spec.md        # this file
```

## Failure handling

Backends fail soft for transient issues so the reply engine can degrade gracefully: timeouts and connection errors return `None` (or `[]` for `list_models`); HTTP 400 with `tools` set raises `ToolsNotSupportedError`; any other unexpected error is logged via `debug_log("...", "llm")` and returns `None`. Callers must already handle `None` because that has been the contract since the original `llm.py`.
