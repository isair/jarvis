# LLM Backend Specification

The `jarvis.llm` package owns every LLM HTTP call Jarvis makes. It exists to keep Jarvis runtime-agnostic: the same reply engine, planner, intent judge, evaluator, memory pipeline, and tools work whether the user runs Ollama, an OpenAI-compatible server (LM Studio, oMLX, llama.cpp's `llama-server`, vLLM, LocalAI), or an Anthropic-compatible server.

This spec covers the abstraction shape, the supported providers, and the two interchangeable entry-point styles (object and function).

## Goals

1. **Pluggable.** New backends drop in by subclassing `LLMBackend` and being registered in `factory.get_llm_backend`. No call sites change.
2. **Behaviour-preserving.** PR 1 changes no observable behaviour: the Ollama backend has identical wire shape, defaults, retry semantics, and error handling as the previous flat `llm.py`.
3. **Privacy-first.** Backends never send data anywhere unless the user has explicitly configured the URL. Defaults remain `127.0.0.1:11434`.

## Phases

| Phase | Scope | Status |
|-------|-------|--------|
| PR 1 | Extract `LLMBackend` ABC, `OllamaBackend`, factory, function-style helpers. Ollama-only. Zero behaviour change. | Done |
| PR 2 | Add `OpenAICompatibleBackend`, `llm_provider` / `llm_*` / `embedding_*` config keys + v2 migration, factory dispatch, `get_embedding_backend`. Foundation only — no call site uses the factory yet. | Done |
| PR 2.5 | Migrate every LLM call site (reply engine, planner, evaluator, intent judge, enrichment, memory ops, tools, embeddings) to `get_llm_backend(settings)` / `get_embedding_backend(settings)`. After this PR, picking `llm_provider: openai_compatible` works end-to-end. | Pending |
| PR 3 | Setup wizard provider page, settings UI provider group, README update. | Pending |
| PR 4 | Add `AnthropicCompatibleBackend` (optional, demand-driven). | Pending |

## Public surface

```python
from jarvis.llm import (
    LLMBackend,                  # provider-agnostic ABC
    OllamaBackend,               # implementation: Ollama
    OpenAICompatibleBackend,     # implementation: OpenAI-compatible servers
    ToolsNotSupportedError,
    get_llm_backend,             # factory: settings → chat backend
    get_embedding_backend,       # factory: settings → embedding backend
    call_llm_direct,             # function-style equivalents that take base_url
    call_llm_streaming,
    chat_with_messages,
    extract_text_from_response,
)
```

Two interchangeable styles dispatch to the same backend:

- **Object-style** — `get_llm_backend(settings).direct(...)`. Use when a settings object is in scope; future-proof against `llm_provider` being added.
- **Function-style** — `call_llm_direct(base_url, ...)`. Use when only a base URL is in scope. Constructs the right backend internally.

Both produce the same HTTP request and response handling.

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

`embed()` is part of the same backend interface so the same provider can serve both chat and embeddings when capable. The `embedding_provider` config key lets users on runtimes without embeddings (e.g. some oMLX builds) route embeddings through Ollama while keeping chat on their preferred runtime. Until call sites migrate (PR 2.5), `src/jarvis/memory/embeddings.py` keeps its own POST against `/api/embeddings` for compatibility.

## Configuration

The provider-aware fields in `Settings` (see [src/jarvis/config.py](../config.py)):

| Key | Default | Meaning |
|-----|---------|---------|
| `llm_provider` | `"ollama"` | `"ollama"` or `"openai_compatible"`. Unknown values fall back to `"ollama"`. |
| `llm_base_url` | `""` (= use `ollama_base_url`) | Active provider base URL. For Ollama, `http://127.0.0.1:11434`. For OpenAI-compatible, `http://localhost:1234/v1` (LM Studio default) or whatever your runtime exposes. |
| `llm_api_key` | `""` | Optional bearer token. Sent only when non-empty. |
| `llm_chat_model` | `""` (= use `ollama_chat_model`) | Active chat model name. |
| `embedding_provider` | `""` (= same as `llm_provider`) | `"ollama"` / `"openai_compatible"`. Override for runtimes without embeddings. |
| `embedding_base_url` | `""` (= inherit from llm config) | Override per-provider URL. |
| `embedding_api_key` | `""` (= use `llm_api_key`) | Override per-provider key. |
| `embedding_model` | `""` (= use `ollama_embed_model`) | Active embedding model name. |

The legacy `ollama_base_url` / `ollama_chat_model` / `ollama_embed_model` keys remain in `Settings` as compatibility aliases populated alongside the new fields. They retain their original meaning so call sites that have not yet migrated to the factory keep working.

### Factory dispatch

- `get_llm_backend(settings)` reads `llm_provider`, then resolves `llm_base_url` (falling back to `ollama_base_url`) and `llm_api_key`.
- `get_embedding_backend(settings)` reads `embedding_provider` (falls back to `llm_provider` when unset), resolves `embedding_base_url` (falls back per-provider: `llm_base_url` for OpenAI-compatible, `ollama_base_url` for Ollama), and `embedding_api_key` (falls back to `llm_api_key`).
- Construction is fail-soft: an unset URL becomes `_DEFAULT_OLLAMA_URL`, so `get_*_backend` never raises. Errors surface at request time, not construction time.

### v1 → v2 config migration

The migration in `_migrate_config` runs once when `_config_version < 2`:

1. If `llm_provider` is unset, default to `"ollama"`.
2. Promote `ollama_base_url` → `llm_base_url`, `ollama_chat_model` → `llm_chat_model`, `ollama_embed_model` → `embedding_model` (only when the new key is empty).
3. Bump `_config_version` to `2` and persist via `_save_json` (which restricts the file to `0o600` on POSIX so credentials are not world-readable).

Existing installs upgrade silently with no behaviour change.

## Wire-shape specifics

### Ollama (`OllamaBackend`)

- Endpoints: `POST /api/chat`, `POST /api/embeddings`, `GET /api/tags`.
- Streaming: JSON-lines (`{...}\n`).
- Tool calls: native `tools` parameter (Ollama 0.4+); arguments returned as a Python dict.
- Options: `num_ctx`, `temperature`, `think` live under a nested `options` object; `stream`, `model`, `messages`, `tools` at the root.

### OpenAI-compatible (`OpenAICompatibleBackend`)

- Endpoints: `POST /chat/completions`, `POST /embeddings`, `GET /models`.
- Streaming: Server-Sent Events. Lines start with `data:` and an empty payload terminator is `data: [DONE]`. Comment lines (`: ping`) and malformed payloads are skipped.
- Tool calls: native `tools` parameter; OpenAI returns `tool_calls[*].function.arguments` as a JSON-encoded string. The backend decodes them to a dict so the reply engine sees a single shape.
- Response normalisation: `_normalise_response` lifts `choices[0].message` to top-level `message` so callers do not branch on provider. Servers that already return Ollama-shaped responses pass through unchanged.
- Options: `temperature`, `max_tokens`, `top_p` etc. live at the payload root; `extra_options` is shallow-merged at the root rather than under a nested `options` object.
- `num_ctx` and `thinking` are accepted for ABC parity and silently ignored — there is no equivalent in the OpenAI shape (context window is configured at server load time; reasoning is a model attribute).
- Authentication: `Authorization: Bearer <api_key>` header sent only when `api_key` is non-empty.

## Function-style helpers

`call_llm_direct`, `call_llm_streaming`, and `chat_with_messages` construct a fresh `OllamaBackend(base_url)` and delegate to the matching method. Their signatures match the previous `llm.py` exactly so the existing ~10 call sites under `src/jarvis/`, `tests/`, and `evals/` import them unchanged.

`import requests` is re-exported from the package `__init__.py` so tests that patch `jarvis.llm.requests.post` keep working after the package split.

## File layout

```
src/jarvis/llm/
├── __init__.py             # public re-exports + function-style helpers
├── backend.py              # LLMBackend ABC + ToolsNotSupportedError
├── ollama.py               # OllamaBackend + extract_text_from_response
├── openai_compatible.py    # OpenAICompatibleBackend + _normalise_response
├── factory.py              # get_llm_backend(settings) + get_embedding_backend(settings)
└── llm.spec.md             # this file
```

## Failure handling

Backends fail soft for transient issues so the reply engine can degrade gracefully: timeouts and connection errors return `None` (or `[]` for `list_models`); HTTP 400 with `tools` set raises `ToolsNotSupportedError`; any other unexpected error is logged via `debug_log("...", "llm")` and returns `None`. Callers must already handle `None` because that has been the contract since the original `llm.py`.
