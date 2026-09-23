"""OpenAI-compatible implementation of :class:`LLMBackend`.

Targets any local server that exposes the OpenAI Chat Completions
shape: LM Studio, oMLX, llama.cpp's ``llama-server``, vLLM, LocalAI,
and similar. The wire shape differs from Ollama in three important
ways, all hidden inside this module so callers see one response
shape:

1. **Endpoints** are ``/chat/completions`` and ``/embeddings`` rather
   than ``/api/chat`` and ``/api/embeddings``. Model listing is at
   ``/models`` rather than ``/api/tags``.
2. **Streaming uses Server-Sent Events** (``data: {...}\\n\\n`` with a
   ``data: [DONE]`` terminator) instead of Ollama's JSON-lines.
3. **Tool-call arguments arrive as a JSON-encoded string**
   (``"{\\"x\\": 1}"``) rather than a dict; the reply engine expects
   a dict, so :meth:`chat` decodes them. The same method also lifts
   ``choices[0].message`` to top-level ``message`` so the engine's
   existing parsing path works without branching on provider.

The error handling and ``ToolsNotSupportedError`` semantics mirror
:class:`OllamaBackend` so callers get a single contract regardless of
which backend is active.
"""
from __future__ import annotations

import functools
import ipaddress
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import json
import requests

from ..debug import debug_log
from .backend import LLMBackend, ToolsNotSupportedError, strip_nonstandard_message_fields


_RETRYABLE_GENERATION_STATUS = frozenset({429, 502, 503, 504})


def _serialised_request(method):
    """Run one request at a time against a single-slot llama.cpp server."""

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        queued_at = time.monotonic()
        with self._request_gate:
            wait_ms = (time.monotonic() - queued_at) * 1000.0
            if wait_ms >= 25.0:
                debug_log(
                    f"LLM single-slot gate: {method.__name__} waited {wait_ms:.1f} ms",
                    "llm",
                )
            return method(self, *args, **kwargs)

    return wrapped


@dataclass
class ServerCapabilities:
    """What an OpenAI-compatible server can actually do, probed with real
    requests. ``reachable`` is False when the server did not respond at all
    (wrong URL, server down); the per-feature flags are only meaningful when
    ``reachable`` is True. ``models`` is the advertised model list."""

    reachable: bool = False
    chat: bool = False
    tools: bool = False
    embeddings: bool = False
    models: List[str] = field(default_factory=list)


def _normalise_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Lift OpenAI's ``choices[0].message`` to top-level ``message``
    (matching Ollama's shape) and JSON-decode any tool-call arguments.

    If the server already returns Ollama's shape (some hybrid servers
    expose both endpoints), the response is passed through unchanged.

    Scope: this helper is OpenAI-shape-specific. Other providers
    (Anthropic, etc.) need their own normaliser inside their own
    backend module — Anthropic's content-block + ``tool_use`` shape
    diverges enough that sharing one normaliser would be more
    confusing than useful. Keep one normaliser per backend.
    """
    if "message" in data and isinstance(data["message"], dict):
        return data

    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        msg = choices[0].get("message")
        if isinstance(msg, dict):
            normalised = dict(data)
            decoded_msg = dict(msg)
            tool_calls = decoded_msg.get("tool_calls")
            if isinstance(tool_calls, list):
                decoded_calls: List[Dict[str, Any]] = []
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        decoded_calls.append(tc)
                        continue
                    decoded_tc = dict(tc)
                    func = decoded_tc.get("function")
                    if isinstance(func, dict):
                        decoded_func = dict(func)
                        args = decoded_func.get("arguments")
                        if isinstance(args, str):
                            try:
                                decoded_func["arguments"] = json.loads(args)
                            except (json.JSONDecodeError, ValueError):
                                # Leave as-is; the engine's content-mode
                                # parser may still recover something.
                                pass
                        decoded_tc["function"] = decoded_func
                    decoded_calls.append(decoded_tc)
                decoded_msg["tool_calls"] = decoded_calls
            normalised["message"] = decoded_msg
            return normalised

    return data


class OpenAICompatibleBackend(LLMBackend):
    """:class:`LLMBackend` implementation for OpenAI-compatible servers."""

    def __init__(self, base_url: str, api_key: Optional[str] = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or None
        self._request_gate = threading.RLock()
        self._session = requests.Session()
        self._models_cache: List[str] = []
        self._models_cache_at = 0.0
        self._configured_model_hint = ""

    @property
    def base_url(self) -> str:
        return self._base_url

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _is_local_server(self) -> bool:
        """Whether generation is served by this PC or the local network."""
        host = (urlparse(self._base_url).hostname or "").strip().lower()
        if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
            return True
        try:
            address = ipaddress.ip_address(host)
            return bool(address.is_loopback or address.is_private or address.is_link_local)
        except ValueError:
            return False

    def _generation_timeout(self, timeout_sec: float):
        """Use a connect deadline, but no read deadline, for local inference.

        ``requests`` treats a float as both a connect and socket-read timeout.
        A single-slot llama.cpp server can spend that interval queued or in
        prefill before returning a byte, which is not a failed generation.
        """
        connect_timeout = min(10.0, max(0.5, float(timeout_sec or 10.0)))
        if self._is_local_server():
            return (connect_timeout, None)
        return (connect_timeout, max(connect_timeout, float(timeout_sec or 30.0)))

    def _post_generation(
        self,
        payload: Dict[str, Any],
        *,
        timeout_sec: float,
        stream: bool = False,
    ) -> requests.Response:
        """POST one generation, absorbing transient server-load states."""
        attempts = 20 if self._is_local_server() else 3
        response: Optional[requests.Response] = None
        for attempt in range(attempts):
            response = requests.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self._generation_timeout(timeout_sec),
                stream=stream,
            )
            status = response.status_code
            if status not in _RETRYABLE_GENERATION_STATUS:
                return response
            retry_after = response.headers.get("Retry-After")
            response.close()
            try:
                delay = float(retry_after) if retry_after else min(2.0, 0.2 * (2 ** attempt))
            except (TypeError, ValueError):
                delay = min(2.0, 0.2 * (2 ** attempt))
            debug_log(
                f"LLM server busy ({status}); retry {attempt + 1}/{attempts} "
                f"in {delay:.2f}s",
                "llm",
            )
            time.sleep(max(0.0, delay))
        assert response is not None
        return response

    def _resolved_model(self, model: str, timeout_sec: float = 3.0) -> str:
        """The id the server actually serves, from ``GET /models``.

        llama.cpp and LM Studio advertise the one loaded model by its
        instance id (``qwen/qwen3-8b`` style), while ``llm_chat_model`` may
        hold the raw GGUF path. The first advertised id is the authority
        for ``/chat/completions`` and ``/embeddings``; an exact match wins
        first. If the server lists nothing, the configured name is used.
        """
        want = str(model or "")
        # A daemon restart can change the configured GGUF while the desktop
        # process and backend singleton survive. Force one catalogue refresh
        # when the configured hint changes; reuse it inside an agent turn.
        if want != self._configured_model_hint:
            self._configured_model_hint = want
            self._models_cache_at = 0.0
        names = self.list_models(timeout_sec=timeout_sec)
        if not names:
            return want
        if want in names:
            return want
        return names[0]

    # ── chat ───────────────────────────────────────────────────────────

    @_serialised_request
    def direct(
        self,
        chat_model: str,
        system_prompt: str,
        user_content: str,
        timeout_sec: float = 10.0,
        thinking: bool = False,
        num_ctx: int = 4096,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        # ``num_ctx`` and ``thinking`` have no equivalent in the OpenAI
        # shape; servers that need a fixed context window configure it
        # at load time, and reasoning is a model attribute rather than
        # a request flag. Both are accepted for signature parity with
        # OllamaBackend and silently ignored here.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        payload: Dict[str, Any] = {
            "model": self._resolved_model(chat_model),
            "messages": messages,
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            with self._post_generation(payload, timeout_sec=timeout_sec) as resp:
                resp.raise_for_status()
                data = resp.json()

            normalised = _normalise_response(data) if isinstance(data, dict) else None
            if normalised:
                msg = normalised.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        return content
                    # Reasoning-first servers (Qwen3/Gemma builds) often leave
                    # ``content`` empty and put the answer in ``reasoning_content``.
                    reasoning = msg.get("reasoning_content")
                    if isinstance(reasoning, str) and reasoning.strip():
                        return reasoning
                debug_log(
                    f"OpenAICompatibleBackend.direct: empty content from response keys={list(data.keys())}",
                    "llm",
                )
        except requests.exceptions.ConnectTimeout:
            debug_log("OpenAICompatibleBackend.direct: connection timed out", "llm")
            return None
        except Exception as e:
            # The exception string can embed the full URL (and any query-string
            # credentials); log only the class so nothing sensitive leaks.
            debug_log(f"OpenAICompatibleBackend.direct: request failed ({type(e).__name__})", "llm")
            return None

        return None

    @_serialised_request
    def streaming(
        self,
        chat_model: str,
        system_prompt: str,
        user_content: str,
        on_token: Optional[Callable[[str], None]] = None,
        timeout_sec: float = 30.0,
        thinking: bool = False,
    ) -> Optional[str]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        payload: Dict[str, Any] = {
            "model": self._resolved_model(chat_model),
            "messages": messages,
            "stream": True,
        }

        try:
            with self._post_generation(
                payload, timeout_sec=timeout_sec, stream=True
            ) as resp:
                resp.raise_for_status()

                full_response: List[str] = []
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else raw
                    if not line.startswith("data:"):
                        # SSE comments (``: ping``) and unrelated lines.
                        continue
                    payload_str = line[len("data:"):].strip()
                    if payload_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") if isinstance(chunk, dict) else None
                    if not isinstance(choices, list) or not choices:
                        continue
                    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
                    if not isinstance(delta, dict):
                        continue
                    # Reasoning-first servers (Qwen3-style llama.cpp builds)
                    # stream ``delta.reasoning_content`` before ``content``
                    # is ever non-null; both are text of this turn.
                    content = delta.get("content")
                    if not isinstance(content, str) or not content:
                        content = delta.get("reasoning_content")
                    if isinstance(content, str) and content:
                        full_response.append(content)
                        if on_token:
                            on_token(content)

                result = "".join(full_response)
                return result if result.strip() else None
        except requests.exceptions.ConnectTimeout:
            return None
        except Exception:
            return None

    @staticmethod
    def _encode_tool_call_arguments(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """JSON-encode ``tool_calls[*].function.arguments`` in assistant messages.

        The OpenAI API spec requires ``arguments`` to be a JSON string, but
        ``normalise_openai_response`` decodes it to a dict for internal use.
        When that assistant message is sent back to the server on the next
        turn, we must re-encode it.
        """
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            tc_list = msg.get("tool_calls")
            if not isinstance(tc_list, list):
                continue
            for tc in tc_list:
                func = tc.get("function")
                if not isinstance(func, dict):
                    continue
                args = func.get("arguments")
                if isinstance(args, dict):
                    func["arguments"] = json.dumps(args)
        return messages

    @_serialised_request
    def chat(
        self,
        chat_model: str,
        messages: List[Dict[str, Any]],
        timeout_sec: float = 30.0,
        extra_options: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        thinking: bool = False,
    ) -> Optional[Dict[str, Any]]:
        sanitised = strip_nonstandard_message_fields(messages)
        sanitised = self._encode_tool_call_arguments(sanitised)
        payload: Dict[str, Any] = {
            "model": self._resolved_model(chat_model),
            "messages": sanitised,
            "stream": False,
        }
        if extra_options and isinstance(extra_options, dict):
            # ``temperature``, ``max_tokens``, ``top_p`` etc. live at the
            # payload root in the OpenAI shape, not under an ``options``
            # nest. Ollama-only knobs (``keep_alive``, ``num_ctx``,
            # ``num_predict``, ``think``) are silently dropped — they have
            # no equivalent in the OpenAI shape and would 400 against most
            # servers. Sampling fields nested under ``options`` are lifted
            # to the payload root.
            for key, value in extra_options.items():
                if key in {"keep_alive", "num_ctx", "num_predict", "think"}:
                    continue
                if key == "options" and isinstance(value, dict):
                    for inner_key, inner_value in value.items():
                        if inner_key in {"num_ctx", "num_predict"}:
                            continue
                        payload[inner_key] = inner_value
                else:
                    payload[key] = value
        if tools and isinstance(tools, list) and len(tools) > 0:
            payload["tools"] = tools

        try:
            with self._post_generation(payload, timeout_sec=timeout_sec) as resp:
                resp.raise_for_status()
                data = resp.json()
            if isinstance(data, dict):
                return _normalise_response(data)
        except requests.exceptions.ConnectTimeout:
            print("  ❌ LLM server connection timed out", flush=True)
            return None
        except requests.exceptions.ConnectionError:
            # ConnectionError messages embed the configured URL via the
            # underlying urllib3 exception, which can leak account-bearing
            # query strings to stdout. Print only the failure mode and
            # bubble the exception so callers (e.g. the intent judge) can
            # distinguish "server unreachable" from a transient HTTP error.
            print("  ❌ LLM connection error", flush=True)
            raise
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 400 and tools:
                raise ToolsNotSupportedError(
                    f"Model {chat_model!r} returned HTTP 400 — native tools API not supported"
                )
            # ``str(e)`` includes "for url: <full URL>" — keep the status code
            # for diagnosis and drop the URL.
            status = e.response.status_code if e.response is not None else "?"
            print(f"  ❌ LLM HTTP error (status {status})", flush=True)
            return None
        except Exception as e:
            # Generic exception messages can carry whatever the caller embedded
            # (URLs, tokens). Print only the exception class so the user knows
            # *something* failed without leaking what.
            print(f"  ❌ LLM error ({type(e).__name__})", flush=True)
            return None

        return None

    # ── embeddings & discovery ────────────────────────────────────────

    @_serialised_request
    def embed(
        self,
        text: str,
        model: str,
        timeout_sec: float = 15.0,
    ) -> Optional[List[float]]:
        try:
            resp = requests.post(
                f"{self._base_url}/embeddings",
                json={"model": model, "input": text},
                headers=self._headers(),
                timeout=timeout_sec,
            )
            status = getattr(resp, "status_code", None)
            if isinstance(status, int) and status >= 400:
                # Some OpenAI-shaped servers (e.g. the native OpenVINO NPU
                # retrieval service) only accept `input` as a list.
                resp = requests.post(
                    f"{self._base_url}/embeddings",
                    json={"model": model, "input": [text]},
                    headers=self._headers(),
                    timeout=timeout_sec,
                )
            resp.raise_for_status()
            data = resp.json()
            arr = data.get("data") if isinstance(data, dict) else None
            if isinstance(arr, list) and arr and isinstance(arr[0], dict):
                vec = arr[0].get("embedding")
                if isinstance(vec, list):
                    return [float(x) for x in vec]
        except Exception:
            return None
        return None

    @_serialised_request
    def list_models(self, timeout_sec: float = 5.0) -> List[str]:
        now = time.monotonic()
        if self._models_cache and now - self._models_cache_at < 30.0:
            return list(self._models_cache)
        try:
            resp = requests.get(
                f"{self._base_url}/models",
                headers=self._headers(),
                timeout=timeout_sec,
            )
            resp.raise_for_status()
            data = resp.json()
            arr = data.get("data", []) if isinstance(data, dict) else []
            names: List[str] = []
            for m in arr:
                if isinstance(m, dict):
                    name = m.get("id")
                    if isinstance(name, str) and name:
                        names.append(name)
            self._models_cache = names
            self._models_cache_at = now
            return list(names)
        except Exception:
            return []

    def _gpu_layers_offloaded(self, model: str):
        """Number of layers the server put on the GPU for ``model``.

        Returns ``int`` >= 0 for a real value and ``None`` for no hint at all.
        ``0`` is a real ``cpu_confirmed``, ``None`` is ``accelerator_unknown``;
        there is no ``-1`` sentinel.
        """
        # 1. LM Studio's native shape, most reliable for ``n_gpu_layers``.
        try:
            resp = requests.get(
                f"{self._base_url}/api/v0/models",
                headers=self._headers(),
                timeout=2.0,
            )
            resp.raise_for_status()
            for entry in (resp.json().get("data") or []):
                if not isinstance(entry, dict) or str(entry.get("id", "")) != str(model):
                    continue
                for source in (
                    (entry.get("meta") or {}).get("n_gpu_layers"),
                    entry.get("n_gpu_layers"),
                    (entry.get("loaded_instances") or [{}])[0].get("n_gpu_layers")
                    if entry.get("loaded_instances")
                    else None,
                ):
                    if isinstance(source, int):
                        return source
                break
        except Exception:
            pass
        # 2. The plain OpenAI-compatible shape of the same number.
        try:
            resp = requests.get(
                f"{self._base_url}/models",
                headers=self._headers(),
                timeout=2.0,
            )
            resp.raise_for_status()
            for entry in (resp.json().get("data") or []):
                if not isinstance(entry, dict) or str(entry.get("id", "")) != str(model):
                    continue
                meta = entry.get("meta") or {}
                for source in (
                    meta.get("n_gpu_layers"),
                    entry.get("n_gpu_layers"),
                    (entry.get("loaded_instances") or [{}])[0].get("n_gpu_layers")
                    if entry.get("loaded_instances")
                    else None,
                ):
                    if isinstance(source, int):
                        return source
                break
        except Exception:
            pass
        return None

    @_serialised_request
    def warm_up(
        self,
        model: str,
        timeout_sec: float = 60.0,
        keep_alive: str = "30m",
    ) -> bool:
        """Warm up the model by sending a minimal inference request.

        Phase 1 (reachability check): calls ``GET /models`` to confirm
        the server is up and has models loaded. Fast (capped at 25 % of
        the budget, max 5 s).

        Phase 2 (model loading): sends a single-token chat completion
        (``max_tokens=1``) so the runtime actually loads the model into
        memory. Without this, an OpenAI-compatible server may leave the
        model cold until the first real request, incurring latency on the
        user's first query. This mirrors what ``OllamaBackend.warm_up()``
        does.

        ``keep_alive`` is accepted for signature parity with
        ``OllamaBackend.warm_up`` but ignored: OpenAI-compatible servers
        manage model residency at server load time and have no per-call
        keep-alive knob.

        Best-effort: errors are swallowed; ``False`` is returned when the
        server is unreachable, the model name is missing, or the inference
        request fails, so the listener can warn the user early."""
        self.last_warmup_metrics = {
            "model": str(model or ""),
            "ok": False,
            "load_time": 0.0,
            "first_token_latency": None,
            "completion_latency": 0.0,
            "tokens_per_s": 0.0,
            "backend": type(self).__name__,
            "gpu_layers": None,
            "accelerator": "accelerator_unknown",
        }
        if not self._base_url or not model:
            return False

        # Phase 1: reachability probe (fast).
        list_to = min(max(timeout_sec * 0.25, 1.0), 5.0)
        if not self.list_models(timeout_sec=list_to):
            return False

        # Phase 2: minimal inference to force model loading.
        remaining = max(0.1, timeout_sec - list_to)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "stream": False,
        }
        started = time.time()
        try:
            with requests.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=remaining,
            ) as resp:
                ok = bool(resp.ok)
        except Exception:
            ok = False
        finished = max(0.0, time.time() - started)
        self.last_warmup_metrics = {
            "model": str(model),
            "ok": ok,
            "load_time": round(finished, 4),
            "first_token_latency": None,
            "completion_latency": round(finished, 4),
            "tokens_per_s": 0.0,
            "backend": type(self).__name__,
            "gpu_layers": None,
            "accelerator": "accelerator_unknown",
        }
        return ok

    @_serialised_request
    def check_capabilities(
        self,
        chat_model: str,
        embed_model: Optional[str] = None,
        timeout_sec: float = 8.0,
    ) -> ServerCapabilities:
        """Probe what the server can actually do with real requests: list its
        models, send a tiny chat completion, try a trivial tool call, and ask
        for an embedding. Returns raw booleans (formatting is the caller's
        job). Never raises — every failure mode collapses to a False flag so
        the setup wizard and startup check can report honestly.

        ``chat`` covers both a plain reply and a tool-call-only reply (an empty
        ``content`` with ``tool_calls`` still proves the chat endpoint works)."""
        caps = ServerCapabilities(models=self.list_models(timeout_sec=timeout_sec))
        if caps.models:
            caps.reachable = True

        # Cap generation: we only need to know the endpoint answers, so a short
        # reply keeps the probe fast on large models and avoids a long
        # generation tripping the timeout and reporting a false "chat broken".
        probe = [{"role": "user", "content": "ping"}]
        probe_opts = {"max_tokens": 16}
        try:
            resp = self.chat(chat_model, probe, timeout_sec=timeout_sec, extra_options=probe_opts)
            if isinstance(resp, dict):
                caps.reachable = True
                msg = resp.get("message")
                msg = msg if isinstance(msg, dict) else {}
                caps.chat = bool((msg.get("content") or "").strip()) or bool(msg.get("tool_calls"))
        except requests.exceptions.ConnectionError:
            # Server unreachable — nothing else can succeed either.
            caps.reachable = False
            return caps
        except Exception:
            pass

        if caps.chat:
            trivial_tool = [{
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "A no-op used to probe tool support.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }]
            try:
                tool_resp = self.chat(chat_model, probe, tools=trivial_tool,
                                       timeout_sec=timeout_sec, extra_options=probe_opts)
                caps.tools = isinstance(tool_resp, dict)
            except ToolsNotSupportedError:
                caps.tools = False
            except Exception:
                caps.tools = False

        em = (embed_model or "").strip() or chat_model
        if self.embed("ping", em, timeout_sec=timeout_sec):
            caps.embeddings = True
            caps.reachable = True

        return caps
