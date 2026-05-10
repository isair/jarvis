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
from typing import Any, Callable, Dict, List, Optional

import json
import requests

from ..debug import debug_log
from .backend import LLMBackend, ToolsNotSupportedError


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

    @property
    def base_url(self) -> str:
        return self._base_url

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    # ── chat ───────────────────────────────────────────────────────────

    def direct(
        self,
        chat_model: str,
        system_prompt: str,
        user_content: str,
        timeout_sec: float = 10.0,
        thinking: bool = False,
        num_ctx: int = 4096,
        temperature: Optional[float] = None,
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
            "model": chat_model,
            "messages": messages,
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        try:
            with requests.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=timeout_sec,
            ) as resp:
                resp.raise_for_status()
                data = resp.json()

            normalised = _normalise_response(data) if isinstance(data, dict) else None
            if normalised:
                msg = normalised.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        return content
                debug_log(
                    f"OpenAICompatibleBackend.direct: empty content from response keys={list(data.keys())}",
                    "llm",
                )
        except requests.exceptions.Timeout:
            debug_log(f"OpenAICompatibleBackend.direct: timeout after {timeout_sec}s", "llm")
            return None
        except Exception as e:
            debug_log(f"OpenAICompatibleBackend.direct: request failed — {e}", "llm")
            return None

        return None

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
            "model": chat_model,
            "messages": messages,
            "stream": True,
        }

        try:
            with requests.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=timeout_sec,
                stream=True,
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
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        full_response.append(content)
                        if on_token:
                            on_token(content)

                result = "".join(full_response)
                return result if result.strip() else None
        except requests.exceptions.Timeout:
            return None
        except Exception:
            return None

    def chat(
        self,
        chat_model: str,
        messages: List[Dict[str, Any]],
        timeout_sec: float = 30.0,
        extra_options: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        thinking: bool = False,
    ) -> Optional[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "model": chat_model,
            "messages": messages,
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
            with requests.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=timeout_sec,
            ) as resp:
                resp.raise_for_status()
                data = resp.json()
            if isinstance(data, dict):
                return _normalise_response(data)
        except requests.exceptions.Timeout:
            print("  ⏱️ LLM request timed out", flush=True)
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

    def list_models(self, timeout_sec: float = 5.0) -> List[str]:
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
            return names
        except Exception:
            return []
