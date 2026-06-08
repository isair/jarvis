"""LiteLLM implementation of :class:`LLMBackend`.

Routes requests through the `litellm <https://github.com/BerriAI/litellm>`_
SDK, which provides a unified interface to 100+ LLM providers (OpenAI,
Anthropic, Google, Azure, AWS Bedrock, Ollama, Groq, Mistral, and more).
Users specify a LiteLLM model string such as ``anthropic/claude-sonnet-4-6``
or ``openai/gpt-4o``; LiteLLM handles provider-specific routing, auth
headers, and response normalisation automatically.

Response normalisation reuses the same ``choices[0].message`` lifting as
:mod:`openai_compatible` so callers see the Ollama-shaped dict they expect.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional

import litellm

from ..debug import debug_log
from .backend import LLMBackend, ToolsNotSupportedError
from .openai_compatible import _normalise_response


class LiteLLMBackend(LLMBackend):
    """:class:`LLMBackend` implementation backed by the LiteLLM SDK."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or None

    def _common_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"drop_params": True}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        return kwargs

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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        kwargs = self._common_kwargs()
        kwargs["timeout"] = timeout_sec
        if temperature is not None:
            kwargs["temperature"] = temperature

        try:
            response = litellm.completion(
                model=chat_model,
                messages=messages,
                stream=False,
                **kwargs,
            )
            content = response.choices[0].message.content
            if isinstance(content, str) and content.strip():
                return content
            debug_log("LiteLLMBackend.direct: empty content in response", "llm")
        except Exception as e:
            debug_log(f"LiteLLMBackend.direct: request failed - {type(e).__name__}", "llm")
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
        kwargs = self._common_kwargs()
        kwargs["timeout"] = timeout_sec

        try:
            response = litellm.completion(
                model=chat_model,
                messages=messages,
                stream=True,
                **kwargs,
            )
            full_response: List[str] = []
            for chunk in response:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                content = getattr(delta, "content", None)
                if isinstance(content, str) and content:
                    full_response.append(content)
                    if on_token:
                        on_token(content)

            result = "".join(full_response)
            return result if result.strip() else None
        except Exception as e:
            debug_log(f"LiteLLMBackend.streaming: request failed - {type(e).__name__}", "llm")
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
        kwargs = self._common_kwargs()
        kwargs["timeout"] = timeout_sec

        if extra_options and isinstance(extra_options, dict):
            for key, value in extra_options.items():
                if key in {"keep_alive", "num_ctx", "num_predict", "think"}:
                    continue
                if key == "options" and isinstance(value, dict):
                    for inner_key, inner_value in value.items():
                        if inner_key in {"num_ctx", "num_predict"}:
                            continue
                        kwargs[inner_key] = inner_value
                else:
                    kwargs[key] = value
        if tools and isinstance(tools, list) and len(tools) > 0:
            kwargs["tools"] = tools

        try:
            response = litellm.completion(
                model=chat_model,
                messages=messages,
                stream=False,
                **kwargs,
            )
            data = response.model_dump()
            if isinstance(data, dict):
                return _normalise_response(data)
        except litellm.exceptions.BadRequestError:
            if tools:
                raise ToolsNotSupportedError(
                    f"Model {chat_model!r} rejected the tools parameter"
                )
            print(f"  ❌ LLM bad request", flush=True)
            return None
        except Exception as e:
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
        kwargs = self._common_kwargs()
        kwargs["timeout"] = timeout_sec
        try:
            response = litellm.embedding(model=model, input=[text], **kwargs)
            data = response.model_dump()
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
            models = litellm.model_list or []
            return [m if isinstance(m, str) else str(m) for m in models]
        except Exception:
            return []
