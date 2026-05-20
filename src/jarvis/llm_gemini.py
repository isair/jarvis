"""Hosted Google Gemini chat for the main reply loop (text-based tool protocol)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests

from jarvis.debug import debug_log

_GEMINI_GENERATE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


def _messages_to_gemini_payload(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Map Jarvis/Ollama-style messages to Gemini generateContent JSON."""
    system_chunks: List[str] = []
    contents: List[Dict[str, Any]] = []

    for msg in messages:
        role = str(msg.get("role") or "")
        content = msg.get("content")
        text = content if isinstance(content, str) else ""
        if role == "system":
            if text.strip():
                system_chunks.append(text)
            continue
        if role == "user":
            contents.append({"role": "user", "parts": [{"text": text or " "}]})
            continue
        if role == "assistant":
            contents.append({"role": "model", "parts": [{"text": text or " "}]})
            continue
        if role == "tool":
            tn = str(msg.get("tool_name") or "tool")
            wrapped = f"[Tool result: {tn}]\n{text}"
            contents.append({"role": "user", "parts": [{"text": wrapped}]})
            continue
        contents.append({"role": "user", "parts": [{"text": text or "(message)"}]})

    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 8192,
        },
    }
    if system_chunks:
        payload["systemInstruction"] = {
            "parts": [{"text": "\n\n".join(system_chunks)}],
        }
    return payload


def _extract_gemini_text(data: Dict[str, Any]) -> Optional[str]:
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    first = candidates[0]
    if not isinstance(first, dict):
        return None
    ccontent = first.get("content")
    if not isinstance(ccontent, dict):
        return None
    parts = ccontent.get("parts")
    if not isinstance(parts, list):
        return None
    texts: List[str] = []
    for p in parts:
        if isinstance(p, dict):
            t = p.get("text")
            if isinstance(t, str) and t:
                texts.append(t)
    if not texts:
        return None
    return "".join(texts)


def gemini_chat_with_messages(
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    timeout_sec: float = 120.0,
) -> Optional[Dict[str, Any]]:
    """
    Send a conversation to Gemini and return an Ollama-shaped response dict::

        {"message": {"content": "<assistant text>"}}

    so ``extract_text_from_response`` and text-based tool-call parsing work.
    """
    key = (api_key or "").strip()
    if not key:
        debug_log("gemini_chat_with_messages: missing API key", "llm")
        print(
            "  Cloud LLM: set GEMINI_API_KEY (or gemini_api_key in config) for Gemini.",
            flush=True,
        )
        return None

    mid = (model or "").strip().replace("/", "-") or "gemini-2.0-flash"
    url = _GEMINI_GENERATE_URL.format(model=mid)
    params = {"key": key}
    body = _messages_to_gemini_payload(messages)

    try:
        with requests.post(
            url,
            params=params,
            json=body,
            timeout=timeout_sec,
        ) as resp:
            raw = resp.text
            if resp.status_code != 200:
                debug_log(
                    f"gemini generateContent HTTP {resp.status_code} (body len={len(raw)})",
                    "llm",
                )
                print(
                    f"  Cloud LLM error: Gemini returned HTTP {resp.status_code}. "
                    "Check model id and API key.",
                    flush=True,
                )
                return None
            data = json.loads(raw)
    except requests.exceptions.Timeout:
        debug_log(f"gemini generateContent timeout after {timeout_sec}s", "llm")
        print("  Cloud LLM error: Gemini request timed out.", flush=True)
        return None
    except Exception as exc:
        debug_log(f"gemini generateContent failed: {exc}", "llm")
        print(f"  Cloud LLM error: {exc}", flush=True)
        return None

    if not isinstance(data, dict):
        return None

    block = data.get("promptFeedback")
    if isinstance(block, dict):
        br = block.get("blockReason")
        if br:
            debug_log(f"gemini prompt blocked: {br}", "llm")
            print(
                f"  Cloud LLM: Gemini blocked the prompt ({br}).",
                flush=True,
            )
            return None

    text = _extract_gemini_text(data)
    if not text:
        debug_log(
            f"gemini empty text; keys={list(data.keys())}",
            "llm",
        )
        return None

    return {"message": {"content": text}}
