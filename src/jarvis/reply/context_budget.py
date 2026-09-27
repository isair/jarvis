"""Bound reply messages while preserving complete tool-call evidence."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from math import ceil
from typing import Callable


@dataclass
class ContextBudgetExceeded(ValueError):
    """Required context or preserved evidence cannot fit the requested budget."""

    required_tokens: int
    budget_tokens: int

    def __str__(self) -> str:
        return f"required context needs {self.required_tokens} estimated tokens; budget is {self.budget_tokens}"


class InvalidToolHistory(ValueError):
    """A tool result has no complete matching request group."""


def estimate_message_tokens(message: dict) -> int:
    """Estimate tokens across scripts with room for role framing.

    English text typically packs several letters per token while CJK and
    emoji text packs far fewer characters. This provider-neutral approximation
    is replaceable by a model-specific tokeniser at the call site.
    """
    encoded = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
    ascii_words = 0
    ascii_symbols = 0
    multibyte = 0
    for character in encoded:
        if ord(character) < 128:
            if character.isalnum() or character.isspace():
                ascii_words += 1
            else:
                ascii_symbols += 1
        else:
            multibyte += ceil(len(character.encode("utf-8")) / 3)
    return ceil(ascii_words / 4 + ascii_symbols / 2 + multibyte) + 8


def _is_genuine_user(message: dict) -> bool:
    return message.get("role") == "user" and not message.get("tool_name")


def _parse_units(messages: list[dict]) -> list[list[dict]]:
    units: list[list[dict]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        calls = message.get("tool_calls") if role == "assistant" else None
        if role == "assistant" and calls is not None and not isinstance(calls, list):
            raise InvalidToolHistory("tool_calls must be a list")
        if calls:
            count = len(calls)
            results = messages[index + 1:index + 1 + count]
            if len(results) != count:
                raise InvalidToolHistory("tool-call group has missing results")
            if any(
                not isinstance(call, dict)
                or not isinstance(call.get("function"), dict)
                or not isinstance(call["function"].get("name"), str)
                or not call["function"]["name"]
                for call in calls
            ):
                raise InvalidToolHistory("tool-call group has malformed requests")
            ids = [call.get("id") for call in calls]
            names = [call["function"]["name"] for call in calls]
            if all(item.get("role") == "tool" for item in results):
                result_ids = [item.get("tool_call_id") for item in results]
                if (
                    any(not isinstance(item, str) or not item for item in ids + result_ids)
                    or len(set(ids)) != count
                    or sorted(result_ids) != sorted(ids)
                ):
                    raise InvalidToolHistory("native tool results do not match requests")
            elif all(item.get("role") == "user" and item.get("tool_name") for item in results):
                result_names = [item.get("tool_name") for item in results]
                if not all(names) or sorted(result_names) != sorted(names):
                    raise InvalidToolHistory("text tool results do not match requests")
            else:
                raise InvalidToolHistory("tool-call group has mixed or missing results")
            units.append([message, *results])
            index += count + 1
            continue
        if role == "assistant":
            end = index + 1
            while end < len(messages) and messages[end].get("role") == "user" and messages[end].get("tool_name"):
                end += 1
            units.append(messages[index:end])
            index = end
            continue
        if role == "tool" or (role == "user" and message.get("tool_name")):
            raise InvalidToolHistory("orphan tool result")
        units.append([message])
        index += 1
    return units


def _history_turns(units: list[list[dict]]) -> list[list[list[dict]]]:
    turns: list[list[list[dict]]] = []
    for unit in units:
        if _is_genuine_user(unit[0]) or not turns:
            turns.append([])
        turns[-1].append(unit)
    return turns


_RESULT_ID = re.compile(r"Full result ID:\s*([0-9a-fA-F-]{32,36})")
_FENCE_BEGIN = re.compile(r"<<<BEGIN UNTRUSTED ([A-Z ]+)>>>")


def _trim_tool_text(content: str, target_chars: int) -> str:
    ids = _RESULT_ID.findall(content)
    pointer = (
        f"\n[Full result ID: {ids[-1]}; call readTaskResult to read more.]"
        if ids else ""
    )
    begin = _FENCE_BEGIN.search(content)
    if begin:
        opening = begin.group(0)
        closing = f"<<<END UNTRUSTED {begin.group(1)}>>>"
        close_at = content.find(closing, begin.end())
        inner = content[begin.end():close_at if close_at >= 0 else len(content)]
        prefix = opening + "\n"
        suffix = "\n[Tool result truncated]\n" + closing + pointer
    else:
        inner = content
        prefix = ""
        suffix = "\n[Tool result truncated]" + pointer
    keep = max(0, target_chars - len(prefix) - len(suffix))
    return prefix + inner[:keep].rstrip() + suffix


def bound_context_messages(
    messages: list[dict],
    *,
    current_user_index: int,
    max_tokens: int,
    reserve_tokens: int = 0,
    estimator: Callable[[dict], int] = estimate_message_tokens,
) -> list[dict]:
    """Return a bounded copy of messages, preserving native and text tool groups.

    The first system message and indicated current user message are required.
    Older conversation turns and then older current-request groups are removed
    before tool result contents are shortened. At least the newest current
    group remains. A caller should reserve room for tools and generation with
    ``reserve_tokens``. An impossible budget raises explicitly.
    """
    if not messages or messages[0].get("role") != "system":
        raise ValueError("first message must be a system message")
    if not 0 < current_user_index < len(messages) or not _is_genuine_user(messages[current_user_index]):
        raise ValueError("current_user_index must identify the current user request")
    budget = int(max_tokens) - int(reserve_tokens)
    if budget <= 0:
        raise ContextBudgetExceeded(1, budget)

    copied = deepcopy(messages)
    system = copied[0]
    current_user = copied[current_user_index]
    history = _history_turns(_parse_units(copied[1:current_user_index]))
    current_units = _parse_units(copied[current_user_index + 1:])

    def flatten() -> list[dict]:
        return [
            system,
            *(item for turn in history for unit in turn for item in unit),
            current_user,
            *(item for unit in current_units for item in unit),
        ]

    def count() -> int:
        return sum(estimator(message) for message in flatten())

    required = estimator(system) + estimator(current_user)
    if required > budget:
        raise ContextBudgetExceeded(required, budget)

    while count() > budget and history:
        history.pop(0)
    while count() > budget and len(current_units) > 1:
        current_units.pop(0)

    while count() > budget:
        candidates = [
            message for unit in current_units for message in unit
            if message.get("role") == "tool" or (
                message.get("role") == "user" and message.get("tool_name")
            )
        ]
        candidates.sort(key=lambda message: len(str(message.get("content", ""))), reverse=True)
        changed = False
        for message in candidates:
            content = message.get("content")
            if not isinstance(content, str):
                continue
            trimmed = _trim_tool_text(content, len(content) // 2)
            if len(trimmed) < len(content):
                message["content"] = trimmed
                changed = True
                break
        if not changed:
            raise ContextBudgetExceeded(count(), budget)

    return flatten()
