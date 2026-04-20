"""Agentic-loop turn evaluator.

After each reply turn that produces natural-language content, a small LLM
decides whether the loop should terminate (query satisfied, or the
assistant is asking a clarifying question) or keep going.

Parses strict JSON; on timeout or parse failure returns ``continue`` so
the max-turn cap stays the only hard backstop.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal, Optional

from ..debug import debug_log
from ..llm import call_llm_direct


EvaluatorReason = Literal["satisfied", "needs_user_input", "continue"]


@dataclass
class EvaluatorResult:
    terminal: bool
    reason: EvaluatorReason
    clarification_question: Optional[str] = None


_EVALUATOR_SYSTEM_PROMPT = (
    "You are an agentic-loop evaluator for an AI assistant. You receive the "
    "user's original query and a short summary of the assistant's latest "
    "turn. Classify the turn as ONE of:\n"
    "  - \"satisfied\": the assistant's reply already addresses the user's "
    "query; the user has received the answer.\n"
    "  - \"needs_user_input\": the assistant literally cannot proceed "
    "without asking the user a clarifying question, and the reply is that "
    "question.\n"
    "  - \"continue\": anything else; the assistant should keep working "
    "(e.g. the reply is partial, acknowledges needing more steps, or is "
    "clearly not a final answer).\n\n"
    "Reply with STRICT JSON only, no prose, no code fences:\n"
    "  {\"terminal\": <bool>, \"reason\": \"satisfied\" | "
    "\"needs_user_input\" | \"continue\"}\n\n"
    "Rules:\n"
    "- \"satisfied\" and \"needs_user_input\" imply terminal=true; "
    "\"continue\" implies terminal=false.\n"
    "- When in doubt, choose \"continue\".\n"
    "- Do NOT answer the user's query yourself. Do NOT add commentary."
)


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_result(raw: str) -> EvaluatorResult:
    """Lenient JSON parse → EvaluatorResult. Failures collapse to 'continue'."""
    if not raw:
        return EvaluatorResult(terminal=False, reason="continue")
    text = raw.strip()
    # Strip common wrappers (markdown fences, quotes).
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    candidate: Optional[dict] = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            candidate = parsed
    except Exception:
        match = _JSON_OBJECT_RE.search(text)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    candidate = parsed
            except Exception:
                candidate = None
    if not candidate:
        return EvaluatorResult(terminal=False, reason="continue")

    reason_raw = str(candidate.get("reason", "")).strip().lower()
    if reason_raw not in ("satisfied", "needs_user_input", "continue"):
        return EvaluatorResult(terminal=False, reason="continue")
    terminal = reason_raw != "continue"
    clarification = candidate.get("clarification_question")
    if not isinstance(clarification, str):
        clarification = None
    return EvaluatorResult(
        terminal=terminal,
        reason=reason_raw,  # type: ignore[arg-type]
        clarification_question=clarification,
    )


def evaluate_turn(
    user_query: str,
    assistant_response_summary: str,
    turns_used: int,
    cfg,
) -> EvaluatorResult:
    """Classify whether the agentic loop should terminate after this turn.

    Fail-open on any error — returning ``continue`` keeps ``agentic_max_turns``
    as the single hard backstop.
    """
    base_url = getattr(cfg, "ollama_base_url", "")
    chat_model = getattr(cfg, "ollama_chat_model", "")
    if not base_url or not chat_model:
        return EvaluatorResult(terminal=False, reason="continue")

    try:
        timeout_sec = float(getattr(cfg, "llm_digest_timeout_sec", 8.0))
    except (TypeError, ValueError):
        timeout_sec = 8.0
    thinking = bool(getattr(cfg, "llm_thinking_enabled", False))

    user_content = (
        f"USER QUERY: {user_query}\n\n"
        f"ASSISTANT TURN (summary): {assistant_response_summary}\n\n"
        f"TURNS USED SO FAR: {turns_used}\n\n"
        "Classify now. Reply with strict JSON only."
    )

    try:
        raw = call_llm_direct(
            base_url=base_url,
            chat_model=chat_model,
            system_prompt=_EVALUATOR_SYSTEM_PROMPT,
            user_content=user_content,
            timeout_sec=timeout_sec,
            thinking=thinking,
        )
    except Exception as e:
        debug_log(f"evaluator failed (non-fatal, continuing loop): {e}", "planning")
        return EvaluatorResult(terminal=False, reason="continue")

    if not raw:
        debug_log("evaluator returned empty response — continuing loop", "planning")
        return EvaluatorResult(terminal=False, reason="continue")

    result = _parse_result(raw)
    debug_log(
        f"evaluator: reason={result.reason} terminal={result.terminal} "
        f"(turn {turns_used})",
        "planning",
    )
    return result
