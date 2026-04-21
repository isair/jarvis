"""Agentic-loop turn evaluator.

After each reply turn that produces natural-language content, a small LLM
decides whether the loop should terminate (the agent has done what it can
with its current allow-list) or keep working (a tool in the allow-list
could directly perform the user's expressed action but the agent replied
in prose instead).

Contract is binary: terminal vs continue. "Satisfied" and
"needs_user_input" are both terminal from the loop's perspective — both
mean stop looping and hand back to the user.

Fail-open on parse or transport failure collapses to ``terminal=True``.
Spinning a broken loop is worse than delivering a possibly-weak reply.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from ..debug import debug_log
from ..llm import call_llm_direct
from ..utils.redact import redact


@dataclass
class EvaluatorResult:
    terminal: bool
    nudge: str = ""
    reason: str = ""


_EVALUATOR_SYSTEM_PROMPT = (
    "You are judging whether an AI agent should keep working or stop. "
    "You see the user's query, the agent's just-produced turn, and the "
    "agent's available tools with one-line descriptions.\n\n"
    "CORE RULE: match the user's expressed action to the toolbox YOURSELF. "
    "Do NOT trust the agent's self-report. If the agent says 'I can't do "
    "this' but a tool in the toolbox can directly do it, that is a false "
    "refusal — return continue with a nudge that names the tool.\n\n"
    "Step-by-step:\n"
    "  1. What did the user ask for? Extract the core action or request.\n"
    "  2. Scan the toolbox. Does any tool's description cover that action?\n"
    "     The special tool `toolSearchTool` is a fallback: if no other tool "
    "fits, the agent is expected to call `toolSearchTool` to discover more "
    "tools, NOT to give up in prose.\n"
    "  3. Did the agent's turn actually invoke a fitting tool, or was it "
    "prose (an offer, a description, an apology, a refusal)?\n\n"
    "Return \"continue\" when a tool in the toolbox covers the user's "
    "action (including `toolSearchTool` as a discovery fallback) and the "
    "agent did not invoke a tool this turn. In the \"nudge\" field, name "
    "the specific tool the agent should call next and what to pass.\n\n"
    "Return \"terminal\" only when:\n"
    "  - the agent already invoked a fitting tool and the turn is a real "
    "answer grounded in the tool result, OR\n"
    "  - the user's request is pure conversation (greeting, chitchat, "
    "opinion) with no action to take, OR\n"
    "  - genuinely no tool in the toolbox (including `toolSearchTool`) "
    "could help, AND the agent's turn honestly communicates that.\n\n"
    "When in doubt between terminal and continue, prefer continue with a "
    "nudge — a wasted extra turn is cheaper than a false refusal reaching "
    "the user.\n\n"
    "Only two outcomes. Output strict JSON only, no prose, no code fences:\n"
    "  {\"terminal\": <bool>, \"nudge\": \"...\", \"reason\": \"...\"}\n\n"
    "The \"nudge\" field is empty when terminal is true. The \"reason\" "
    "field is a short log hint, never shown to the user.\n"
    "Do NOT answer the user's query yourself. Do NOT add commentary."
)


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_result(raw: str) -> EvaluatorResult:
    """Lenient JSON parse. Failures collapse to terminal=True (fail-open).

    Biased toward terminal: a stuck loop is worse than a possibly-weak
    reply, so any parse ambiguity ends the loop rather than continuing it.
    """
    if not raw:
        return EvaluatorResult(terminal=True, reason="evaluator_failed_open")
    text = raw.strip()
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
        return EvaluatorResult(terminal=True, reason="evaluator_failed_open")

    terminal_raw = candidate.get("terminal")
    if not isinstance(terminal_raw, bool):
        return EvaluatorResult(terminal=True, reason="evaluator_failed_open")
    nudge = candidate.get("nudge", "")
    if not isinstance(nudge, str):
        nudge = ""
    reason = candidate.get("reason", "")
    if not isinstance(reason, str):
        reason = ""
    return EvaluatorResult(
        terminal=bool(terminal_raw),
        nudge=nudge.strip(),
        reason=reason.strip(),
    )


def _resolve_evaluator_model(cfg) -> str:
    """Pick the LLM model for the evaluator pass.

    Resolution order: explicit ``evaluator_model`` → ``intent_judge_model`` →
    ``ollama_chat_model``. The evaluator is a small classification job;
    reusing the judge model keeps it on a small, already-warm model.
    """
    for candidate in (
        getattr(cfg, "evaluator_model", ""),
        getattr(cfg, "intent_judge_model", ""),
        getattr(cfg, "ollama_chat_model", ""),
    ):
        if candidate:
            return candidate
    return ""


def _format_available_tools(tools: list[tuple[str, str]]) -> str:
    if not tools:
        return "(none)"
    lines = []
    for name, desc in tools:
        desc_clean = (desc or "").strip().splitlines()[0] if desc else ""
        lines.append(f"- {name}: {desc_clean}" if desc_clean else f"- {name}")
    return "\n".join(lines)


def evaluate_turn(
    user_query: str,
    assistant_response_summary: str,
    available_tools: list[tuple[str, str]],
    turns_used: int,
    cfg,
) -> EvaluatorResult:
    """Classify whether the agentic loop should terminate after this turn.

    ``available_tools`` is a list of ``(name, one_line_description)`` tuples
    supplied by the engine — not redacted; it is engine-controlled, not
    user data.

    Fail-open returns ``terminal=True`` with ``reason="evaluator_failed_open"``.
    """
    user_query = redact(user_query) if isinstance(user_query, str) else ""
    assistant_response_summary = (
        redact(assistant_response_summary)
        if isinstance(assistant_response_summary, str)
        else ""
    )
    if not isinstance(available_tools, list):
        available_tools = []

    base_url = getattr(cfg, "ollama_base_url", "")
    chat_model = _resolve_evaluator_model(cfg)
    if not base_url or not chat_model:
        return EvaluatorResult(terminal=True, reason="evaluator_failed_open")

    try:
        timeout_sec = float(getattr(cfg, "llm_digest_timeout_sec", 8.0))
    except (TypeError, ValueError):
        timeout_sec = 8.0
    thinking = bool(getattr(cfg, "llm_thinking_enabled", False))

    tools_block = _format_available_tools(available_tools)
    user_content = (
        f"USER QUERY: {user_query}\n\n"
        f"ASSISTANT TURN (summary): {assistant_response_summary}\n\n"
        f"AGENT TOOLBOX:\n{tools_block}\n\n"
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
        debug_log(f"evaluator failed (non-fatal, terminal): {e}", "planning")
        return EvaluatorResult(terminal=True, reason="evaluator_failed_open")

    if not raw:
        debug_log("evaluator returned empty response — terminal", "planning")
        return EvaluatorResult(terminal=True, reason="evaluator_failed_open")

    result = _parse_result(raw)
    debug_log(
        f"evaluator: terminal={result.terminal} nudge={result.nudge!r} "
        f"reason={result.reason!r} (turn {turns_used})",
        "planning",
    )
    return result
