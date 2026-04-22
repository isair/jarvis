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
from ..llm import call_llm_direct, chat_with_messages, extract_text_from_response
from ..utils.redact import redact


@dataclass
class EvaluatorResult:
    terminal: bool
    nudge: str = ""
    reason: str = ""
    # Structured tool-call intent. When the judge has identified a
    # specific tool + arguments in the nudge (salvage path or an
    # obvious missed invocation), it also emits this dict so the
    # engine can execute the call directly instead of relying on the
    # chat model to obey a free-form nudge. Shape: {"name": str,
    # "arguments": dict}. None when the judge is not confident.
    tool_call: Optional[dict] = None


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
    "  2. Check `TOOLS ALREADY INVOKED THIS REPLY`. If a tool covering the "
    "user's action has ALREADY been invoked with sensible args and returned "
    "a non-error result, the action is done — return terminal. Do NOT "
    "ask the agent to re-run a tool that already ran successfully, even if "
    "the current prose turn reads weakly. The engine executed the tool; "
    "the chat model's failure to narrate it is not grounds for another "
    "invocation.\n"
    "  3. Otherwise scan the toolbox. Does any tool's description cover "
    "that action? The special tool `toolSearchTool` is a fallback: if no "
    "other tool fits, the agent is expected to call `toolSearchTool` to "
    "discover more tools, NOT to give up in prose.\n"
    "  4. Did the agent's turn actually invoke a fitting tool, or was it "
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
    "SINGLE-PART vs MULTI-PART QUERIES: a single-part query asks one "
    "thing (\"what's the weather today\", \"who directed Possessor\", "
    "\"open YouTube\"). A multi-part query asks for two or more "
    "distinct pieces of information, usually joined by \"and\", \"or\", "
    "a comma, or phrased as a compare/list request (\"who directed "
    "Possessor AND what else have they directed\", \"compare the "
    "weather in Paris and London\", \"tell me about X, Y, and Z\").\n"
    "  - For SINGLE-PART queries: if the agent's turn contains concrete "
    "facts that address the ask (names, numbers, dates, locations, "
    "weather conditions, temperatures, conclusions tied to the ask), "
    "return terminal. You do NOT need proof that a tool ran this turn — "
    "the engine already logs tool calls; the presence of grounded facts "
    "in the reply is sufficient evidence of a real answer. Do NOT force "
    "an extra turn just because the turn reads conversationally.\n"
    "  - For MULTI-PART queries: count the parts. If every part is "
    "addressed with concrete facts in the reply, terminal. If at least "
    "one part is unaddressed or not yet answered, return continue and "
    "nudge for the missing part.\n\n"
    "GARBLED / MALFORMED TURNS: if the agent's turn is not readable "
    "English prose — for example it contains raw tool-protocol markers "
    "like `tool_code` or `tool_output` blocks, special sentinel tokens "
    "like `<unused88>` (or any `<unused…>` variant), bare `tool_calls:` "
    "text, truncated JSON, or code/data dumps where a natural reply "
    "should be — return \"continue\". Shipping garbled text to the "
    "user is worse than one extra turn. The engine also catches the "
    "known shapes deterministically; your job here is defence-in-depth "
    "for novel leaks.\n\n"
    "  SALVAGE a failed tool call when you can. If the garbled turn "
    "looks like the agent tried to invoke a tool but emitted the "
    "protocol as text — e.g. `tool_code\\nprint(google_search.search("
    "query=\"sam smith biography\"))`, or a bare `tool_calls: "
    "[{\"name\": \"webSearch\", \"arguments\": {\"query\": \"...\"}}]` "
    "JSON blob, or a `<unused…>` block wrapping a tool invocation — "
    "extract the intended tool and arguments and name the tool in the "
    "nudge, e.g. \"call webSearch with query='sam smith biography'\". "
    "Only name a tool that actually appears in the toolbox above; if "
    "the extracted tool is not in the allow-list, pick the closest "
    "matching tool or fall back to a \"produce a natural-language "
    "reply\" nudge. If the garbled turn is unrecoverable (truncated "
    "JSON with no name, bare `<unused88>` with no content, random "
    "data dump), nudge \"produce a natural-language reply\" instead. "
    "Do NOT fabricate arguments the garbled turn did not contain.\n\n"
    "When in doubt: for MULTI-PART queries with any part unaddressed, "
    "prefer continue — a wasted extra turn is cheaper than handing back "
    "a half-answer. For SINGLE-PART queries whose ask is already "
    "addressed by concrete facts in the turn, prefer terminal — looping "
    "past a good answer burns the agentic-turn budget, which fires the "
    "max-turns digest summariser and prepends a \"could not fully "
    "finish\" caveat onto an otherwise correct reply. That caveat is a "
    "worse UX than terminating on the grounded reply.\n\n"
    "STRUCTURED TOOL CALL: whenever you name a specific tool AND "
    "arguments in the nudge (salvage path, or an obvious missed "
    "invocation), ALSO emit a structured `tool_call` field with the "
    "exact same intent. The engine uses it to execute the call directly "
    "on behalf of the agent — this is the only reliable path when the "
    "chat model is a small one that tends to ignore textual nudges. "
    "Shape: `\"tool_call\": {\"name\": \"<toolName>\", \"arguments\": "
    "{<k>: <v>, ...}}`. The `name` MUST appear in the toolbox above. "
    "`arguments` must be a JSON object — use `{}` when the tool takes "
    "none. OMIT the field (or set it to null) when you are nudging for "
    "prose (\"produce a natural-language reply\") or when you cannot "
    "identify the exact arguments — never fabricate arguments you did "
    "not extract from the garbled turn or derive from the user query.\n\n"
    "  ARGUMENT KEYS MUST BE EXACT. Each tool in the toolbox is listed "
    "with its parameter signature, e.g. `webSearch(search_query: string "
    "required)`. When you emit `arguments`, use those exact parameter "
    "names verbatim — do NOT invent plausible-sounding alternatives "
    "(\"query\" when the schema says \"search_query\", \"url\" when it "
    "says \"page_url\"). The engine will reject a call whose keys do "
    "not match the schema. If the toolbox entry shows no parameters, "
    "pass `{}`. If you are unsure what arguments a tool takes, omit "
    "`tool_call` entirely and nudge in prose.\n\n"
    "Only two outcomes. Output strict JSON only, no prose, no code fences:\n"
    "  {\"terminal\": <bool>, \"nudge\": \"...\", \"reason\": \"...\", "
    "\"tool_call\": {\"name\": \"...\", \"arguments\": {...}} | null}\n\n"
    "The \"nudge\" field is empty when terminal is true. The \"reason\" "
    "field is a short log hint, never shown to the user. The "
    "\"tool_call\" field is null when terminal is true or when no "
    "specific tool invocation was identified.\n"
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
    tool_call: Optional[dict] = None
    tc_raw = candidate.get("tool_call")
    if isinstance(tc_raw, dict):
        name = tc_raw.get("name")
        if isinstance(name, str) and name.strip():
            args_raw = tc_raw.get("arguments")
            if not isinstance(args_raw, dict):
                args_raw = {}
            tool_call = {"name": name.strip(), "arguments": args_raw}

    return EvaluatorResult(
        terminal=bool(terminal_raw),
        nudge=nudge.strip(),
        reason=reason.strip(),
        tool_call=tool_call,
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


def _format_param_schema(schema: Optional[dict]) -> str:
    """Render a JSON schema as a compact ``(arg: type [required], ...)`` summary.

    The evaluator uses this to emit ``tool_call.arguments`` with the correct
    argument keys. Without the schema, a small evaluator model tends to
    hallucinate plausible-looking argument names (``query`` instead of
    ``search_query``) that pass through the engine's allow-list check but
    fail the tool's own validation, producing an infinite repair loop.
    """
    if not isinstance(schema, dict):
        return ""
    props = schema.get("properties")
    if not isinstance(props, dict) or not props:
        return "()"
    required = set()
    req_raw = schema.get("required")
    if isinstance(req_raw, list):
        required = {str(r) for r in req_raw if isinstance(r, str)}
    parts = []
    for key, spec in props.items():
        type_hint = ""
        if isinstance(spec, dict):
            t = spec.get("type")
            if isinstance(t, str):
                type_hint = t
            elif isinstance(t, list):
                type_hint = "|".join(str(x) for x in t if isinstance(x, str))
        req_marker = " required" if key in required else ""
        if type_hint:
            parts.append(f"{key}: {type_hint}{req_marker}")
        else:
            parts.append(f"{key}{req_marker}")
    return "(" + ", ".join(parts) + ")"


def _format_available_tools(tools: list) -> str:
    """Render the toolbox for the evaluator prompt.

    Accepts either ``(name, desc)`` or ``(name, desc, schema)`` tuples. When
    a schema is supplied its parameter names and types are rendered inline
    so the evaluator emits ``tool_call.arguments`` with real argument keys
    rather than guessed ones.
    """
    if not tools:
        return "(none)"
    lines = []
    for entry in tools:
        if not isinstance(entry, tuple):
            continue
        name = entry[0] if len(entry) >= 1 else ""
        desc = entry[1] if len(entry) >= 2 else ""
        schema = entry[2] if len(entry) >= 3 else None
        desc_clean = (desc or "").strip().splitlines()[0] if desc else ""
        params = _format_param_schema(schema) if schema else ""
        head = f"{name}{params}" if params else f"{name}"
        lines.append(f"- {head}: {desc_clean}" if desc_clean else f"- {head}")
    return "\n".join(lines)


def _format_invoked_tools(invoked: list[tuple[str, str, str]]) -> str:
    """Render the ``(name, args_summary, result_summary)`` history for the prompt.

    Args and results are truncated — the evaluator only needs enough to tell
    that the tool ran and produced output, not the full payload.
    """
    if not invoked:
        return "(none yet this reply)"
    lines = []
    for name, args_s, result_s in invoked:
        args_clean = (args_s or "").strip().replace("\n", " ")
        result_clean = (result_s or "").strip().replace("\n", " ")
        if len(args_clean) > 160:
            args_clean = args_clean[:157] + "…"
        if len(result_clean) > 240:
            result_clean = result_clean[:237] + "…"
        lines.append(
            f"- {name} args={args_clean or '{}'} → result={result_clean or '(empty)'}"
        )
    return "\n".join(lines)


_ALLOWED_MESSAGE_KEYS = {"role", "content", "tool_call_id", "tool_calls", "name"}


def _sanitise_messages(messages: list[dict]) -> list[dict]:
    """Strip engine-internal annotations (e.g. ``tool_name`` for duplicate
    detection) that are not part of the Ollama chat-completion wire format.

    Keeping the wire shape deterministic is load-bearing: Ollama reuses the
    KV cache when the incoming prefix byte-matches the last request, so any
    drift in message keys or ordering defeats the prompt-cache reuse that
    the tail-append evaluator path was designed for.
    """
    clean: list[dict] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        clean.append({k: v for k, v in m.items() if k in _ALLOWED_MESSAGE_KEYS})
    return clean


def _build_evaluator_directive(
    user_query: str,
    assistant_response_summary: str,
    available_tools: list[tuple[str, str]],
    invoked_tools: list[tuple[str, str, str]],
    turns_used: int,
) -> str:
    """Build the tail-appended user directive for the cache-friendly path.

    The chat-turn history (system prompt + dialogue + user query +
    assistant reply) is already in the KV cache. This directive adds
    the classification rubric and dynamic context on top, and ALSO
    repeats the user query + assistant turn under explicit ``USER
    QUERY`` / ``ASSISTANT TURN (summary)`` framing.

    The repetition is load-bearing. Without it, small models struggle to
    identify which assistant message in the chat history is the one they
    should be classifying — the history contains synthesised tool-call
    envelopes, tool results and prose replies interleaved, and the
    evaluator kept re-firing the same structured ``tool_call`` in a
    tight loop because the model scored an older tool-call envelope as
    "the prose to judge" instead of the latest assistant reply. The
    cost of duplicating two short strings here is negligible compared
    with the reliability the explicit framing buys us, and the shared
    prefix with the chat turn still carries the full KV cache benefit.
    """
    tools_block = _format_available_tools(available_tools)
    invoked_block = _format_invoked_tools(invoked_tools)
    return (
        f"{_EVALUATOR_SYSTEM_PROMPT}\n\n"
        f"USER QUERY: {user_query}\n\n"
        f"ASSISTANT TURN (summary): {assistant_response_summary}\n\n"
        f"AGENT TOOLBOX (for this turn):\n{tools_block}\n\n"
        f"TOOLS ALREADY INVOKED THIS REPLY (with args and results):\n{invoked_block}\n\n"
        f"TURNS USED SO FAR: {turns_used}\n\n"
        "Classify the ASSISTANT TURN above now. Reply with strict JSON only. "
        "Do NOT answer the user's query again, do NOT call a tool, do NOT emit prose."
    )


def evaluate_turn(
    user_query: str,
    assistant_response_summary: str,
    available_tools: list,
    turns_used: int,
    cfg,
    invoked_tools: Optional[list[tuple[str, str, str]]] = None,
    chat_messages: Optional[list[dict]] = None,
) -> EvaluatorResult:
    """Classify whether the agentic loop should terminate after this turn.

    ``available_tools`` is a list of ``(name, one_line_description)`` or
    ``(name, one_line_description, input_schema)`` tuples supplied by the
    engine — not redacted; it is engine-controlled, not user data. When the
    schema is present, its parameter names/types are rendered inline in the
    toolbox block so the evaluator emits ``tool_call.arguments`` with real
    argument keys rather than hallucinated ones.

    ``invoked_tools`` is an optional list of ``(name, args_summary,
    result_summary)`` tuples for tools already executed during this reply.
    This lets the evaluator tell the difference between "agent hasn't tried
    the tool" (nudge it) and "tool already ran successfully but agent
    replied in prose instead of summarising" (terminal — don't re-run). The
    result_summary is redacted defensively because tool output can echo
    user-provided text.

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
    if invoked_tools is None or not isinstance(invoked_tools, list):
        invoked_tools = []
    else:
        invoked_tools = [
            (
                str(n),
                str(a) if a is not None else "",
                redact(str(r)) if r is not None else "",
            )
            for entry in invoked_tools
            if isinstance(entry, tuple) and len(entry) == 3
            for n, a, r in [entry]
        ]

    base_url = getattr(cfg, "ollama_base_url", "")
    chat_model = _resolve_evaluator_model(cfg)
    if not base_url or not chat_model:
        return EvaluatorResult(terminal=True, reason="evaluator_failed_open")

    try:
        timeout_sec = float(getattr(cfg, "llm_digest_timeout_sec", 8.0))
    except (TypeError, ValueError):
        timeout_sec = 8.0
    thinking = bool(getattr(cfg, "llm_thinking_enabled", False))

    # Cache-friendly path: when the evaluator runs on the same model as the
    # chat turn AND the caller hands us the live message history, ride that
    # history as a tail-appended user directive instead of firing a fresh
    # request with a separate system prompt. Ollama's KV cache reuses the
    # matching prefix, so the evaluator only pays prefill cost for the short
    # directive rather than re-processing the full chat context on every
    # loop iteration. This is a pure latency optimisation — the parse and
    # fail-open contract is identical on both paths.
    chat_model_for_main_turn = getattr(cfg, "ollama_chat_model", "")
    use_cache_path = (
        isinstance(chat_messages, list)
        and len(chat_messages) > 0
        and chat_model
        and chat_model == chat_model_for_main_turn
    )
    if use_cache_path:
        directive = _build_evaluator_directive(
            user_query,
            assistant_response_summary,
            available_tools,
            invoked_tools,
            turns_used,
        )
        messages = _sanitise_messages(chat_messages) + [
            {"role": "user", "content": directive}
        ]
        try:
            resp = chat_with_messages(
                base_url=base_url,
                chat_model=chat_model,
                messages=messages,
                timeout_sec=timeout_sec,
                thinking=thinking,
            )
        except Exception as e:
            debug_log(f"evaluator (cache path) failed (non-fatal, terminal): {e}", "planning")
            return EvaluatorResult(terminal=True, reason="evaluator_failed_open")
        if not isinstance(resp, dict):
            debug_log("evaluator (cache path) empty response — terminal", "planning")
            return EvaluatorResult(terminal=True, reason="evaluator_failed_open")
        raw = extract_text_from_response(resp) or ""
    else:
        tools_block = _format_available_tools(available_tools)
        invoked_block = _format_invoked_tools(invoked_tools)
        user_content = (
            f"USER QUERY: {user_query}\n\n"
            f"ASSISTANT TURN (summary): {assistant_response_summary}\n\n"
            f"AGENT TOOLBOX:\n{tools_block}\n\n"
            f"TOOLS ALREADY INVOKED THIS REPLY (with args and results):\n{invoked_block}\n\n"
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
