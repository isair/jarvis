"""Controlled replay of query rewriting and separate transcript context."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import ipaddress
import json
import os
import re
import statistics
import tempfile
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlsplit
from unittest.mock import patch

import requests

from jarvis.debug import debug_log
from jarvis.listening.intent_judge import IntentJudge, IntentJudgeConfig
from evals.rewrite_baseline import RewriteBaselineJudge
from jarvis.listening.transcript_buffer import TranscriptSegment

from evals.helpers import is_fallback_reply, is_max_turns_digest

ARMS = ("rewrite", "raw_only", "raw_context")


def comparison_settings(base_url: str, model: str, *, fast_model: str = "", api_key: str = ""):
    """Construct isolated production Settings without changing the user's config."""
    from jarvis.config import load_settings
    RequestRecorder(base_url, no_thinking=True)
    with tempfile.TemporaryDirectory(prefix="jarvis-query-comparison-") as directory:
        path = Path(directory) / "config.json"
        path.write_text(json.dumps({
            "llm_provider": "openai_compatible", "llm_base_url": base_url,
            "llm_chat_model": model, "fast_model": fast_model or model,
            "db_path": ":memory:", "location_enabled": False,
            "mcps": {}, "tts_enabled": False, "memory_enrichment_source": "diary",
        }))
        with patch.dict(os.environ, {"JARVIS_CONFIG_PATH": str(path)}):
            cfg = load_settings()
    # Credentials are held in memory and excluded from benchmark files.
    return replace(cfg, llm_api_key=api_key, sqlite_vss_path=None)


@dataclass(frozen=True)
class ComparisonCase:
    name: str
    category: str
    transcript: tuple[tuple[str, bool], ...]
    expected_tool: str | None
    argument_terms: tuple[str, ...]
    answer_terms: tuple[str, ...]
    payload: str
    dialogue: tuple[tuple[str, str], ...] = ()
    hot_window: bool = False
    last_tts: str = ""


@dataclass
class ComparisonResult:
    case: str
    category: str
    arm: str
    repeat: int
    query: str = ""
    reply: str = ""
    selected_tools: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    requests: list[dict] = field(default_factory=list)
    latency_ms: float = 0
    error: str = ""


def judge_context(case: ComparisonCase) -> tuple[list[TranscriptSegment], str]:
    segments = [
        TranscriptSegment(text, 1000 + 2 * i, 1002 + 2 * i, is_during_tts=echo)
        for i, (text, echo) in enumerate(case.transcript)
    ]
    current = case.transcript[-1][0]
    wake = None if case.hot_window else segments[-1].start_time + 0.8
    state = IntentJudge()._build_user_prompt(
        segments, wake, case.last_tts, 999 if case.last_tts else 0,
        case.hot_window, current,
    )
    return segments, state


def prepare_query(case: ComparisonCase, arm: str, *, judgment=None) -> tuple[str, str]:
    if arm not in ARMS:
        raise ValueError(f"Unknown comparison arm: {arm}")
    if arm == "rewrite":
        if judgment is None or not judgment.directed or judgment.stop or not judgment.query.strip():
            raise ValueError("No usable directed query from the intent judge")
        return judgment.query, ""
    if arm == "raw_only":
        return case.transcript[-1][0], ""
    from jarvis.listening.speech_context import SpeechContext
    segments, _ = judge_context(case)
    context = SpeechContext.capture(segments, current_text=case.transcript[-1][0], last_tts=case.last_tts)
    return case.transcript[-1][0], context.render()


def _arguments_match(case: ComparisonCase, args: dict) -> bool:
    from jarvis.tools.registry import BUILTIN_TOOLS
    if not isinstance(args, dict) or case.expected_tool not in BUILTIN_TOOLS:
        return False
    schema = BUILTIN_TOOLS[case.expected_tool].inputSchema
    if any(key not in args for key in schema.get("required", [])):
        return False
    properties = schema.get("properties", {})
    used = {key: value for key, value in args.items() if key in properties}
    if any(properties[key].get("type") == "string" and not isinstance(value, str)
           for key, value in used.items()):
        return False
    text = json.dumps(list(used.values()), ensure_ascii=False).casefold()
    return all(term.casefold() in text for term in case.argument_terms)


def _valid_tool_calls(calls) -> bool:
    if not isinstance(calls, list) or not calls:
        return False
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            return False
        function = call["function"]
        if not isinstance(function.get("name"), str) or not function["name"].strip():
            return False
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                return False
        if not isinstance(arguments, dict):
            return False
    return True


def score_result(case: ComparisonCase, result: ComparisonResult) -> dict:
    selected = set(result.selected_tools) - {"stop", "toolSearchTool"}
    routing = case.expected_tool in selected if case.expected_tool else not selected
    matching = [call for call in result.tool_calls if call["name"] == case.expected_tool]
    arguments = any(_arguments_match(case, call["args"]) for call in matching)
    if case.expected_tool is None:
        arguments = not result.tool_calls
    unexpected = any(call["name"] != case.expected_tool for call in result.tool_calls)
    answer_text = re.sub(r"(?<=\d)[,\u00a0\u202f](?=\d)", "", result.reply.casefold())
    answer = bool(result.reply.strip()) and not is_fallback_reply(result.reply) and not is_max_turns_digest(result.reply) and all(
        term.casefold() in answer_text for term in case.answer_terms
    )
    valid = bool(result.requests) and all(request.get("valid", False) for request in result.requests)
    return {
        "routing_correct": routing, "arguments_correct": arguments,
        "answer_correct": answer, "unexpected_tools": unexpected,
        "model_valid": valid,
        "passed": bool(arguments and answer and valid and not unexpected and not result.error),
    }


def summarise(rows: list[dict]) -> dict:
    summaries = {}
    for arm in ARMS:
        group = [row for row in rows if row["arm"] == arm]
        if not group:
            continue
        cases = {row["case"] for row in group}
        latencies = [row["latency_ms"] for row in group]
        successful = [row["latency_ms"] for row in group if row["passed"]]
        def mean_tokens(key):
            totals = [sum(r[key] for r in row["requests"]) for row in group
                      if row["requests"] and all(isinstance(r.get(key), int) for r in row["requests"])]
            return round(statistics.mean(totals), 1) if totals else None
        summaries[arm] = {
            "runs": len(group), "passed": sum(row["passed"] for row in group),
            "pass_rate": sum(row["passed"] for row in group) / len(group),
            "unique_cases": len(cases),
            "cases_all_repeats_pass": sum(
                all(row["passed"] for row in group if row["case"] == name) for name in cases
            ),
            "errors": sum(bool(row.get("error")) for row in group),
            "invalid_model_runs": sum(not row["model_valid"] for row in group),
            "routing_correct": sum(row["routing_correct"] for row in group),
            "arguments_correct": sum(row["arguments_correct"] for row in group),
            "answer_correct": sum(row["answer_correct"] for row in group),
            "median_latency_ms": round(statistics.median(latencies), 1),
            "median_success_latency_ms": round(statistics.median(successful), 1) if successful else None,
            "mean_model_requests": round(statistics.mean(len(row["requests"]) for row in group), 2),
            "mean_input_tokens": mean_tokens("input_tokens"),
            "mean_output_tokens": mean_tokens("output_tokens"),
            "runs_with_complete_token_usage": sum(
                bool(row["requests"]) and all(
                    isinstance(r.get("input_tokens"), int) and isinstance(r.get("output_tokens"), int)
                    for r in row["requests"]
                ) for row in group
            ),
        }
    return summaries


class RequestRecorder:
    """Instrument local model requests without changing production functions.

    The context arm appends the same fenced data to router, planner and reply
    requests. The request profile is uniform across arms. Credentials and
    request headers are excluded from the trace.
    """

    def __init__(self, base_url: str, *, no_thinking: bool):
        parsed = urlsplit(base_url)
        try:
            local = parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname).is_loopback
        except (TypeError, ValueError):
            local = False
        if not local or parsed.scheme not in {"http", "https"} or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Comparison requires a loopback model endpoint")
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.no_thinking = no_thinking
        self.phase = "reply"
        self.records = []

    @contextlib.contextmanager
    def stage(self, name):
        previous, self.phase = self.phase, name
        try:
            yield
        finally:
            self.phase = previous

    def post(self, url, **kwargs):
        if url != self.endpoint:
            self.records.append({"phase": self.phase, "valid": False, "error": "unexpected endpoint"})
            raise requests.exceptions.RequestException("Unexpected model endpoint")
        payload = copy.deepcopy(kwargs.pop("json"))
        payload["temperature"] = 0
        if self.no_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        fingerprint = hashlib.sha256(json.dumps(payload["messages"], sort_keys=True).encode()).hexdigest()
        record = {"phase": self.phase, "valid": False, "context_attached": any("<<<BEGIN TRANSCRIPT>>>" in str(m.get("content", "")) for m in payload["messages"]),
                  "prompt_sha256": fingerprint, "timeout_sec": kwargs.get("timeout"),
                  "max_tokens": payload.get("max_tokens")}
        started = time.monotonic()
        try:
            with requests.Session() as session:
                session.trust_env = False
                kwargs.pop("allow_redirects", None)
                response = session.post(url, json=payload, allow_redirects=False, **kwargs)
            record["http_status"] = response.status_code
            if response.status_code != 200:
                record["error"] = "HTTP failure"
                return response
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
            content = (message.get("content") or "").strip()
            calls = message.get("tool_calls")
            structure_valid = calls is None or (
                isinstance(calls, list) and (not calls or _valid_tool_calls(calls))
            )
            complete = choice.get("finish_reason") == "stop" or (
                choice.get("finish_reason") == "tool_calls" and bool(calls) and structure_valid
            )
            record.update(
                content=message.get("content") or "",
                tool_calls=calls or [],
                finish_reason=choice.get("finish_reason"),
                valid=bool(content or calls) and complete and structure_valid,
            )
            if self.phase == "rewrite" and message.get("reasoning_content"):
                # The production judge can recover a complete JSON verdict from reasoning.
                record["valid"] = bool(content or message.get("reasoning_content", "").strip()) and complete and structure_valid
            usage = data.get("usage")
            if usage is None:
                usage = {}
            if not isinstance(usage, dict):
                raise ValueError("Invalid model token usage")
            details = usage.get("prompt_tokens_details")
            if details is None:
                details = {}
            if not isinstance(details, dict):
                raise ValueError("Invalid model cache usage")
            record.update(
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                cached_tokens=details.get("cached_tokens"),
            )
            return response
        except Exception as exc:
            record["valid"] = False
            record["error"] = type(exc).__name__
            raise
        finally:
            record["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
            self.records.append(record)


def run_case(case: ComparisonCase, arm: str, repeat: int, cfg, *, no_thinking=True) -> ComparisonResult:
    """Replay the real intent judge and reply engine with synthetic tool data."""
    from jarvis.memory.conversation import DialogueMemory
    from jarvis.memory.db import Database
    from jarvis.reply import engine
    from jarvis.tools.types import ToolExecutionResult

    result = ComparisonResult(case.name, case.category, arm, repeat)
    recorder = RequestRecorder(cfg.llm_base_url, no_thinking=no_thinking)
    memory = DialogueMemory(inactivity_timeout=300, max_interactions=20)
    for role, content in case.dialogue:
        memory.add_message(role, content)
    router, planner = engine.select_tools, engine.plan_query

    def route(*args, **kwargs):
        with recorder.stage("router"):
            selected = router(*args, **kwargs)
        result.selected_tools = list(selected)
        return selected

    def plan(*args, **kwargs):
        with recorder.stage("planner"):
            return planner(*args, **kwargs)

    def tool_run(db, cfg, tool_name, tool_args, **_kwargs):
        args = tool_args or {}
        result.tool_calls.append({"name": tool_name, "args": args})
        correct = tool_name == case.expected_tool and _arguments_match(case, args)
        return ToolExecutionResult(
            success=correct,
            reply_text=case.payload if correct else "No matching result in this synthetic fixture.",
        )

    started = time.monotonic()
    try:
        with contextlib.closing(Database(":memory:", sqlite_vss_path=None)) as db, contextlib.ExitStack() as stack:
            stack.enter_context(patch("jarvis.llm.openai_compatible.requests.post", recorder.post))
            stack.enter_context(patch.object(engine, "select_tools", route))
            stack.enter_context(patch.object(engine, "plan_query", plan))
            stack.enter_context(patch.object(engine, "run_tool_with_retries", tool_run))
            # This replay has no persisted diary entries to retrieve.
            stack.enter_context(patch("jarvis.memory.conversation.search_conversation_memory_by_keywords", return_value=[]))
            # The replay prints one compact result; retain engine output in memory.
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            judgment = None
            if arm == "rewrite":
                segments, _ = judge_context(case)
                judge = RewriteBaselineJudge(IntentJudgeConfig(
                    cfg=cfg, model=cfg.fast_model, timeout_sec=cfg.intent_judge_timeout_sec,
                ))
                with recorder.stage("rewrite"):
                    judgment = judge.judge(
                        segments, wake_timestamp=None if case.hot_window else segments[-1].start_time + 0.8,
                        last_tts_text=case.last_tts,
                        last_tts_finish_time=999 if case.last_tts else 0,
                        in_hot_window=case.hot_window,
                        current_text=case.transcript[-1][0],
                    )
            result.query, _ = prepare_query(case, arm, judgment=judgment)
            speech_context = None
            if arm == "raw_context":
                from jarvis.listening.speech_context import SpeechContext
                segments, _ = judge_context(case)
                speech_context = SpeechContext.capture(segments, current_text=result.query, last_tts=case.last_tts, assistant_names=(cfg.wake_word, *cfg.wake_aliases))
            result.reply = engine.run_reply_engine(db, cfg, None, result.query, memory, quiet=True, speech_context=speech_context) or ""
    except Exception as exc:
        location = traceback.extract_tb(exc.__traceback__)[-1]
        result.error = f"{type(exc).__name__} at {Path(location.filename).name}:{location.lineno}"
        debug_log(f"🧪 Query comparison {case.name}/{arm}: {result.error}", "voice")
    finally:
        result.latency_ms = round((time.monotonic() - started) * 1000, 1)
        result.requests = recorder.records
    return result
