"""⏱️ LLM call timing recorder for local backend calls.

The recorder wraps Ollama and OpenAI-compatible backend methods and groups
elapsed time by the calling LLM context.

Usage:
    with TimingRecorder() as rec:
        run_reply_engine(...)
    rec.print_report()
    assert rec.p95("evaluator") < rec.p95("main_chat_turn")  # shape check
"""

from __future__ import annotations

import sys
import time
import statistics
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Optional

from jarvis.llm.ollama import OllamaBackend
from jarvis.llm.openai_compatible import OpenAICompatibleBackend


# Map caller __qualname__ → graph context name. Matches the 13 contexts in
# docs/llm_contexts.md. Anything not listed gets lumped into "other" so we
# notice new call sites drift in without us updating the doc.
#
# ⚠️  This mapping mirrors docs/llm_contexts.md. When you add, remove, or
# rename an LLM context per the CLAUDE.md rule, update both in the same PR
# — the perf harness silently buckets unknown callers into "other:<qualname>"
# so drift here is visible but not loud.
_CALLER_TO_CONTEXT: dict[str, str] = {
    # Context 1 — main chat loop uses chat_with_messages
    "run_reply_engine": "main_chat_turn",
    # Context 2 — intent judge (calls via internal helper)
    "IntentJudge.evaluate": "intent_judge",
    "IntentJudge._call_llm": "intent_judge",
    # Context 3 — evaluator
    "evaluate_turn": "evaluator",
    # Context 4 — memory enrichment extractor
    "extract_search_params_for_memory": "enrichment_extract",
    # Context 5 — memory digest (per batch)
    "_distil_batch": "memory_digest",
    "digest_memory_for_query": "memory_digest",
    # Context 6 — tool-result digest (per batch)
    "_distil_tool_batch": "tool_result_digest",
    "digest_tool_result_for_query": "tool_result_digest",
    "_maybe_digest_tool_result": "tool_result_digest",
    # Context 7 — max-turn loop digest
    "digest_loop_for_max_turns": "max_turn_digest",
    # Context 8 — tool router
    # (Context 9 — tool searcher — reuses select_tools_with_llm so it falls
    # under the same bucket; that's intentional per docs/llm_contexts.md.)
    "select_tools_with_llm": "tool_router",
    # Context 10 — conversation summariser
    "generate_conversation_summary": "summariser",
    # Context 11 — graph fact extraction
    "extract_graph_memories": "graph_extract",
    # Context 12 — graph best-child picker
    "_llm_pick_best_child": "graph_best_child",
    # Context 13 — tool-specific LLM calls
    "_extract_place_from_user_text": "tool_weather",
    "extract_and_log_meal": "tool_nutrition",
    "generate_followups_for_meal": "tool_nutrition",
}


@dataclass
class _Call:
    context: str
    duration_sec: float
    model: str
    prompt_chars: int
    response_chars: int
    provider: str = ""
    outcome: str = "success"


@dataclass
class TimingRecorder:
    calls: list[_Call] = field(default_factory=list)
    _originals: dict = field(default_factory=dict)

    def __enter__(self) -> "TimingRecorder":
        self._patch()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._unpatch()

    # ── context inference ────────────────────────────────────────────────
    @staticmethod
    def _infer_context(skip_frames: int = 2) -> str:
        """Walk the stack looking for the nearest function whose qualname is
        in our context map. Skip ``skip_frames`` to step over the wrapper
        itself. Falls back to ``"other:<qualname>"`` when no known caller is
        found — visible in the report so drift shows up."""
        frame = sys._getframe(skip_frames)
        first_unknown: Optional[str] = None
        while frame is not None:
            qual = frame.f_code.co_qualname if hasattr(frame.f_code, "co_qualname") else frame.f_code.co_name
            if qual in _CALLER_TO_CONTEXT:
                return _CALLER_TO_CONTEXT[qual]
            # Also match by the bare function name (qualname can be e.g.
            # ClassName.method — strip the class part).
            bare = qual.rsplit(".", 1)[-1]
            if bare in _CALLER_TO_CONTEXT:
                return _CALLER_TO_CONTEXT[bare]
            if first_unknown is None and not qual.startswith(("<", "_patch", "_unpatch")):
                first_unknown = qual
            frame = frame.f_back
        return f"other:{first_unknown or 'unknown'}"

    # ── patching ─────────────────────────────────────────────────────────
    def _wrap(self, name: str, original: Callable) -> Callable:
        def wrapped(*args, **kwargs):
            ctx = self._infer_context(skip_frames=2)
            backend = args[0]
            model = kwargs.get("chat_model") or (args[1] if len(args) > 1 else "")
            prompt_chars = 0
            if name == "chat":
                msgs = kwargs.get("messages") or (args[2] if len(args) > 2 else [])
                if isinstance(msgs, list):
                    prompt_chars = sum(len(str(m.get("content", ""))) for m in msgs)
            else:
                sys_p = kwargs.get("system_prompt") or (args[2] if len(args) > 2 else "")
                user_c = kwargs.get("user_content") or (args[3] if len(args) > 3 else "")
                prompt_chars = len(str(sys_p)) + len(str(user_c))

            t0 = time.perf_counter()
            result = None
            outcome = "error"
            try:
                result = original(*args, **kwargs)
                outcome = "success" if result is not None else "empty"
                return result
            finally:
                if isinstance(result, str):
                    response_chars = len(result)
                elif isinstance(result, dict):
                    response_chars = len(str(result.get("content", "")))
                else:
                    response_chars = 0
                self.calls.append(_Call(
                    context=ctx,
                    duration_sec=time.perf_counter() - t0,
                    model=str(model),
                    prompt_chars=prompt_chars,
                    response_chars=response_chars,
                    provider=type(backend).__name__,
                    outcome=outcome,
                ))

        return wrapped

    def _patch(self) -> None:
        """Patch backend classes so factory-created instances are captured."""
        self._originals["_sites"] = []
        for backend in (OllamaBackend, OpenAICompatibleBackend):
            for name in ("direct", "streaming", "chat"):
                original = getattr(backend, name)
                setattr(backend, name, self._wrap(name, original))
                self._originals["_sites"].append((backend, name, original))

    def _unpatch(self) -> None:
        for mod, name, original in self._originals.get("_sites", []):
            setattr(mod, name, original)
        self._originals.clear()

    # ── queries ──────────────────────────────────────────────────────────
    def by_context(self) -> dict[str, list[_Call]]:
        out: dict[str, list[_Call]] = {}
        for c in self.calls:
            out.setdefault(c.context, []).append(c)
        return out

    def durations(self, context: str) -> list[float]:
        return [c.duration_sec for c in self.calls if c.context == context]

    def p50(self, context: str) -> float:
        ds = self.durations(context)
        return statistics.median(ds) if ds else 0.0

    def p95(self, context: str) -> float:
        ds = self.durations(context)
        if not ds:
            return 0.0
        if len(ds) == 1:
            return ds[0]
        ds_sorted = sorted(ds)
        idx = max(0, int(round(0.95 * (len(ds_sorted) - 1))))
        return ds_sorted[idx]

    def total(self, context: Optional[str] = None) -> float:
        if context is None:
            return sum(c.duration_sec for c in self.calls)
        return sum(c.duration_sec for c in self.calls if c.context == context)

    # ── reporting ────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        buckets = self.by_context()
        return {
            "total_calls": len(self.calls),
            "total_sec": round(self.total(), 3),
            "contexts": {
                ctx: {
                    "calls": len(calls),
                    "total_sec": round(sum(c.duration_sec for c in calls), 3),
                    "p50_sec": round(self.p50(ctx), 3),
                    "p95_sec": round(self.p95(ctx), 3),
                    "avg_prompt_chars": int(statistics.mean(c.prompt_chars for c in calls)) if calls else 0,
                    "avg_response_chars": int(statistics.mean(c.response_chars for c in calls)) if calls else 0,
                    "models": sorted({c.model for c in calls if c.model}),
                }
                for ctx, calls in buckets.items()
            },
        }

    def print_report(self, title: str = "LLM pipeline timings") -> None:
        print(f"\n⏱️  {title}")
        print(f"   total calls: {len(self.calls)}   total wall time: {self.total():.2f}s")
        rows = sorted(
            self.by_context().items(),
            key=lambda kv: -sum(c.duration_sec for c in kv[1]),
        )
        header = f"   {'context':<22} {'n':>3}  {'total':>7}  {'p50':>6}  {'p95':>6}  {'prompt':>7}  model"
        print(header)
        print("   " + "-" * (len(header) - 3))
        for ctx, calls in rows:
            total = sum(c.duration_sec for c in calls)
            print(
                f"   {ctx:<22} {len(calls):>3}  "
                f"{total:>6.2f}s  {self.p50(ctx):>5.2f}s  {self.p95(ctx):>5.2f}s  "
                f"{int(statistics.mean(c.prompt_chars for c in calls)):>7}  "
                f"{','.join(sorted({c.model for c in calls if c.model}))}"
            )


@contextmanager
def record_timings():
    """Convenience context manager."""
    rec = TimingRecorder()
    with rec:
        yield rec
