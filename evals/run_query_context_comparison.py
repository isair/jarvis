"""Run the local query/context comparison and save complete synthetic traces."""

import argparse
import dataclasses
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from evals.query_context_cases import CASES
from evals.query_context_comparison import (
    ARMS, RequestRecorder, comparison_settings, run_case, score_result, summarise,
)


def run_comparison(cfg, output: Path, *, repeats=3, cases=CASES, no_thinking=True):
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    RequestRecorder(cfg.llm_base_url, no_thinking=no_thinking)
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": cfg.llm_base_url, "chat_model": cfg.llm_chat_model,
        "fast_model": cfg.fast_model, "repeats": repeats,
        "temperature": 0, "template_enable_thinking": False if no_thinking else "server default",
        "arms": list(ARMS), "case_count": len(cases),
        "classifier_included": False,
        "tool_results": "synthetic", "order": "rotating arms by case and repeat",
        "planner_timeout_sec": cfg.planner_timeout_sec,
        "intent_judge_timeout_sec": cfg.intent_judge_timeout_sec,
        "empty_diary_lookup": "stubbed; no persisted memory or embeddings",
    }
    rows = []
    output.parent.mkdir(parents=True, exist_ok=True)

    def save(status):
        output.write_text(json.dumps({
            "metadata": dict(metadata, status=status),
            "summary": summarise(rows), "rows": rows,
        }, ensure_ascii=False, indent=2) + "\n")

    save("running")
    for repeat in range(repeats):
        for index, case in enumerate(cases):
            offset = (index + repeat) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                result = run_case(case, arm, repeat, cfg, no_thinking=no_thinking)
                row = dict(dataclasses.asdict(result), **score_result(case, result))
                rows.append(row)
                save("running")
                icon = "✅" if row["passed"] else "❌"
                print(f"{icon} {case.name} / {arm} / repeat {repeat + 1}: "
                      f"{result.latency_ms / 1000:.2f}s, "
                      f"routing={row['routing_correct']}, arguments={row['arguments_correct']}, "
                      f"answer={row['answer_correct']}, model={row['model_valid']}"
                      + (f", error={result.error}" if result.error else ""), flush=True)
    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    save("complete")
    print(f"📊 Results saved to {output}", flush=True)
    for arm, summary in summarise(rows).items():
        print(f"  🧪 {arm}: {summary['passed']}/{summary['runs']} pass, "
              f"median {summary['median_latency_ms'] / 1000:.2f}s", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Loopback OpenAI-compatible base URL, including /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--fast-model", default="", help="Rewrite/router model; defaults to the chat model")
    parser.add_argument("--api-key-env", default="EVAL_COMPARISON_API_KEY", help="Environment variable name, never a literal credential")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--planner-timeout", type=float, help="Exploratory timeout override, recorded in the report")
    parser.add_argument("--judge-timeout", type=float, help="Exploratory timeout override, recorded in the report")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[], help="Optional case names")
    parser.add_argument("--server-thinking", action="store_true", help="Use server-default reasoning instead of enable_thinking=false")
    args = parser.parse_args()
    cases = tuple(case for case in CASES if not args.case or case.name in args.case)
    unknown = set(args.case) - {case.name for case in CASES}
    if unknown:
        parser.error(f"Unknown cases: {', '.join(sorted(unknown))}")
    cfg = comparison_settings(args.base_url, args.model, fast_model=args.fast_model,
                              api_key=os.environ.get(args.api_key_env, ""))
    overrides = {}
    for field, value in (("planner_timeout_sec", args.planner_timeout),
                         ("intent_judge_timeout_sec", args.judge_timeout)):
        if value is not None:
            if value <= 0:
                parser.error("Timeouts must be positive")
            overrides[field] = value
    cfg = dataclasses.replace(cfg, **overrides)
    run_comparison(cfg, args.output, repeats=args.repeats, cases=cases,
                   no_thinking=not args.server_thinking)


if __name__ == "__main__":
    main()
