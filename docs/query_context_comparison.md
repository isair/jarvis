# Speech-reference evaluation

The replay compares three paths through production routing, planning and replies:

- `rewrite`: an evaluation-only frozen query-rewriting judge.
- `raw_only`: recognised speech without ambient references.
- `raw_context`: recognised speech with the immutable redacted speech snapshot.

Each run uses isolated settings, fresh in-memory dialogue and synthetic tool results. No user diary, graph, settings or external tools are read or changed. Endpoints must be loopback; proxy inheritance and redirects are disabled. Credentials remain in memory and are excluded from reports.

The scorer checks tool arguments against declared schemas, expected referents, grounded answers and complete model responses. Every attempted call must match the requested operation. Multiple actions fail the single-operation fixtures, including wrong-then-right and duplicate writes. Routing is reported separately from completed results. Failed and timed-out runs remain in denominators.

Run from the activated project environment:

```sh
PYTHONPATH=src:.:evals python -m evals.run_query_context_comparison \
  --provider ollama --base-url http://127.0.0.1:11434 --model gemma4:e4b \
  --fast-model gemma4:e2b --repeats 3 --output /tmp/speech-reference.json
```

Production settings supply deadlines unless an explicit exploratory override is requested; overrides are recorded in the report. The replay controls sampling uniformly across arms, and records each request, its phase, deadline, output cap, response validity, token counts and latency. It evaluates synthetic recognised speech, not microphone transcription or real acoustic echo. Intent decisions require their own directedness/cancellation evals.

Native Ollama replay uses `/api/chat` and its native thinking control. OpenAI-compatible replay uses `/chat/completions` and the server-specific template control; a server may ignore that extension, so it does not establish native Ollama reasoning behaviour.
