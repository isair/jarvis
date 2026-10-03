# Local intent classifier qualification

The voice intent judge has two responsibilities: classify speech (`directed`
and `stop`), and synthesise a clean, self-contained query from the transcript.
Typed decision models can supply the first part. They cannot generate a query
or resolve a reference into a new string. The runtime judge uses the FAST-tier
LLM for the full contract.

[Jev](https://docs.typesafe.ai/introduction) exposes atomic typed decisions.
This evaluation uses the self-hostable System One protocol implemented by
[Laya](https://github.com/NandhaKishorM/laya). It permits loopback endpoints only,
disables environment proxies and redirects, and sends no credentials. Jarvis
has no dependency on Laya or a proprietary classification service.

## Run

Install the classifier server in its own environment so its Torch and
Transformers requirements do not affect Jarvis. For the evaluated server:

```bash
python -m venv /tmp/jarvis-intent-model
/tmp/jarvis-intent-model/bin/pip install 'laya[serve]==0.3.24'
LAYA_HOST=127.0.0.1 LAYA_PORT=18000 LAYA_MODELS=multilingual LAYA_DEVICE=cpu \
  LAYA_REVISION=55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851 \
  /tmp/jarvis-intent-model/bin/laya-serve
```

The first server start downloads the model weights (about 647 MB for the
multilingual checkpoint); cached inference runs locally. The evaluation uses
synthetic speech fixtures, not the user's transcript or audio.

From the activated Jarvis development environment:

```bash
EVAL_INTENT_CLASSIFIER_URL=http://127.0.0.1:18000 \
  python -m pytest evals/test_intent_classifier.py -v \
  -o junit_family=xunit1 --junitxml=/tmp/intent-classifier-results.xml
```

`EVAL_INTENT_CLASSIFIER_MODEL` defaults to `multilingual`.
`EVAL_INTENT_CLASSIFIER_THRESHOLD` defaults to `0.9`: both probabilities must
support either yes or no at that confidence. Uncertain scores abstain and fail
qualification. The URL is opt-in; an unset URL skips these live evals, while an
unavailable configured server fails. No LLM fallback masks classifier failures.

## Contract and coverage

Each case uses `IntentJudge._build_user_prompt` to supply the same transcript,
timestamps, current-segment markers, TTS context, hot-window state and alias
normalisation as the listener. Two independent `noul` questions ask whether
the current speech is directed and whether it is a stop command. A stop verdict
without directedness is invalid. Incomplete answers, non-finite or out-of-range
probabilities, and reported input truncation fail the case.

The 54 cases comprise 42 existing single- and multi-segment intent cases and 12
additional cases covering Spanish, Turkish, French and Japanese speech,
quoted stops, the word "stop" in ordinary questions and pure TTS echo. The
suite asserts both decisions on every case. The existing LLM intent evals
remain responsible for query extraction, wake-word removal and topic synthesis.

## Candidate evidence

| Candidate | Environment | Qualification |
|-----------|-------------|---------------|
| Laya multilingual, 322M parameters | Laya 0.3.24, CPU, Apple M5 Max | 26/54 pass, 24 abstentions, 4 confident errors |

The model bundle is `convaiinnovations/laya`, revision
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`. The confidence threshold is `0.9`.
The confident errors comprise three missed stops and one narrative mention
classified as directed. Median request latency is 72.0 ms, with a maximum of
126.4 ms across these 54
cases. These measurements cover local HTTP inference with cached weights;
they exclude server startup and downloading. The candidate does not qualify
for runtime use. There is no measured accuracy comparison with the FAST-tier
LLM, and these classification results do not establish end-to-end query quality.

Qualification requires every declared case to pass without fallback, followed
by the listener and query-synthesis evals for any runtime integration. Passing
this finite suite is necessary evidence, rather than a guarantee for arbitrary
speech or languages.
