"""Local System One client for intent classification qualification.

This client returns decisions only. Reference resolution belongs to the
downstream reply pipeline, which receives the original speech and context.
"""

import ipaddress
import math
from dataclasses import dataclass
from urllib.parse import urlsplit

import requests

from jarvis.debug import debug_log


@dataclass(frozen=True)
class IntentDecision:
    directed: bool
    stop: bool


def _local_endpoint(base_url: str) -> str:
    parsed = urlsplit(base_url)
    try:
        local = parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname).is_loopback
    except (ValueError, TypeError):
        local = False
    if (
        not local or parsed.scheme not in {"http", "https"}
        or parsed.username is not None or parsed.password is not None
        or parsed.query or parsed.fragment
    ):
        raise ValueError("The classifier must use a loopback endpoint without credentials or a query")
    return base_url.rstrip("/") + "/v1/systemone"


def classify_intent(
    state: str,
    *,
    base_url: str,
    model: str,
    assistant_name: str,
    threshold: float = 0.9,
    timeout_sec: float = 3.0,
) -> IntentDecision | None:
    """Classify full judge context, abstaining when either answer is uncertain.

    Malformed, contradictory, truncated and unavailable responses raise errors.
    The eval must count these as failures rather than invoke an LLM fallback.
    """
    if not math.isfinite(threshold) or not 0.5 < threshold <= 1:
        raise ValueError("threshold must be finite and in (0.5, 1]")
    if not math.isfinite(timeout_sec) or timeout_sec <= 0:
        raise ValueError("timeout must be positive and finite")
    endpoint = _local_endpoint(base_url)
    payload = {
        "model": model,
        "state": state,
        "max_len": 8192,
        "questions": {
            "directed": {
                "type": "noul",
                "instructions": (
                    f"The current speech is directed at {assistant_name}. "
                    "Addressed questions, commands and statements count. "
                    "Hot-window follow-ups count. A narrative mention or TTS echo does not. "
                    "Stop commands count as directed."
                ),
            },
            "stop": {
                "type": "noul",
                "instructions": "The current speaker is asking the assistant to stop speaking.",
            },
        },
    }
    with requests.Session() as session:
        # Speech stays local even when the environment configures a proxy.
        session.trust_env = False
        response = session.post(
            endpoint, json=payload, timeout=timeout_sec, allow_redirects=False,
        )
    if response.status_code != 200:
        raise ValueError(f"Classifier HTTP {response.status_code}")
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Classifier response must be an object")
    usage = data.get("usage", {})
    if not isinstance(usage, dict):
        raise ValueError("Invalid classifier usage")
    for metadata in (data, usage):
        if any(metadata.get(key) for key in (
            "truncated", "state_tokens_dropped", "truncated_questions",
        )):
            raise ValueError("Classifier context was truncated")
    answers = data.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("Missing classifier answers")
    scores = {}
    for key in payload["questions"]:
        answer = answers.get(key)
        if not isinstance(answer, dict) or answer.get("type") != "noul":
            raise ValueError(f"Missing or invalid {key} answer")
        score = answer.get("noul")
        if (
            isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score) or not 0 <= score <= 1
        ):
            raise ValueError(f"Invalid {key} probability")
        scores[key] = score
    if any(max(score, 1 - score) < threshold for score in scores.values()):
        debug_log("🧪 Intent classifier: abstained on uncertain scores", "voice")
        return None
    directed, stop = scores["directed"] > 0.5, scores["stop"] > 0.5
    if stop and not directed:
        raise ValueError("Classifier returned contradictory undirected stop")
    debug_log(f"🧪 Intent classifier: directed={directed}, stop={stop}", "voice")
    return IntentDecision(directed=directed, stop=stop)
