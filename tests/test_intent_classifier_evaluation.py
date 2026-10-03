"""Behavioural tests for the local intent-classifier qualification harness."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from evals.intent_classifier import classify_intent

pytestmark = pytest.mark.unit


@pytest.fixture
def classifier_server():
    class Handler(BaseHTTPRequestHandler):
        status = 200
        response = {
            "answers": {
                "directed": {"type": "noul", "noul": 0.99},
                "stop": {"type": "noul", "noul": 0.01},
            },
            "usage": {"truncated": False},
        }
        requests = []
        delay = 0
        raw_body = None

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.requests.append((self.path, body))
            time.sleep(self.delay)
            self.send_response(self.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Location", "https://example.com/")
            self.end_headers()
            try:
                self.wfile.write(self.raw_body or json.dumps(self.response).encode())
            except BrokenPipeError:
                pass

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", Handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def classify(server, **kwargs):
    return classify_intent(
        "Transcript: Jarvis what time is it", base_url=server,
        model="test-model", assistant_name="Jarvis", **kwargs,
    )


def test_confident_directed_speech_is_accepted(classifier_server):
    url, handler = classifier_server
    result = classify(url)
    assert (result.directed, result.stop) == (True, False)
    path, payload = handler.requests[-1]
    assert path == "/v1/systemone"
    assert payload["state"] == "Transcript: Jarvis what time is it"
    assert payload["model"] == "test-model"
    assert set(payload["questions"]) == {"directed", "stop"}


@pytest.mark.parametrize("directed,stop,expected", [
    (0.99, 0.99, (True, True)),
    (0.01, 0.01, (False, False)),
])
def test_stop_and_ambient_decisions(classifier_server, directed, stop, expected):
    url, handler = classifier_server
    handler.response["answers"]["directed"]["noul"] = directed
    handler.response["answers"]["stop"]["noul"] = stop
    result = classify(url)
    assert (result.directed, result.stop) == expected


@pytest.mark.parametrize("probability", [0.5, 0.89, 0.11])
def test_uncertain_scores_abstain(classifier_server, probability):
    url, handler = classifier_server
    handler.response["answers"]["directed"]["noul"] = probability
    assert classify(url, threshold=0.9) is None


def test_threshold_controls_abstention(classifier_server):
    url, handler = classifier_server
    handler.response["answers"]["directed"]["noul"] = 0.8
    assert classify(url, threshold=0.9) is None
    assert classify(url, threshold=0.75).directed is True


def test_stop_cannot_be_undirected(classifier_server):
    url, handler = classifier_server
    handler.response["answers"]["directed"]["noul"] = 0.01
    handler.response["answers"]["stop"]["noul"] = 0.99
    with pytest.raises(ValueError, match="contradictory"):
        classify(url)


@pytest.mark.parametrize("value", [True, None, "0.99", -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_probabilities_are_failures(classifier_server, value):
    url, handler = classifier_server
    handler.response["answers"]["directed"]["noul"] = value
    with pytest.raises(ValueError):
        classify(url)


@pytest.mark.parametrize("response", [
    {}, [], {"answers": {}},
    {"answers": {"directed": {"type": "choice", "noul": 0.99}}},
    {"answers": {"directed": {"type": "noul", "noul": 0.99}}},
])
def test_incomplete_answers_are_failures(classifier_server, response):
    url, handler = classifier_server
    handler.response = response
    with pytest.raises(ValueError):
        classify(url)


@pytest.mark.parametrize("usage", [
    {"truncated": True}, {"state_tokens_dropped": 1},
    {"truncated_questions": ["stop"]},
])
def test_truncated_context_is_a_failure(classifier_server, usage):
    url, handler = classifier_server
    handler.response["usage"] = usage
    with pytest.raises(ValueError, match="truncated"):
        classify(url)


@pytest.mark.parametrize("status", [302, 401, 503])
def test_http_errors_and_redirects_are_failures(classifier_server, status):
    url, handler = classifier_server
    handler.status = status
    with pytest.raises(ValueError, match="HTTP"):
        classify(url)


@pytest.mark.parametrize("url", [
    "https://api.typesafe.ai", "http://example.com", "http://localhost.example.com",
    "http://user:secret@127.0.0.1", "file:///tmp/model", "http://127.0.0.1/?key=secret",
])
def test_only_local_endpoints_are_allowed(url):
    with pytest.raises(ValueError, match="loopback"):
        classify(url)


def test_environment_proxy_cannot_receive_speech(classifier_server, monkeypatch):
    url, _ = classifier_server
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    assert classify(url).directed is True


def test_non_json_answer_is_a_failure(classifier_server):
    url, handler = classifier_server
    handler.raw_body = b"model unavailable"
    with pytest.raises(ValueError):
        classify(url)


def test_timeout_is_a_failure(classifier_server):
    url, handler = classifier_server
    handler.delay = 0.1
    with pytest.raises(requests.exceptions.Timeout):
        classify(url, timeout_sec=0.02)


def test_top_level_truncation_is_a_failure(classifier_server):
    url, handler = classifier_server
    handler.response["truncated"] = True
    with pytest.raises(ValueError, match="truncated"):
        classify(url)


@pytest.mark.parametrize("threshold", [0.5, 1.01, float("nan")])
def test_invalid_confidence_threshold_is_rejected(classifier_server, threshold):
    url, _ = classifier_server
    with pytest.raises(ValueError, match="threshold"):
        classify(url, threshold=threshold)
