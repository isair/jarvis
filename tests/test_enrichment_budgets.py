"""Retries and digest batches share a finite execution budget."""
import threading
from unittest.mock import Mock

import pytest
from jarvis.reply import enrichment
from jarvis.reply.execution import ExecutionControl, ExecutionCancelled

pytestmark = pytest.mark.unit


def test_extractor_does_not_retry_after_cancellation(monkeypatch, mock_config):
    cancelled = threading.Event()
    control = ExecutionControl(cancelled, 10)
    def first(**kwargs):
        cancelled.set()
        return "invalid JSON"
    backend = Mock(side_effect=first)
    monkeypatch.setattr(enrichment, "call_llm_direct", backend)
    with pytest.raises(ExecutionCancelled):
        enrichment.extract_search_params_for_memory("old facts", mock_config, "local",
                                                      control=control)
    assert backend.call_count == 1


def test_digest_batches_stop_after_shared_deadline(monkeypatch, mock_config):
    now = [100.0]
    monkeypatch.setattr(enrichment, "_monotonic", lambda: now[0], raising=False)
    def distil(*args, **kwargs):
        now[0] += 6
        return "A grounded fact."
    backend = Mock(side_effect=distil)
    monkeypatch.setattr(enrichment, "_distil_batch", backend)
    result = enrichment.digest_memory_for_query("history", ["a" * 1900] * 3, [],
                                                mock_config, "local", timeout_sec=5)
    assert backend.call_count == 1
    assert result == "A grounded fact."
