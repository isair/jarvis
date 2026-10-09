"""Memory extraction and hygiene have distinct observable timing contexts."""
import json
from unittest.mock import patch

import pytest

from jarvis.memory import graph_ops
from tests.performance.timing_recorder import TimingRecorder

pytestmark = pytest.mark.unit


def test_graph_extraction_and_review_have_separate_timings(mock_config):
    responses = iter([
        json.dumps([{'branch': 'USER', 'fact': 'The user has a cat named Miso'}]),
        '{"0": "DURABLE"}',
    ])
    with patch.object(graph_ops, 'call_llm_direct', side_effect=lambda **kwargs: next(responses)):
        with TimingRecorder() as recorder:
            facts = graph_ops.extract_graph_memories('Summary', mock_config, 'local-model')
        assert facts == [('user', 'The user has a cat named Miso')]
        assert set(recorder.by_context()) == {'graph_extract', 'graph_fact_hygiene'}
