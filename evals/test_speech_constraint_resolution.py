"""The current request supplies constraints absent from a shortened plan."""
import pytest
from evals.query_context_comparison import comparison_settings
from jarvis.reply.planner import resolve_next_tool_call


@pytest.mark.eval
@pytest.mark.parametrize('query,count', [('Pick three items', 3), ('Pick five items', 5),
                                         ('Üç öğe seç', 3), ('Beş öğe seç', 5)])
def test_step_resolution_uses_the_current_request_count(query, count):
    cfg = comparison_settings('http://127.0.0.1:11434', 'gemma4:e4b',
                              fast_model='gemma4:e2b', provider='ollama')
    schema = [{'type': 'function', 'function': {'name': 'selectItems',
              'description': 'Select the requested number of items.',
              'parameters': {'type': 'object', 'properties': {
                  'count': {'type': 'integer', 'description': 'Number of items to select.'}},
                  'required': ['count']}}}]
    result = resolve_next_tool_call(cfg, 'selectItems count=<requested number>', [], schema,
                                    original_query=query)
    assert result == ('selectItems', {'count': count})
