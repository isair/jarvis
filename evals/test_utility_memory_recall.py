"""Utility questions avoid history while personal and episodic recall survives."""
import pytest

from evals.helpers import voice_config
from evals.tool_routing import requires_judge_llm
from jarvis.reply.enrichment import extract_search_params_for_memory

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize('query', [
    'what time is it right now',
    'what is 17 multiplied by 23',
    'Define photosynthesis',
    '¿Qué significa fotosíntesis?',
    'On yedi ile yirmi üçün çarpımı nedir?',
])
def test_utility_query_has_no_memory_search_parameters(query):
    cfg = voice_config()
    parameters = extract_search_params_for_memory(query, cfg, cfg.fast_model, timeout_sec=15.0)
    assert parameters, 'Unavailable inference is not a successful skip decision'
    assert parameters.get('keywords') == [], parameters
    assert parameters.get('questions') == [], parameters
    assert not parameters.get('from') and not parameters.get('to'), parameters


@pytest.mark.parametrize('query, personal', [
    ('What did we discuss about cooking?', False),
    ('What did we discuss about photosynthesis yesterday?', False),
    ('What news might interest me?', True),
    ('¿Qué te conté sobre mis alergias?', False),
])
def test_history_dependent_query_retains_memory_search_parameters(query, personal):
    cfg = voice_config()
    parameters = extract_search_params_for_memory(query, cfg, cfg.fast_model, timeout_sec=15.0)
    assert parameters.get('keywords'), parameters
    if personal:
        assert parameters.get('questions'), parameters
    if 'yesterday' in query:
        assert parameters.get('from') and parameters.get('to'), parameters
