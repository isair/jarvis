"""Max-turn live evals reject missing findings and late incompletion caveats."""
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import requests

from jarvis.reply import enrichment

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('case_index', range(3))
@pytest.mark.parametrize('failure', ['missing', 'ungrounded', 'uncaveated', 'late'])
def test_max_turn_eval_rejects_unusable_partial_reply(monkeypatch, case_index, failure):
    monkeypatch.setattr(requests, 'get', lambda *args, **kwargs: SimpleNamespace(status_code=404))
    holder = {'reply': None}
    monkeypatch.setattr(enrichment, 'digest_loop_for_max_turns', lambda *args, **kwargs: holder['reply'])
    def unexpected_post(*args, **kwargs):
        pytest.fail('⏳ Eval guard checks must not contact a live model')
    monkeypatch.setattr(requests, 'post', unexpected_post)
    path = Path(__file__).resolve().parents[1] / 'evals' / 'test_max_turn_digest.py'
    case = runpy.run_path(str(path))
    query, name, data, facts, caveats = case['CASES'][case_index]
    text = ' '.join(facts) + '.'
    holder['reply'] = None if failure == 'missing' else text
    if failure == 'ungrounded':
        holder['reply'] = caveats[0] + ' finish.'
    if failure == 'late':
        holder['reply'] += ' ' + caveats[0] + ' finish.'
    with pytest.raises(AssertionError):
        case['test_max_turn_digest_preserves_partial_findings'](query, name, data, facts, caveats)
