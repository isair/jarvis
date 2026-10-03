"""Relevance evals reject dropped evidence and invented personal preferences."""
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import requests

from jarvis.reply import enrichment

pytestmark = pytest.mark.unit


def load_cases(monkeypatch, result):
    monkeypatch.setattr(requests, 'get', lambda *args, **kwargs: SimpleNamespace(status_code=404))
    monkeypatch.setattr(enrichment, 'digest_memory_for_query', lambda **kwargs: result)

    def unexpected_post(*args, **kwargs):
        pytest.fail('🧠 Relevance guard checks must not contact a live model')

    monkeypatch.setattr(requests, 'post', unexpected_post)
    path = Path(__file__).resolve().parents[1] / 'evals' / 'test_memory_digest_relevance.py'
    return runpy.run_path(str(path))


def run_case(namespace, index):
    namespace['test_memory_digest_preserves_query_specific_relevance'](*namespace['CASES'][index])


@pytest.mark.parametrize('index', [0, 1, 2, 3, 6])
@pytest.mark.parametrize('result', ['', None, 'The assistant discussed an unrelated topic.'])
def test_positive_relevance_eval_rejects_missing_evidence(monkeypatch, index, result):
    with pytest.raises(AssertionError):
        run_case(load_cases(monkeypatch, result), index)


@pytest.mark.parametrize('index', [4, 5])
def test_negative_relevance_eval_rejects_unrelated_memory(monkeypatch, index):
    with pytest.raises(AssertionError):
        run_case(load_cases(monkeypatch, 'The user asked about Harbour Notes.'), index)


@pytest.mark.parametrize('index, result', [
    (0, 'The user prefers sourdough bagels.'),
    (1, 'The user loves Harbour Notes.'),
    (2, 'The user likes Amber Choir.'),
    (3, "The user's favourite dish is mercimek soup."),
    (6, 'Harbour Notes was written by Ada North.'),
    (6, 'The assistant said Harbour Notes was written by Someone Else.'),
])
def test_relevance_eval_rejects_fabrication_or_lost_attribution(monkeypatch, index, result):
    with pytest.raises(AssertionError):
        run_case(load_cases(monkeypatch, result), index)


@pytest.mark.parametrize('index, result', [
    (0, 'The user asked about sourdough bagels.'),
    (1, 'The user asked about Harbour Notes.'),
    (2, 'The user listened to Amber Choir.'),
    (3, 'Kullanıcı mercimek çorbası tarifini sordu.'),
    (4, ''),
    (5, ''),
    (6, 'The assistant said the author of Harbour Notes is Ada North.'),
])
def test_relevance_eval_accepts_grounded_output(monkeypatch, index, result):
    run_case(load_cases(monkeypatch, result), index)
