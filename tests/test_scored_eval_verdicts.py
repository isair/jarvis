"""Unavailable or malformed verification cannot satisfy scored eval checks."""
from unittest.mock import patch

import pytest

from evals import helpers
from evals.test_knowledge_extraction import _judge_extraction_quality

pytestmark = pytest.mark.unit

_CASES = [
    (_judge_extraction_quality,
     ('My shed access code is 6932.', ['The user\'s shed access code is 6932.']),
     ('NOVELTY', 'SELF_CONTAINED', 'NO_ASSISTANT_VOICE', 'NO_STALE_DATA', 'COMPLETENESS')),
    (helpers.judge_response_answers_query,
     ('What is the weather?', 'The weather is twelve degrees Celsius and clear, ideal for a short walk.'),
     ('RELEVANCE', 'COMPLETENESS', 'ACCURACY', 'NO_DEFLECTION')),
    (helpers.judge_search_query_quality,
     ('Weather in London?', 'London weather today', 'London'),
     ('INTENT_MATCH', 'LOCATION_AWARENESS', 'TIME_AWARENESS', 'SPECIFICITY')),
    (helpers.judge_tool_usage_appropriateness,
     ('Read report.txt.', ['localFiles'], [{'operation': 'read', 'path': 'report.txt'}]),
     ('TOOL_SELECTION', 'ARG_QUALITY', 'EFFICIENCY')),
]


@pytest.mark.parametrize('judge,args,criteria', _CASES)
@pytest.mark.parametrize('output', [None, '', ' ', 'OVERALL: NOT PASS', 'OVERALL: PASS'])
def test_incomplete_verification_remains_unknown(judge, args, criteria, output):
    with patch.object(helpers, 'call_judge_llm', return_value=output):
        result = judge(*args)
    assert not result.is_passed
    assert result.score == 0
    assert not result.is_verified
    assert not result.criteria_scores
    assert 'incomplete' in result.reasoning.casefold()


@pytest.mark.parametrize('judge,args,criteria', _CASES)
@pytest.mark.parametrize('overall', ['PASS', 'FAIL'])
def test_complete_verdict_preserves_measured_scores(judge, args, criteria, overall):
    scores = {name: i + 5 for i, name in enumerate(criteria)}
    output = '\n'.join(f'{name}: {score}' for name, score in scores.items())
    output += f'\nOVERALL: {overall}\nREASONING: Recorded evidence assessed.'
    with patch.object(helpers, 'call_judge_llm', return_value=output):
        result = judge(*args)
    assert result.is_passed == (overall == 'PASS')
    assert result.is_verified
    assert result.criteria_scores == {name.casefold(): score / 10 for name, score in scores.items()}
    assert result.score == pytest.approx(sum(scores.values()) / len(scores) / 10)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'contradictory', 'out-of-range', 'non-finite', 'no-reasoning', 'unknown-verdict', 'explained-verdict'])
def test_invalid_scored_output_cannot_keep_a_passing_score(mutation):
    judge, args, criteria = _CASES[0]
    rows = [f'{name}: 9' for name in criteria]
    rows += ['OVERALL: PASS', 'REASONING: Recorded evidence assessed.']
    if mutation == 'missing':
        rows.pop(0)
    elif mutation == 'duplicate':
        rows.append(rows[0])
    elif mutation == 'contradictory':
        rows.append('OVERALL: FAIL')
    elif mutation == 'out-of-range':
        rows[0] = f'{criteria[0]}: 11'
    elif mutation == 'non-finite':
        rows[0] = f'{criteria[0]}: nan'
    elif mutation == 'no-reasoning':
        rows.pop()
    else:
        rows[-2] = 'OVERALL: NOT PASS' if mutation == 'unknown-verdict' else 'OVERALL: PASS because it looks fine'
    with patch.object(helpers, 'call_judge_llm', return_value='\n'.join(rows)):
        result = judge(*args)
    assert not result.is_passed and not result.is_verified
    assert result.score == 0 and not result.criteria_scores
