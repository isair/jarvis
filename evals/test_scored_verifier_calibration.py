"""Scored verification recognises correct evidence and explicit failures."""
import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import (
    judge_response_answers_query, judge_search_query_quality,
    judge_tool_usage_appropriateness,
)
from evals.test_knowledge_extraction import _judge_extraction_quality

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize('judge,args,expected', [
    (judge_response_answers_query,
     ('What is the temperature in London?', 'London is 12 C and clear.', 'Recorded London reading: 12 C, clear.'), True),
    (judge_response_answers_query,
     ('What is the temperature in London?', 'How can I help you today?', 'Recorded London reading: 12 C, clear.'), False),
    (judge_search_query_quality, ('What is the weather in London?', 'London current weather', 'London'), True),
    (judge_search_query_quality, ('What is the weather in London?', 'how to bake a chocolate cake', 'London'), False),
    (judge_tool_usage_appropriateness,
     ('Read report.txt.', ['localFiles'], [{'operation': 'read', 'path': 'report.txt'}], ['localFiles']), True),
    (judge_tool_usage_appropriateness,
     ('Read report.txt.', ['getWeather'], [{}], ['localFiles']), False),
    (_judge_extraction_quality, ('My shed access code is 6932.', ['The user\'s shed access code is 6932.']), True),
    (_judge_extraction_quality, ('My shed access code is 6932.', ['Paris is the capital of France.']), False),
])
def test_complete_scored_verifier_distinguishes_recorded_outcomes(judge, args, expected):
    result = judge(*args)
    assert result.is_verified, result.reasoning
    assert result.is_passed == expected, result
