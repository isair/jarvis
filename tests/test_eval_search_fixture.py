"""Unrelated tool outcomes cannot stand in for a search result."""
import pytest

from evals.test_distinct_tool_calls import _record_fixture_search, _SEARCH_ARGUMENT

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('tool_name', ['stop', 'toolSearchTool'])
def test_unrelated_tool_cannot_produce_a_fixture_weather_reading(tool_name):
    completed_searches = set()
    result = _record_fixture_search({'London': 12}, completed_searches, tool_name,
                                    {_SEARCH_ARGUMENT: 'London weather'})
    assert result.success is False
    assert not completed_searches
