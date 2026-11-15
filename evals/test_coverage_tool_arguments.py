"""Native recovery calls preserve literal selectors and resolved query intent."""
import pytest

from evals.helpers import isolated_voice_settings
from evals.test_compound_result_guidance import OwnedFilmLookup
from evals.test_text_tool_schema import OwnedEntryManager
from jarvis.reply.compound_query import resolve_uncovered_request
from jarvis.tools import registry

pytestmark = pytest.mark.eval


@pytest.mark.parametrize('query,entries,operation,scope,remaining', [
    ('List all my reminder entries.', {'tea', 'soup'}, 'enumerate_entries', 'whole_collection', {'tea', 'soup'}),
    ('Delete every reminder entry.', {'tea', 'soup'}, 'discard_entries', 'whole_collection', set()),
    ('Delete only the tea reminder and keep the soup reminder.', {'tea', 'soup'}, 'discard_entries', 'matching_labels', {'soup'}),
    ('Delete only the entry named "tea reminder" and keep the entry named "soup reminder".', {'tea reminder', 'soup reminder'}, 'discard_entries', 'matching_labels', {'soup reminder'}),
    ('Delete only the entry named tea reminder and keep the entry named soup reminder.', {'tea reminder', 'soup reminder'}, 'discard_entries', 'matching_labels', {'soup reminder'}),
])
def test_recovery_selector_arguments_produce_the_requested_effect(
        monkeypatch, query, entries, operation, scope, remaining):
    tool = OwnedEntryManager()
    tool.entries = set(entries)
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    schemas = registry.generate_tools_json_schema([tool.name, 'stop', 'toolSearchTool'])
    proposal = resolve_uncovered_request(isolated_voice_settings(), query, [], schemas)
    assert proposal is not None, 'An unresolved available action needs a concrete recovery proposal'
    assert proposal['name'] == tool.name
    assert proposal['arguments']['operation'] == operation
    assert proposal['arguments']['scope'] == scope
    result = tool.run(proposal['arguments'], None)
    assert result.success
    assert len(tool.calls) == 1
    assert tool.entries == remaining


def test_recovery_query_uses_the_discovered_entity(monkeypatch):
    tool = OwnedFilmLookup()
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    first_args = {'query': 'Who directed Possessor?'}
    first = tool.run(first_args, None)
    schemas = registry.generate_tools_json_schema([tool.name, 'stop', 'toolSearchTool'])
    proposal = resolve_uncovered_request(
        isolated_voice_settings(),
        'Who directed Possessor and what other films has that director made?',
        [{'name': tool.name, 'arguments': first_args, 'result': first.reply_text, 'success': first.success}],
        schemas,
    )
    assert proposal is not None
    assert proposal['name'] == tool.name
    query = proposal['arguments']['query'].casefold()
    assert 'brandon' in query and 'cronenberg' in query
    result = tool.run(proposal['arguments'], None)
    assert result.success
    assert len(tool.calls) == 2
    assert 'Antiviral' in result.reply_text and 'Infinity Pool' in result.reply_text
