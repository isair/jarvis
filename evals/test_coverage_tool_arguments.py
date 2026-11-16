"""Native recovery calls preserve literal selectors and resolved query intent."""
import pytest

from evals.helpers import isolated_voice_settings
from evals.test_compound_result_guidance import OwnedFilmLookup
from evals.test_text_tool_schema import OwnedEntryManager, observed_entry_names
from jarvis.reply.compound_query import resolve_uncovered_request
from jarvis.tools import registry

pytestmark = pytest.mark.eval


@pytest.mark.parametrize('query,entries,operation,scope,remaining,known_namespace', [
    ('List all my reminder entries.', {'tea', 'soup'}, 'enumerate_entries', 'whole_collection', {'tea', 'soup'}, False),
    ('Delete every reminder entry.', {'tea', 'soup'}, 'discard_entries', 'whole_collection', set(), False),
    ('Delete only the tea reminder and keep the soup reminder.', {'tea', 'soup'}, 'discard_entries', 'matching_labels', {'soup'}, True),
    ('Delete only the entry named "tea reminder" and keep the entry named "soup reminder".', {'tea reminder', 'soup reminder'}, 'discard_entries', 'matching_labels', {'soup reminder'}, False),
    ('Delete only the entry named tea reminder and keep the entry named soup reminder.', {'tea reminder', 'soup reminder'}, 'discard_entries', 'matching_labels', {'soup reminder'}, False),
])
def test_recovery_selector_arguments_produce_the_requested_effect(
        monkeypatch, query, entries, operation, scope, remaining, known_namespace):
    tool = OwnedEntryManager()
    tool.entries = set(entries)
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    schemas = registry.generate_tools_json_schema([tool.name, 'stop', 'toolSearchTool'])
    prior = [observed_entry_names(tool)] if known_namespace else []
    tool.calls.clear()
    proposal = resolve_uncovered_request(isolated_voice_settings(), query, [], schemas, prior_results=prior)
    assert proposal is not None, 'An unresolved available action needs a concrete recovery proposal'
    assert proposal['name'] == tool.name
    assert proposal['arguments']['operation'] == operation
    assert proposal['arguments']['scope'] == scope
    result = tool.run(proposal['arguments'], None)
    assert result.success
    assert len(tool.calls) == 1
    assert tool.entries == remaining


@pytest.mark.parametrize('first_query', [
    'Who directed Possessor?',
    'director of Possessor and their filmography',
])
def test_recovery_query_uses_the_discovered_entity(monkeypatch, first_query):
    tool = OwnedFilmLookup()
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    first_args = {'query': first_query}
    first = tool.run(first_args, None)
    schemas = registry.generate_tools_json_schema([tool.name, 'stop', 'toolSearchTool'])
    proposal = resolve_uncovered_request(
        isolated_voice_settings(),
        'Who directed Possessor and what other films has that director made?',
        [{'name': tool.name, 'arguments': first_args, 'result': first.reply_text, 'success': first.success}],
        schemas, prior_results=[],
    )
    assert proposal is not None
    assert proposal['name'] == tool.name
    query = proposal['arguments']['query'].casefold()
    assert 'brandon' in query and 'cronenberg' in query
    result = tool.run(proposal['arguments'], None)
    assert result.success
    assert len(tool.calls) == 2
    assert 'Antiviral' in result.reply_text and 'Infinity Pool' in result.reply_text


@pytest.mark.parametrize('entries,target,remaining', [
    ({'tea', 'soup'}, 'tea', {'soup'}),
    ({'tea reminder', 'soup reminder'}, 'tea reminder', {'soup reminder'}),
])
def test_recovery_discovers_unknown_names_before_mutating(monkeypatch, entries, target, remaining):
    tool = OwnedEntryManager()
    tool.entries = set(entries)
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    schemas = registry.generate_tools_json_schema([tool.name, 'stop', 'toolSearchTool'])
    cfg = isolated_voice_settings()
    query = 'Delete only the tea reminder and keep the soup reminder.'
    first = resolve_uncovered_request(cfg, query, [], schemas, prior_results=[])
    assert first is not None and first['name'] == tool.name
    assert first['arguments']['operation'] == 'enumerate_entries'
    assert first['arguments']['scope'] == 'whole_collection'
    observed = tool.run(first['arguments'], None)
    assert observed.success
    second = resolve_uncovered_request(cfg, query, [{
        'name': tool.name, 'arguments': first['arguments'], 'success': True, 'result': observed.reply_text,
    }], schemas, prior_results=[])
    assert second is not None and second['name'] == tool.name
    assert second['arguments']['operation'] == 'discard_entries'
    assert second['arguments']['scope'] == 'matching_labels'
    assert second['arguments']['labels'] == [target]
    result = tool.run(second['arguments'], None)
    assert result.success
    assert len(tool.calls) == 2
    assert tool.entries == remaining


def test_previous_mutation_does_not_complete_a_new_request(monkeypatch):
    tool = OwnedEntryManager()
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    prior = [observed_entry_names(tool)]
    previous = tool.run({'operation': 'discard_entries', 'scope': 'matching_labels', 'labels': ['tea']}, None)
    assert previous.success
    prior.append({'name': tool.name, 'tool_call_id': 'previous-mutation',
                  'result': previous.reply_text, 'resource_references': []})
    tool.entries.add('tea')
    tool.calls.clear()
    schemas = registry.generate_tools_json_schema([tool.name, 'stop', 'toolSearchTool'])
    proposal = resolve_uncovered_request(
        isolated_voice_settings(), 'Delete only the entry named tea and keep the entry named soup.',
        [], schemas, prior_results=prior,
    )
    assert proposal is not None and proposal['name'] == tool.name
    assert proposal['arguments']['operation'] == 'discard_entries'
    assert proposal['arguments']['scope'] == 'matching_labels'
    assert proposal['arguments']['labels'] == ['tea']
    assert tool.run(proposal['arguments'], None).success
    assert len(tool.calls) == 1
    assert tool.entries == {'soup'}


def test_missing_identity_does_not_produce_a_mutation(monkeypatch):
    tool = OwnedEntryManager()
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    arguments = {'operation': 'enumerate_entries', 'scope': 'whole_collection'}
    observed = tool.run(arguments, None)
    schemas = registry.generate_tools_json_schema([tool.name, 'stop', 'toolSearchTool'])
    proposal = resolve_uncovered_request(
        isolated_voice_settings(), 'Delete only the cocoa reminder in this collection and keep the soup reminder.',
        [{'name': tool.name, 'arguments': arguments, 'success': True, 'result': observed.reply_text}],
        schemas, prior_results=[],
    )
    assert proposal is None, 'An observed namespace without the requested record requires clarification'
    assert len(tool.calls) == 1
    assert tool.entries == {'tea', 'soup'}


@pytest.mark.parametrize('entries,query',[
    ({'reminder','soup'},'Delete only the cocoa reminder in this collection and keep the soup reminder.'),
    ({'note','shopping'},'Remove only my travel note and keep my shopping note.'),
    ({'tea','soup'},'Delete only the cocoa reminder.'),
    ({'tea','tea reminder','soup'},'Delete only the tea reminder and keep the soup reminder.'),
    ({'tea green','tea black','soup'},'Delete only the tea reminder and keep the soup reminder.'),
])
def test_contract_snapshot_never_guesses_an_absent_or_ambiguous_target(monkeypatch,entries,query):
    tool=OwnedEntryManager();tool.entries=set(entries)
    monkeypatch.setitem(registry.BUILTIN_TOOLS,tool.name,tool)
    args={'operation':'enumerate_entries','scope':'whole_collection'}
    observed=tool.run(args,None)
    schemas=registry.generate_tools_json_schema([tool.name,'stop','toolSearchTool'])
    proposal=resolve_uncovered_request(isolated_voice_settings(),query,[{'name':tool.name,'arguments':args,'success':True,'result':observed.reply_text}],schemas,prior_results=[])
    assert proposal is None
    assert len(tool.calls)==1
    assert tool.entries==entries

def test_contract_fresh_complete_snapshot_replaces_stale_identity(monkeypatch):
    tool=OwnedEntryManager()
    monkeypatch.setitem(registry.BUILTIN_TOOLS,tool.name,tool)
    prior=[observed_entry_names(tool)]
    removed=tool.run({'operation':'discard_entries','scope':'matching_labels','labels':['tea']},None)
    assert removed.success
    tool.calls.clear()
    args={'operation':'enumerate_entries','scope':'whole_collection'};observed=tool.run(args,None)
    schemas=registry.generate_tools_json_schema([tool.name,'stop','toolSearchTool'])
    proposal=resolve_uncovered_request(isolated_voice_settings(),'Delete only the tea reminder and keep the soup reminder.',[{'name':tool.name,'arguments':args,'success':True,'result':observed.reply_text}],schemas,prior_results=prior)
    assert proposal is None
    assert len(tool.calls)==1
    assert tool.entries=={'soup'}

@pytest.mark.parametrize('query,target,keeper',[
    ('Entferne nur die Tee-Erinnerung und behalte Suppe.','Tee','Suppe'),
    ('Sadece çay hatırlatıcısını sil, çorba hatırlatıcısı kalsın.','çay','çorba'),
    ('Delete only the tea reminder and keep the soup reminder.','tea '+('kettle '*50),'soup'),
])
def test_contract_discovers_then_copies_the_actual_identifier(monkeypatch,query,target,keeper):
    tool=OwnedEntryManager();tool.entries={target,keeper}
    monkeypatch.setitem(registry.BUILTIN_TOOLS,tool.name,tool)
    cfg=isolated_voice_settings();schemas=registry.generate_tools_json_schema([tool.name,'stop','toolSearchTool'])
    first=resolve_uncovered_request(cfg,query,[],schemas,prior_results=[])
    assert first=={'name':tool.name,'arguments':{'operation':'enumerate_entries','scope':'whole_collection'}}
    observed=tool.run(first['arguments'],None);assert observed.success
    second=resolve_uncovered_request(cfg,query,[{'name':tool.name,'arguments':first['arguments'],'success':True,'result':observed.reply_text}],schemas,prior_results=[])
    assert second=={'name':tool.name,'arguments':{'operation':'discard_entries','scope':'matching_labels','labels':[target]}}
    assert tool.run(second['arguments'],None).success
    assert len(tool.calls)==2
    assert tool.entries=={keeper}
