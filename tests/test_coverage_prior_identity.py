"""Prior tool identities stay distinct from current action evidence."""
import json
from unittest.mock import Mock, patch

import pytest

from jarvis.reply import compound_query
from jarvis.reply.tool_identity import observed_tool_identities

pytestmark = pytest.mark.unit


def _request(name='ownedLookup', identity='observed'):
    return {'role': 'assistant', 'content': '', 'tool_calls': [{
        'id': identity, 'type': 'function', 'function': {
            'name': name, 'arguments': {'expiry_announcement': 'private future speech'},
        },
    }]}


def _result(**changes):
    result = {'role': 'tool', 'tool_name': 'ownedLookup', 'tool_call_id': 'observed',
              'tool_failed': False, 'content': 'Observed names: tea, soup.',
              'resource_references': [{'kind': 'entry', 'id': 'owned-id', 'label': 'tea'}]}
    result.update(changes)
    return result


@pytest.mark.parametrize('role', ['tool', 'user'])
def test_only_matched_successful_results_supply_prior_identity(role):
    evidence = observed_tool_identities([_request(), _result(role=role)])
    assert evidence == [{'name': 'ownedLookup', 'tool_call_id': 'observed',
                         'result': 'Observed names: tea, soup.',
                         'resource_references': [{'kind': 'entry', 'id': 'owned-id', 'label': 'tea'}]}]
    assert 'private future speech' not in json.dumps(evidence)


@pytest.mark.parametrize('changes', [
    {'tool_failed': True}, {'tool_failed': None}, {'tool_feedback': True},
    {'role': 'assistant'}, {'tool_call_id': 'orphan'}, {'tool_name': 'otherNamespace'},
    {'tool_name': None}, {'tool_failed': 0},
])
def test_claims_failures_feedback_and_orphans_cannot_supply_identity(changes):
    assert observed_tool_identities([_request(), _result(**changes)]) == []


def test_duplicate_request_identifiers_cannot_establish_a_namespace():
    assert observed_tool_identities([_request(), _request('otherNamespace'), _result()]) == []


def test_prior_observations_do_not_become_current_execution_results(mock_config):
    backend = Mock()
    backend.direct.return_value = '{"complete":true}'
    prior = [{'name': 'ownedLookup', 'tool_call_id': 'old', 'result': 'A previous action succeeded.'}]
    with patch.object(compound_query, 'get_llm_backend', return_value=backend):
        compound_query.resolve_uncovered_request(mock_config, 'Perform the action again.', [], [], prior_results=prior)
    packet = json.loads(backend.direct.call_args.args[2])
    assert packet['results'] == []
    assert packet['prior_identity_evidence'] == prior


def test_prior_budget_keeps_whole_values_and_reports_omissions(mock_config):
    backend = Mock()
    backend.direct.return_value = '{"complete":true}'
    literal = 'quoted "name" with \\ and 日本語'
    small = {'name': 'ownedLookup', 'tool_call_id': 'small', 'result': literal}
    large = {'name': 'ownedLookup', 'tool_call_id': 'large',
             'result': '\\' * compound_query.COVERAGE_INPUT_BYTE_LIMIT}
    with patch.object(compound_query, 'get_llm_backend', return_value=backend):
        compound_query.resolve_uncovered_request(mock_config, 'Find the record.', [], [], prior_results=[small, large])
    encoded = backend.direct.call_args.args[2]
    packet = json.loads(encoded)
    assert len(encoded.encode('utf-8')) <= compound_query.COVERAGE_INPUT_BYTE_LIMIT
    assert packet['prior_identity_evidence'] == [small]
    assert packet['prior_identity_evidence_omitted'] is True


@pytest.mark.parametrize('role', ['tool', 'user'])
def test_truncated_carryover_cannot_establish_a_partial_record_name(role):
    from jarvis.memory.conversation import DialogueMemory

    memory = DialogueMemory()
    complete_name = 'a complete multiword record label'
    content = 'Observed names: ' + complete_name
    memory.record_tool_turn([_request(), _result(role=role, content=content)])
    clipped = memory.get_recent_turns_with_tools(per_entry_chars=len(content) - 6)
    assert observed_tool_identities(clipped) == []
    # Reading a shorter context must not mutate the complete stored observation.
    complete = memory.get_recent_turns_with_tools(per_entry_chars=len(content))
    evidence = observed_tool_identities(complete)
    assert len(evidence) == 1 and evidence[0]['result'] == content
