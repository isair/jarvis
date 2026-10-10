"""Step resolution uses declared nested types, enums and required fields."""
import pytest

from conftest import requires_judge_llm
from helpers import voice_config
from jarvis.reply.planner import resolve_next_tool_call

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize('step', [
    'fetchRecords retrieve five unfinished records, excluding archived records',
    'fetchRecords arşivlenmiş kayıtlar hariç, tamamlanmamış beş kaydı getir',
])
def test_resolved_arguments_follow_nested_schema(step):
    parameters = {
        'type': 'object',
        'properties': {
            'filter': {'type': 'object', 'properties': {
                'state': {'type': 'string', 'enum': ['pending', 'complete'],
                          'description': 'pending: unfinished; complete: finished'},
            }, 'required': ['state']},
            'limit': {'type': 'integer', 'minimum': 1, 'maximum': 50},
            'include_archived': {'type': 'boolean'},
        },
        'required': ['filter', 'limit', 'include_archived'],
    }
    schema = [{'type': 'function', 'function': {
        'name': 'fetchRecords', 'description': 'Retrieve matching records.',
        'parameters': parameters,
    }}]
    resolved = resolve_next_tool_call(voice_config(), step, [], schema, timeout_sec=60)
    assert resolved == ('fetchRecords', {
        'filter': {'state': 'pending'}, 'limit': 5, 'include_archived': False,
    }), resolved
