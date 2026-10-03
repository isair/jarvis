"""Required arguments remain grounded when resolving dependent tool steps."""
import pytest

from evals.helpers import voice_config
from evals.tool_routing import requires_judge_llm
from jarvis.reply.planner import resolve_next_tool_call

pytestmark = [pytest.mark.eval, requires_judge_llm]
CASES = [
    ('library__lookup', 'Amber Atlas', 'Selected document: Amber Atlas.', ['item']),
    ('library__lookup', 'Kehribar Atlası', 'Seçilen belge: Kehribar Atlası.', ['item']),
    ('local-catalogue__lookup', 'Glass Harbour', 'Selected document: Glass Harbour. Scope: local.', ['item', 'scope']),
]


@pytest.mark.parametrize('tool_name, item, result, required', CASES)
def test_required_resolver_arguments_preserve_discovered_values(tool_name, item, result, required):
    cfg = voice_config()
    schema = [{'type': 'function', 'function': {
        'name': tool_name,
        'description': 'Look up the selected document in the local catalogue.',
        'parameters': {'type': 'object', 'properties': {
            'item': {'type': 'string'}, 'scope': {'type': 'string'},
        }, 'required': required},
    }}]
    step = f"{tool_name} item='<selected document from prior result>'"
    if 'scope' in required:
        step += " scope='local'"
    resolved = resolve_next_tool_call(
        cfg, step, [('selectDocument', '{}', result)], schema, timeout_sec=60.0,
    )
    assert resolved is not None, '🗺️ A complete grounded call must remain resolvable'
    name, arguments = resolved
    assert name == tool_name, f'🛠️ Resolver changed the selected tool: {resolved}'
    assert arguments.get('item', '').casefold() == item.casefold(), f'📚 Discovered document was lost: {resolved}'
    assert set(required) <= arguments.keys(), f'🗺️ Required fields are missing: {resolved}'
    if 'scope' in required:
        assert arguments['scope'] == 'local', f'📚 Explicit scope was changed: {resolved}'
