"""Conversation caches reuse identical inputs and respect changed evidence."""
import pytest

from jarvis.reply import engine
from jarvis.tools.registry import ToolSpec

pytestmark = pytest.mark.unit


@pytest.fixture
def reply_context(monkeypatch, mock_config, db, dialogue_memory):
    mock_config.llm_chat_model = 'gpt-oss:20b'
    mock_config.ollama_chat_model = mock_config.llm_chat_model
    mock_config.tool_selection_strategy = 'llm'
    mock_config.memory_digest_enabled = False
    mock_config.tool_result_digest_enabled = False
    mock_config.memory_enrichment_source = 'diary'
    monkeypatch.setattr(engine, 'plan_query', lambda **kwargs: ['Reply to the user.'])
    monkeypatch.setattr(engine, '_live_time_location_string', lambda cfg: '')
    monkeypatch.setattr('jarvis.memory.graph_ops.build_warm_profile', lambda store: {})
    monkeypatch.setattr('jarvis.memory.graph_ops.format_warm_profile_block', lambda profile: '')

    def chat(cfg, messages, *, tools=None, **kwargs):
        names = [tool['function']['name'] for tool in tools or []]
        names = [name for name in names if name not in {'stop', 'toolSearchTool'}]
        return {'message': {'content': 'Available: ' + ', '.join(names) if names else 'Reply only'}}

    monkeypatch.setattr(engine, 'chat_with_messages', chat)
    return lambda text: engine.run_reply_engine(db, mock_config, None, text, dialogue_memory)


def test_repeated_followup_routes_for_the_current_dialogue(monkeypatch, reply_context, dialogue_memory):
    def route(*, context_hint, **kwargs):
        return ['fetchMeals', 'stop'] if 'meal log' in (context_hint or '') else ['getWeather', 'stop']

    monkeypatch.setattr(engine, 'select_tools', route)
    dialogue_memory.add_message('user', 'Check the weather in London.')
    dialogue_memory.add_message('assistant', 'We can review the forecast.')
    assert 'getWeather' in reply_context('What about tomorrow?')
    dialogue_memory.add_message('user', 'I am tracking my food intake.')
    dialogue_memory.add_message('assistant', 'We can review your meal log.')
    reply = reply_context('What about tomorrow?')
    assert 'fetchMeals' in reply
    assert 'getWeather' not in reply


def test_repeated_query_respects_changed_live_facts(monkeypatch, reply_context):
    live = {'hint': None}
    monkeypatch.setattr(engine, '_build_enrichment_context_hint', lambda cfg, messages: live['hint'])
    monkeypatch.setattr(engine, 'select_tools', lambda **kwargs: ['stop'] if kwargs['context_hint'] else ['getTime', 'stop'])
    assert 'getTime' in reply_context('What time is it?')
    live['hint'] = 'Current time is available in the live context.'
    assert reply_context('What time is it?') == 'Reply only'


@pytest.mark.parametrize('changed_field', ['description', 'inputSchema'])
def test_repeated_query_respects_changed_catalogue(monkeypatch, reply_context, mock_config, changed_field):
    mock_config.mcps = {'fixture': {}}
    specs = {'topic': ToolSpec('topic', 'weather', {'type': 'object', 'required': ['weather']})}
    monkeypatch.setattr('jarvis.tools.registry.get_cached_mcp_tools', lambda: specs.copy())
    monkeypatch.setattr(engine, '_build_enrichment_context_hint', lambda cfg, messages: 'Stable context')

    def route(*, mcp_tools, **kwargs):
        spec = mcp_tools['topic']
        changed = spec.description == 'food' if changed_field == 'description' else 'food' in spec.inputSchema['required']
        return ['fetchMeals', 'stop'] if changed else ['getWeather', 'stop']

    monkeypatch.setattr(engine, 'select_tools', route)
    assert 'getWeather' in reply_context('Check my topic.')
    old = specs['topic']
    specs['topic'] = ToolSpec(old.name, 'food' if changed_field == 'description' else old.description,
                             {'type': 'object', 'required': ['food']} if changed_field == 'inputSchema' else old.inputSchema)
    reply = reply_context('Check my topic.')
    assert 'fetchMeals' in reply
    assert 'getWeather' not in reply


@pytest.mark.parametrize('description', ['weather', '天気 ☀️', '\ud800weather'],
                         ids=['ascii', 'unicode', 'decoded-surrogate'])
def test_identical_router_inputs_reuse_result_if_router_is_unavailable(monkeypatch, reply_context, mock_config, description):
    mock_config.mcps = {'fixture': {}}
    specs = {'topic': ToolSpec('topic', description, {'type': 'object', 'properties': {'a': {}, 'b': {}}})}
    monkeypatch.setattr('jarvis.tools.registry.get_cached_mcp_tools', lambda: specs.copy())
    monkeypatch.setattr(engine, '_build_enrichment_context_hint', lambda cfg, messages: 'Stable context')
    monkeypatch.setattr(engine, 'select_tools', lambda **kwargs: ['getWeather', 'stop'])
    first = reply_context('Check my topic.')
    specs['topic'] = ToolSpec('topic', description, {'properties': {'b': {}, 'a': {}}, 'type': 'object'})

    def unavailable(**kwargs):
        raise AssertionError('Router unavailable for identical inputs')

    monkeypatch.setattr(engine, 'select_tools', unavailable)
    assert reply_context('Check my topic.') == first


@pytest.fixture
def memory_reply_context(monkeypatch, reply_context):
    monkeypatch.setattr(engine, 'select_tools', lambda **kwargs: ['webSearch', 'stop'])
    monkeypatch.setattr(engine, 'plan_query', lambda **kwargs: [])
    monkeypatch.setattr('jarvis.memory.conversation.search_conversation_memory_by_keywords',
                        lambda **kwargs: ['MEAL_MEMORY_FIXTURE' if 'food' in kwargs['keywords'] else 'WEATHER_MEMORY_FIXTURE'])

    def chat(cfg, messages, **kwargs):
        context = '\n'.join(message.get('content', '') for message in messages if message['role'] == 'system')
        reply = ('Meal memory' if 'MEAL_MEMORY_FIXTURE' in context else
                 'Weather memory' if 'WEATHER_MEMORY_FIXTURE' in context else 'No memory retrieved')
        return {'message': {'content': reply}}

    monkeypatch.setattr(engine, 'chat_with_messages', chat)
    return reply_context


def test_repeated_memory_query_recalls_the_current_topic(monkeypatch, memory_reply_context, dialogue_memory):
    monkeypatch.setattr(engine, 'extract_search_params_for_memory',
                        lambda *args, **kwargs: {'keywords': ['food' if 'meal log' in (kwargs['context_hint'] or '') else 'weather']})
    dialogue_memory.add_message('user', 'We were discussing the weather.')
    dialogue_memory.add_message('assistant', 'Tell me what you want to recall.')
    assert memory_reply_context('What did I say about it?') == 'Weather memory'
    dialogue_memory.add_message('user', 'I am discussing my food intake.')
    dialogue_memory.add_message('assistant', 'We are talking about your meal log.')
    assert memory_reply_context('What did I say about it?') == 'Meal memory'


def test_identical_memory_inputs_reuse_params_if_extractor_is_unavailable(monkeypatch, memory_reply_context):
    monkeypatch.setattr(engine, '_build_enrichment_context_hint', lambda cfg, messages: 'Stable context')
    monkeypatch.setattr(engine, 'extract_search_params_for_memory', lambda *args, **kwargs: {'keywords': ['weather']})
    assert memory_reply_context('What did I say about it?') == 'Weather memory'

    def unavailable(*args, **kwargs):
        raise RuntimeError('Extractor unavailable for identical inputs')

    monkeypatch.setattr(engine, 'extract_search_params_for_memory', unavailable)
    assert memory_reply_context('What did I say about it?') == 'Weather memory'


def test_full_catalogue_does_not_pin_tool_exposure(monkeypatch, reply_context):
    monkeypatch.setattr(engine, '_build_enrichment_context_hint', lambda cfg, messages: 'Stable context')
    selections = iter([list(engine.BUILTIN_TOOLS), ['getWeather', 'stop']])
    monkeypatch.setattr(engine, 'select_tools', lambda **kwargs: next(selections))
    assert 'fetchMeals' in reply_context('Check the forecast.')
    reply = reply_context('Check the forecast.')
    assert 'getWeather' in reply
    assert 'fetchMeals' not in reply
