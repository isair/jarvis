"""Failed tool outcomes widen the next allow-list without changing the router cache."""

from unittest.mock import Mock, patch

import pytest

from jarvis.reply import engine
from jarvis.tools.types import ToolExecutionResult

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.fixture
def reply_harness(monkeypatch, mock_config, db, dialogue_memory):
    mock_config.llm_chat_model = 'gpt-oss:20b'
    mock_config.planner_enabled = False
    mock_config.voice_debug = False
    mock_config.memory_digest_enabled = False
    mock_config.tool_result_digest_enabled = False
    mock_config.tts_engine = 'none'
    chat = Mock(return_value={'message': {'content': 'Ready.'}})
    monkeypatch.setattr(engine, 'chat_with_messages', chat)
    monkeypatch.setattr(engine, 'plan_query', lambda *args, **kwargs: [])
    monkeypatch.setattr(engine, 'extract_search_params_for_memory', lambda *args, **kwargs: {})
    return mock_config, db, dialogue_memory, chat


def tool_turn(tool, *, failed, native, call_id='lookup'):
    content = 'Lookup unavailable.' if failed else 'Confirmed lookup result.'
    if not native:
        return [{'role': 'user', 'content': content, 'tool_name': tool, 'tool_failed': failed}]
    return [
        {'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': call_id, 'function': {'name': tool, 'arguments': {}},
        }]},
        {'role': 'tool', 'tool_call_id': call_id, 'tool_name': tool,
         'content': content, 'tool_failed': failed},
    ]


def previous_turn(memory, results):
    memory.add_message('user', 'Look up the requested information.')
    memory.record_tool_turn(results)
    memory.add_message('assistant', 'Here is the lookup status.')


def reply(harness, query, route):
    cfg, db, memory, chat = harness
    with patch.object(engine, 'select_tools', return_value=route):
        result = engine.run_reply_engine(db, cfg, None, query, memory)
    assert result == 'Ready.'
    return {tool['function']['name'] for tool in chat.call_args.kwargs['tools']}


@pytest.mark.parametrize('native', [True, False], ids=['native-results', 'text-results'])
@pytest.mark.parametrize('failed', [True, False], ids=['failed', 'successful'])
@pytest.mark.parametrize('query', [
    "I'm in London",
    "Right, sorry, I'm in Edinburgh. Please try the lookup again for tomorrow morning if you would.",
], ids=['short-follow-up', 'long-follow-up'])
def test_only_failed_previous_tools_widen_the_allowlist(reply_harness, native, failed, query):
    _, _, memory, _ = reply_harness
    previous_turn(memory, tool_turn('getWeather', failed=failed, native=native))
    expected = {'webSearch', 'stop', 'toolSearchTool'}
    if failed:
        expected.add('getWeather')
    assert reply(reply_harness, query, ['webSearch']) == expected


def test_cold_start_keeps_the_router_allowlist(reply_harness):
    assert reply(reply_harness, 'Hello.', ['webSearch']) == {'webSearch', 'stop', 'toolSearchTool'}


@pytest.mark.parametrize('native', [True, False])
def test_only_failed_siblings_carry_over(reply_harness, native):
    _, _, memory, _ = reply_harness
    weather = tool_turn('getWeather', failed=True, native=native, call_id='weather')
    search = tool_turn('webSearch', failed=False, native=native, call_id='search')
    if native:
        weather[0]['tool_calls'].extend(search[0]['tool_calls'])
        results = [weather[0], weather[1], search[1]]
    else:
        results = weather + search
    previous_turn(memory, results)
    assert reply(reply_harness, "I'm in Paris", ['fetchWebPage']) == {
        'fetchWebPage', 'getWeather', 'stop', 'toolSearchTool',
    }


@pytest.mark.parametrize('native', [True, False])
def test_corrected_success_does_not_widen_the_allowlist(reply_harness, native):
    _, _, memory, _ = reply_harness
    previous_turn(memory,
                  tool_turn('getWeather', failed=True, native=native, call_id='missing')
                  + tool_turn('getWeather', failed=False, native=native, call_id='corrected'))
    assert reply(reply_harness, 'Search for something else.', ['webSearch']) == {
        'webSearch', 'stop', 'toolSearchTool',
    }


def test_carryover_does_not_pollute_router_cache(reply_harness):
    cfg, db, memory, chat = reply_harness
    previous_turn(memory, tool_turn('getWeather', failed=True, native=True))
    with patch.object(engine, '_build_enrichment_context_hint', return_value='Stable context'), \
         patch.object(engine, 'select_tools', side_effect=[
             ['webSearch'], AssertionError('The cached routing decision should be reused.'),
         ]):
        engine.run_reply_engine(db, cfg, None, "I'm in London", memory)
        assert {tool['function']['name'] for tool in chat.call_args.kwargs['tools']} == {
            'webSearch', 'getWeather', 'stop', 'toolSearchTool',
        }
        memory.clear_tool_carryover()
        engine.run_reply_engine(db, cfg, None, "I'm in London", memory)
    assert {tool['function']['name'] for tool in chat.call_args.kwargs['tools']} == {
        'webSearch', 'stop', 'toolSearchTool',
    }


def test_planner_failure_widens_the_followup_allowlist(reply_harness):
    cfg, db, memory, _ = reply_harness
    cfg.llm_chat_model = 'gemma4:e2b'
    cfg.planner_enabled = True
    with patch.object(engine, 'select_tools', return_value=['getWeather']), \
         patch.object(engine, 'plan_query', return_value=['getWeather', 'Reply to the user.']), \
         patch.object(engine, 'run_tool_with_retries', return_value=ToolExecutionResult(
             success=False, reply_text='Please supply a location.',
         )):
        assert engine.run_reply_engine(db, cfg, None, "How's the weather?", memory) == 'Ready.'
    cfg.llm_chat_model = 'gpt-oss:20b'
    cfg.planner_enabled = False
    assert reply(reply_harness, "I'm in Berlin", ['webSearch']) == {
        'webSearch', 'getWeather', 'stop', 'toolSearchTool',
    }


def test_orphan_call_does_not_widen_the_allowlist_and_logs_the_gap(reply_harness, monkeypatch, capsys):
    monkeypatch.setenv('JARVIS_VOICE_DEBUG', '1')
    _, _, memory, _ = reply_harness
    previous_turn(memory, tool_turn('getWeather', failed=True, native=True)[:1])
    assert reply(reply_harness, "I'm in London", ['webSearch']) == {
        'webSearch', 'stop', 'toolSearchTool',
    }
    assert 'no matching role=tool result' in capsys.readouterr().err
