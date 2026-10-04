"""Planner dialogue retains spoken entities across native tool traffic."""
from unittest.mock import patch

import pytest
from jarvis.reply import engine
from jarvis.tools.types import ToolExecutionResult

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('tool_count', [0, 3, 7])
def test_followup_entity_survives_native_tool_messages(monkeypatch, mock_config, db, dialogue_memory, tool_count):
    monkeypatch.setattr('jarvis.reply.planner.call_llm_direct', lambda **kwargs: pytest.fail('🧠 Concrete fixture must not infer'))
    mock_config.location_enabled = False
    entity = 'Natsume Sōseki'
    dialogue_memory.add_message('user', f'Tell me about {entity}.')
    native_messages = []
    for index in range(tool_count):
        call_id = f'fixture-{index}'
        native_messages.extend([
            {'role': 'assistant', 'content': '', 'tool_calls': [{
                'id': call_id, 'type': 'function', 'function': {'name': 'webSearch', 'arguments': {'search_query': entity}},
            }]},
            {'role': 'tool', 'tool_name': 'webSearch', 'tool_call_id': call_id, 'content': 'Reference material.', 'tool_failed': False},
        ])
    dialogue_memory.record_tool_turn(native_messages)
    dialogue_memory.add_message('assistant', 'Which of his books would you like to discuss?')
    mock_config.llm_chat_model = 'gemma4:e2b'
    invoked = []

    def plan(**kwargs):
        topic = entity if entity in kwargs['dialogue_context'] else 'unresolved author'
        return [f"webSearch search_query='{topic} books'", 'Reply to the user.']

    def run_tool(db, cfg, tool_name, tool_args, **kwargs):
        invoked.append((tool_name, tool_args))
        return ToolExecutionResult(success=True, reply_text='Kokoro and I Am a Cat.')

    with patch.object(engine, 'select_tools', return_value=['webSearch', 'stop']), \
         patch.object(engine, 'plan_query', side_effect=plan), \
         patch.object(engine, 'run_tool_with_retries', side_effect=run_tool), \
         patch.object(engine, 'chat_with_messages', return_value={'message': {'content': 'Kokoro and I Am a Cat.'}}):
        reply = engine.run_reply_engine(db, mock_config, None, 'What books did he write?', dialogue_memory)

    assert reply == 'Kokoro and I Am a Cat.'
    assert ('webSearch', {'search_query': f'{entity} books'}) in invoked


@pytest.mark.parametrize('traffic', [
    [{'role': 'system', 'content': 'Internal fixture.'}] * 8,
    [{'role': 'user', 'content': '  '}, {'role': 'assistant', 'content': ''}] * 4,
])
def test_ineligible_messages_cannot_displace_dialogue(mock_config, db, dialogue_memory, traffic):
    mock_config.location_enabled = False
    dialogue_memory.add_message('user', 'The chosen author is Ursula Le Guin.')
    dialogue_memory.record_tool_turn(traffic)
    contexts = []

    def plan(**kwargs):
        contexts.append(kwargs['dialogue_context'])
        return ['Reply to the user.']

    with patch.object(engine, 'select_tools', return_value=['webSearch', 'stop']), \
         patch.object(engine, 'plan_query', side_effect=plan), \
         patch.object(engine, 'chat_with_messages', return_value={'message': {'content': 'Ursula Le Guin.'}}):
        assert engine.run_reply_engine(db, mock_config, None, 'Who is the chosen author?', dialogue_memory) == 'Ursula Le Guin.'
    assert contexts == ['user: The chosen author is Ursula Le Guin.']


def test_dialogue_selection_retains_chronology_and_existing_size_bounds(mock_config, db, dialogue_memory):
    mock_config.location_enabled = False
    expected = []
    for index in range(engine._HINT_RECENT_MESSAGES + 3):
        role = 'user' if index % 2 == 0 else 'assistant'
        content = f'entity-{index} ' + 'あ' * (engine._HINT_MESSAGE_CHAR_LIMIT + 50)
        dialogue_memory.add_message(role, content)
        expected.append(f'{role}: {content[:engine._HINT_MESSAGE_CHAR_LIMIT]}')
    contexts = []

    def plan(**kwargs):
        contexts.append(kwargs['dialogue_context'])
        return ['Reply to the user.']

    with patch.object(engine, 'select_tools', return_value=['webSearch', 'stop']), \
         patch.object(engine, 'plan_query', side_effect=plan), \
         patch.object(engine, 'chat_with_messages', return_value={'message': {'content': 'Selected.'}}):
        assert engine.run_reply_engine(db, mock_config, None, 'Which entity were we discussing?', dialogue_memory) == 'Selected.'
    assert contexts == ['\n'.join(expected[-engine._HINT_RECENT_MESSAGES:])]
