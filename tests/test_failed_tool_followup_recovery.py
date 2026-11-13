"""A selected failed-task continuation needs an actual current result."""
from unittest.mock import patch
import pytest
from jarvis.reply import engine
from jarvis.tools.types import ToolExecutionResult

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


def prior_failure(memory,tool,failed=True):
    memory.add_message('user','Look up the requested information.')
    memory.record_tool_turn([{'role':'user','content':'The lookup needs another input.',
                             'tool_name':tool,'tool_failed':failed}])
    memory.add_message('assistant','Which input should I use?')


def run(mock_config,db,memory,tool,responses,*,route=None,success=True,max_turns=4,seen=None):
    mock_config.planner_enabled=False
    mock_config.agentic_max_turns=max_turns
    outputs=iter(responses)
    def chat(**kwargs):
        if seen is not None:
            seen.append([dict(message) for message in kwargs["messages"]])
        return next(outputs)
    def execute(db,cfg,tool_name,tool_args,**kwargs):
        return ToolExecutionResult(success=success,reply_text='Confirmed result: 14 units.' if success else 'Lookup still unavailable.')
    with patch.object(engine,'select_tools',return_value=route if route is not None else [tool,'stop']), \
         patch.object(engine,'plan_query',return_value=[]), \
         patch.object(engine,'extract_search_params_for_memory',return_value={}), \
         patch.object(engine,'run_tool_with_retries',side_effect=execute), \
         patch.object(engine,'chat_with_messages',side_effect=chat):
        return engine.run_reply_engine(db,mock_config,None,'Use this supplied input.',memory)


def prose(text):
    return {'message':{'role':'assistant','content':text}}


def call(tool):
    return {'message':{'role':'assistant','content':'','tool_calls':[{'id':'followup',
                     'function':{'name':tool,'arguments':{'query':'supplied input'}}}]}}


@pytest.mark.parametrize('model',['gemma4:e2b','gpt-oss:20b'])
@pytest.mark.parametrize('tool',['getWeather','webSearch'])
def test_unsupported_followup_is_corrected_before_delivery(mock_config,db,dialogue_memory,model,tool):
    mock_config.llm_chat_model=model
    prior_failure(dialogue_memory,tool)
    reply=run(mock_config,db,dialogue_memory,tool,
              [prose('Invented result: 18 units.'),call(tool),prose('Confirmed result: 14 units.')])
    assert reply=='Confirmed result: 14 units.'
    assert 'Invented result' not in str(dialogue_memory.get_recent_messages())


@pytest.mark.parametrize('max_turns',[1,2,4])
def test_persistent_unsupported_answer_is_not_delivered(mock_config,db,dialogue_memory,max_turns):
    prior_failure(dialogue_memory,'getWeather')
    reply=run(mock_config,db,dialogue_memory,'getWeather',
              [prose('Invented result: 18 units.')]*4,max_turns=max_turns)
    assert reply==engine._UNEXECUTED_ACTION_REPLY
    assert '18 units' not in str(dialogue_memory.get_recent_messages())


@pytest.mark.parametrize('field',['question','blocker'])
def test_explicit_noncompletion_remains_deliverable(mock_config,db,dialogue_memory,field):
    import json
    prior_failure(dialogue_memory,'getWeather')
    reply=run(mock_config,db,dialogue_memory,'getWeather',[prose(json.dumps({field:'Please supply a more precise input.'}))])
    assert reply=='Please supply a more precise input.'


@pytest.mark.parametrize('route',[['stop'],['webSearch','stop']])
def test_carryover_alone_does_not_force_an_unrelated_reply(mock_config,db,dialogue_memory,route):
    prior_failure(dialogue_memory,'getWeather')
    reply=run(mock_config,db,dialogue_memory,'getWeather',[prose('I will keep replies brief.')],route=route)
    assert reply=='I will keep replies brief.'


def test_prior_success_can_supply_an_answer_without_refetch(mock_config,db,dialogue_memory):
    prior_failure(dialogue_memory,'getWeather',failed=False)
    reply=run(mock_config,db,dialogue_memory,'getWeather',[prose('The earlier result remains available.')])
    assert reply=='The earlier result remains available.'


def test_failed_current_retry_does_not_establish_result(mock_config,db,dialogue_memory):
    prior_failure(dialogue_memory,'getWeather')
    reply=run(mock_config,db,dialogue_memory,'getWeather',
              [call('getWeather'),prose('Invented result: 18 units.'),prose('{"blocker":"The lookup is unavailable."}')],success=False)
    assert reply=='The lookup is unavailable.'


def test_full_catalogue_exposure_does_not_establish_continuation_intent(mock_config,db,dialogue_memory):
    prior_failure(dialogue_memory,'getWeather')
    reply=run(mock_config,db,dialogue_memory,'getWeather',[prose('I will keep replies brief.')],
              route=list(engine.BUILTIN_TOOLS))
    assert reply=='I will keep replies brief.'


def test_tool_discovery_is_not_a_current_lookup_result(mock_config,db,dialogue_memory):
    prior_failure(dialogue_memory,'getWeather')
    reply=run(mock_config,db,dialogue_memory,'getWeather',
              [call('toolSearchTool'),prose('Invented result: 18 units.'),
               prose('{"blocker":"No lookup result is available."}')])
    assert reply=='No lookup result is available.'


def test_stop_dismissal_bypasses_recovery_and_clears_carryover(mock_config,db,dialogue_memory):
    prior_failure(dialogue_memory,'getWeather')
    with patch.object(engine,'select_tools',return_value=['getWeather','stop']), \
         patch.object(engine,'plan_query',return_value=[]), \
         patch.object(engine,'extract_search_params_for_memory',return_value={}), \
         patch.object(engine,'run_tool_with_retries',return_value=ToolExecutionResult(success=True,reply_text=engine.STOP_SIGNAL)), \
         patch.object(engine,'chat_with_messages',return_value=call('stop')):
        reply=engine.run_reply_engine(db,mock_config,None,'Stop.',dialogue_memory)
    assert reply is None
    assert not engine._previous_turn_failed_tool_names(dialogue_memory.get_recent_turns_with_tools())


def test_rejected_draft_is_not_reused_as_context(mock_config,db,dialogue_memory):
    prior_failure(dialogue_memory,'getWeather')
    seen=[]
    reply=run(mock_config,db,dialogue_memory,'getWeather',
              [prose('Invented result: 18 units.'),call('getWeather'),prose('Confirmed result: 14 units.')],seen=seen)
    assert reply=='Confirmed result: 14 units.'
    assert all('Invented result' not in str(messages) for messages in seen[1:])
    assert any('Which input should I use?' in message.get('content','') for message in seen[1])


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('latest_failed', [False, True])
def test_latest_prior_outcome_controls_followup_requirement(mock_config, db, dialogue_memory, native, latest_failed):
    dialogue_memory.add_message('user', 'Look up the requested information.')
    results = []
    for index, failed in enumerate([not latest_failed, latest_failed]):
        content = 'Lookup unavailable.' if failed else 'Confirmed result: 14 units.'
        if native:
            results.extend([
                {'role': 'assistant', 'content': '', 'tool_calls': [{
                    'id': str(index), 'function': {'name': 'getWeather', 'arguments': {}},
                }]},
                {'role': 'tool', 'tool_call_id': str(index), 'tool_name': 'getWeather',
                 'content': content, 'tool_failed': failed},
            ])
        else:
            results.append({'role': 'user', 'tool_name': 'getWeather',
                            'content': content, 'tool_failed': failed})
    dialogue_memory.record_tool_turn(results)
    dialogue_memory.add_message('assistant', 'The lookup is unavailable.' if latest_failed else 'Confirmed result: 14 units.')
    reply = run(mock_config, db, dialogue_memory, 'getWeather',
                [prose('The earlier result was 14 units.')]*4)
    expected = engine._UNEXECUTED_ACTION_REPLY if latest_failed else 'The earlier result was 14 units.'
    assert reply == expected
