"""Native tool batches complete before the next model response."""
from copy import deepcopy
from threading import Event
from unittest.mock import patch
import pytest

import pytest

pytestmark = pytest.mark.unit


def _call(city, call_id=None, name="getWeather", arguments=None):
    request = {"type": "function", "function": {
        "name": name, "arguments": {"location": city} if arguments is None else arguments,
    }}
    if call_id is not None:
        request["id"] = call_id
    return request


def _run(mock_config, db, dialogue_memory, requests, *, outcome=None, budget=8, cancel=None, allowed=None, followup=None, textual=False):
    from jarvis.reply import engine
    from jarvis.tools.types import ToolExecutionResult

    mock_config.llm_chat_model = "gpt-oss:20b"
    mock_config.ollama_chat_model = "gpt-oss:20b"
    mock_config.agentic_max_turns = budget
    calls, histories = [], []

    def chat(*args, **kwargs):
        histories.append(deepcopy(kwargs.get("messages", args[2] if len(args) > 2 else [])))
        if len(histories) == 1:
            if textual:
                import json
                return {'message': {'content': json.dumps({'tool_calls': requests})}}
            return {"message": {"role": "assistant", "content": "", "tool_calls": deepcopy(requests)}}
        if len(histories) == 2 and followup:
            return {"message": {"role": "assistant", "content": "", "tool_calls": deepcopy(followup)}}
        return {"message": {"role": "assistant", "content": "Weather checked."}}

    def execute(db, cfg, tool_name, tool_args, **kwargs):
        calls.append((tool_name, deepcopy(tool_args)))
        if outcome:
            return outcome(len(calls))
        return ToolExecutionResult(success=True, reply_text=f"Weather: {tool_args}", error_message=None)

    with patch.object(engine, "plan_query", return_value=[]), \
         patch.object(engine, "select_tools", return_value=allowed or ["getWeather", "stop"]), \
         patch.object(engine, "extract_search_params_for_memory", return_value={}), \
         patch.object(engine, "chat_with_messages", side_effect=chat), \
         patch.object(engine, "run_tool_with_retries", side_effect=execute):
        reply = engine.run_reply_engine(db=db, cfg=mock_config, tts=None, text="Check London and Paris weather",
                                        dialogue_memory=dialogue_memory, cancel_event=cancel)
    return calls, histories, reply


def _paired_results(history):
    request_message = next(m for m in history if m.get("role") == "assistant" and m.get("tool_calls"))
    requests = request_message["tool_calls"]
    results = [m for m in history if m.get("role") == "tool"]
    assert [r["id"] for r in requests] == [r["tool_call_id"] for r in results]
    assert [r["function"]["name"] for r in requests] == [r["tool_name"] for r in results]
    assert len({r["id"] for r in requests}) == len(requests)
    return results


@pytest.mark.parametrize("ids", [("london", "paris"), (None, None), ("same", "same")])
def test_batch_executes_every_request_with_paired_history(mock_config, db, dialogue_memory, ids):
    calls, histories, reply = _run(mock_config, db, dialogue_memory,
                                   [_call("London", ids[0]), _call("Paris", ids[1])])
    assert calls == [("getWeather", {"location": "London"}), ("getWeather", {"location": "Paris"})]
    assert reply == "Weather checked."
    assert [r["tool_failed"] for r in _paired_results(histories[1])] == [False, False]


@pytest.mark.parametrize("extra", [
    _call("Paris", "bad", arguments="not json"),
    _call("Paris", "bad", name="unknownTool"),
    _call("London", "duplicate"),
])
def test_rejected_request_does_not_hide_valid_sibling(mock_config, db, dialogue_memory, extra):
    calls, histories, _ = _run(mock_config, db, dialogue_memory,
                               [_call("London", "first"), extra, _call("Berlin", "last")])
    assert calls == [("getWeather", {"location": "London"}), ("getWeather", {"location": "Berlin"})]
    results = _paired_results(histories[1])
    assert len(results) == 3
    assert results[1]["tool_feedback"] is True


def test_cancellation_stops_later_requests(mock_config, db, dialogue_memory):
    from jarvis.tools.types import ToolExecutionResult
    cancelled = Event()

    def cancel_after_first(count):
        cancelled.set()
        return ToolExecutionResult(success=True, reply_text="London checked.", error_message=None)

    calls, histories, reply = _run(mock_config, db, dialogue_memory,
        [_call("London", "first"), _call("Paris", "second")], outcome=cancel_after_first, cancel=cancelled)
    assert calls == [("getWeather", {"location": "London"})]
    assert len(histories) == 1
    assert reply is None


def test_oversized_batch_has_no_partial_effects(mock_config, db, dialogue_memory):
    calls, histories, _ = _run(mock_config, db, dialogue_memory,
        [_call("London", "first"), _call("Paris", "second"), _call("Berlin", "third")], budget=2)
    assert calls == []
    results = _paired_results(histories[1])
    assert all(r["tool_failed"] and "No action in this batch was performed" in r["content"] for r in results)


def test_malformed_entries_do_not_hide_recognisable_calls(mock_config, db, dialogue_memory):
    calls, histories, _ = _run(mock_config, db, dialogue_memory,
        [None, {"function": {}}, _call("London", "valid"), {"function": {"name": None}}])
    assert calls == [("getWeather", {"location": "London"})]
    assert len(_paired_results(histories[1])) == 1


def test_stop_prevents_later_actions(mock_config, db, dialogue_memory):
    from jarvis.reply.engine import STOP_SIGNAL
    from jarvis.tools.types import ToolExecutionResult
    calls, histories, reply = _run(mock_config, db, dialogue_memory,
        [_call("", "stop", name="stop", arguments={}), _call("Paris", "later")],
        outcome=lambda count: ToolExecutionResult(success=True, reply_text=STOP_SIGNAL, error_message=None))
    assert calls == [("stop", {})]
    assert len(histories) == 1
    assert reply is None


def test_discovery_only_authorises_subsequent_model_requests(mock_config, db, dialogue_memory):
    from jarvis.tools.types import ToolExecutionResult

    def discovery_result(count):
        return ToolExecutionResult(success=True, reply_text=(
            "getWeather: Report current weather." if count == 1 else "Paris: 18 C."))

    calls, histories, _ = _run(mock_config, db, dialogue_memory,
        [_call("", "search", name="toolSearchTool", arguments={"query": "weather"}),
         _call("Paris", "guessed")],
        allowed=["webSearch", "stop"], followup=[_call("Paris", "selected")], outcome=discovery_result)
    assert calls == [("toolSearchTool", {"query": "weather"}), ("getWeather", {"location": "Paris"})]
    outcomes = _paired_results(histories[1])
    assert [r["tool_failed"] for r in outcomes] == [False, True]
    final_outcomes = [m for m in histories[2] if m.get("role") == "tool"]
    assert final_outcomes[-1]["tool_call_id"] == "selected"
    assert final_outcomes[-1]["tool_failed"] is False


@pytest.mark.parametrize("stop_first", [True, False])
def test_stop_dismisses_even_an_oversized_batch(mock_config, db, dialogue_memory, stop_first):
    from jarvis.reply.engine import STOP_SIGNAL
    from jarvis.tools.types import ToolExecutionResult
    stop = _call("", "dismiss", name="stop", arguments={})
    actions = [_call("London", "first"), _call("Paris", "second")]
    requests = [stop, *actions] if stop_first else [*actions, stop]
    calls, histories, reply = _run(mock_config, db, dialogue_memory, requests, budget=2,
        outcome=lambda count: ToolExecutionResult(success=True, reply_text=STOP_SIGNAL))
    assert calls == [("stop", {})]
    assert len(histories) == 1
    assert reply is None


def test_distinct_matching_timer_actions_execute_in_one_batch_but_not_on_retry(mock_config, db, dialogue_memory):
    args = {'action': 'set', 'minutes': 5, 'label': 'tea', 'expiry_announcement': 'Tea is ready'}
    calls, histories, reply = _run(mock_config, db, dialogue_memory,
        [_call('', 'one', name='timer', arguments=args), _call('', 'two', name='timer', arguments=args)],
        allowed=['timer', 'stop'], followup=[_call('', 'retry', name='timer', arguments=args)])
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert len(_paired_results(histories[1])) == 2
    assert len(histories) == 3


def test_changed_expiry_wording_does_not_restart_a_completed_countdown(mock_config, db, dialogue_memory):
    args = {'action': 'set', 'minutes': 5, 'label': 'tea', 'expiry_announcement': 'Tea is ready'}
    retry = {**args, 'expiry_announcement': 'Your tea is ready'}
    calls, histories, reply = _run(mock_config, db, dialogue_memory,
        [_call('', 'one', name='timer', arguments=args)], allowed=['timer', 'stop'],
        followup=[_call('', 'retry', name='timer', arguments=retry)])
    assert calls == [('timer', args)]
    assert len(histories) == 3


def test_label_or_duration_units_on_a_retry_do_not_start_another_countdown(mock_config, db, dialogue_memory):
    args = {'action': 'set', 'minutes': 5, 'expiry_announcement': 'Tea is ready'}
    retry = {'action': 'set', 'seconds': 300, 'label': 'çay', 'expiry_announcement': 'Your tea is ready'}
    calls, histories, reply = _run(mock_config, db, dialogue_memory,
        [_call('', 'one', name='timer', arguments=args)], allowed=['timer', 'stop'],
        followup=[_call('', 'retry', name='timer', arguments=retry)])
    assert calls == [('timer', args)]
    assert len(histories) == 3


def test_distinct_labels_in_the_same_creation_batch_remain_separate(mock_config, db, dialogue_memory):
    tea = {'action': 'set', 'minutes': 5, 'label': 'tea', 'expiry_announcement': 'Tea is ready'}
    soup = {**tea, 'label': 'soup'}
    calls, histories, reply = _run(mock_config, db, dialogue_memory,
        [_call('', 'tea', name='timer', arguments=tea), _call('', 'soup', name='timer', arguments=soup)],
        allowed=['timer', 'stop'], followup=[_call('', 'retry', name='timer', arguments={**soup, 'minutes': 0, 'seconds': 300})])
    assert calls == [('timer', tea), ('timer', soup)]


def test_distinct_named_countdowns_on_successive_turns_remain_separate(mock_config, db, dialogue_memory):
    tea = {'action': 'set', 'minutes': 5, 'label': 'tea', 'expiry_announcement': 'Tea is ready'}
    soup = {**tea, 'label': 'soup'}
    calls, histories, reply = _run(mock_config, db, dialogue_memory,
        [_call('', 'tea', name='timer', arguments=tea)], allowed=['timer', 'stop'],
        followup=[_call('', 'soup', name='timer', arguments=soup)])
    assert calls == [('timer', tea), ('timer', soup)]


def test_a_complete_json_text_batch_executes_every_distinct_request(mock_config, db, dialogue_memory):
    requests = [_call('London', 'one'), _call('Paris', 'two')]
    calls, histories, reply = _run(mock_config, db, dialogue_memory, requests, textual=True)
    assert calls == [('getWeather', {'location': 'London'}), ('getWeather', {'location': 'Paris'})]
    assert len(_paired_results(histories[1])) == 2


def test_missing_expiry_can_be_supplied_after_a_failed_attempt(mock_config, db, dialogue_memory):
    from jarvis.tools.types import ToolExecutionResult
    incomplete = {'action': 'set', 'minutes': 5}
    complete = {**incomplete, 'expiry_announcement': 'Tea is ready'}
    effects = []
    def outcome(attempt):
        if attempt == 1:
            return ToolExecutionResult(False, None, error_message='Expiry text is missing.')
        effects.append('started')
        return ToolExecutionResult(True, 'Countdown state: running')
    calls, histories, reply = _run(mock_config, db, dialogue_memory,
        [_call('', 'missing', name='timer', arguments=incomplete)], outcome=outcome,
        allowed=['timer', 'stop'], followup=[_call('', 'complete', name='timer', arguments=complete)])
    assert len(calls) == 2
    assert effects == ['started']


@pytest.mark.parametrize('encoded', [False, True])
def test_future_alarm_speech_is_not_repeated_in_confirmation_context(encoded):
    import json
    from jarvis.reply import engine
    args = {'action': 'set', 'minutes': 5, 'expiry_announcement': 'The tea timer has elapsed'}
    call = _call('', 'owned', name='timer', arguments=args)
    if encoded:
        call['function']['arguments'] = json.dumps(args)
    messages = [{'role': 'assistant', 'content': 'tool_calls: '+json.dumps([call]), 'tool_calls': [call]},
                {'role': 'tool', 'tool_call_id': 'owned', 'content': 'Countdown state: running'}]
    before = deepcopy(messages)
    for text_protocol in (False, True):
        visible = engine._model_messages(messages, text_protocol=text_protocol)
        assert 'The tea timer has elapsed' not in json.dumps(visible)
        requests = json.loads(visible[0]['content'])['tool_calls']
        assert requests[0]['id'] == visible[1]['tool_call_id'] == 'owned'
        assert requests[0]['function']['arguments'] == {'action': 'set', 'minutes': 5}
        assert visible[0]['tool_calls'] == requests
    assert messages == before


def test_max_turn_summary_does_not_receive_future_alarm_speech(monkeypatch, mock_config, db, dialogue_memory):
    from jarvis.reply import engine
    from jarvis.tools.types import ToolExecutionResult
    import json
    mock_config.llm_chat_model = mock_config.ollama_chat_model = 'gpt-oss:20b'
    mock_config.agentic_max_turns = 1
    monkeypatch.setattr(engine, 'plan_query', lambda **kw: [])
    monkeypatch.setattr(engine, 'select_tools', lambda **kw: ['timer', 'stop'])
    monkeypatch.setattr(engine, 'extract_search_params_for_memory', lambda **kw: {})
    args = {'action': 'set', 'minutes': 5, 'expiry_announcement': 'Your tea timer has elapsed'}
    monkeypatch.setattr(engine, 'chat_with_messages', lambda **kw: {'message': {'tool_calls': [_call('', 'one', name='timer', arguments=args)]}})
    effects = []
    def execute(**kw):
        effects.append(kw['tool_args'])
        return ToolExecutionResult(True, 'Countdown state: running\nElapsed: false')
    monkeypatch.setattr(engine, 'run_tool_with_retries', execute)
    def digest(**kw):
        assert args['expiry_announcement'] not in json.dumps(kw['loop_messages'])
        assert 'Countdown state: running' in json.dumps(kw['loop_messages'])
        return 'The timer is running for five minutes.'
    monkeypatch.setattr(engine, 'digest_loop_for_max_turns', digest)
    assert engine.run_reply_engine(db, mock_config, None, 'Start a five minute timer.', dialogue_memory) == 'The timer is running for five minutes.'
    assert effects == [args]
