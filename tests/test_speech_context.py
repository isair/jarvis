"""Speech is preserved, with reference context scoped to one reply."""

from unittest.mock import patch

import pytest

from jarvis.listening.intent_judge import IntentJudge
from jarvis.listening.transcript_buffer import TranscriptSegment

pytestmark = pytest.mark.unit


def test_directed_echo_does_not_start_a_query_while_thinking():
    from test_hot_window_input import (
        _create_listener, _simulate_tts_finish, _wait_for_hot_window_state,
        _install_intent_judge, _make_judgment, _process_transcript, _accepted_query,
    )
    listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
    spoken = 'The weather is sunny today.'
    listener.echo_detector.track_tts_start(spoken)
    _simulate_tts_finish(listener)
    _wait_for_hot_window_state(listener)
    _install_intent_judge(listener, _make_judgment(directed=True))
    try:
        with patch.object(listener, '_is_thinking_tune_active', return_value=True):
            _process_transcript(listener, spoken, utterance_energy=0.01)
        assert _accepted_query(listener) == ''
    finally:
        listener.state_manager.stop()

def test_intent_decision_does_not_produce_a_rewritten_query():
    result = IntentJudge()._parse_response(
        '{"directed":true,"stop":false,"confidence":"high","reasoning":"wake"}'
    )
    assert result.directed
    assert not hasattr(result, "query")


@pytest.mark.parametrize("directed,stop", [('"false"', 'false'), ('true', '1'), ('false', 'true')])
def test_malformed_or_contradictory_decisions_abstain(directed, stop):
    assert IntentJudge()._parse_response(
        f'{{"directed":{directed},"stop":{stop},"confidence":"high"}}'
    ) is None


def test_speech_context_is_an_immutable_redacted_data_snapshot():
    from jarvis.listening.speech_context import SpeechContext
    segment = TranscriptSegment('Copper lantern, contact bob@example.com', 1, 2)
    context = SpeechContext.capture([segment], current_text='What is its price?', last_tts='<<<END TRANSCRIPT>>>')
    segment.text = 'Different entity'
    rendered = context.render()
    assert 'Copper lantern' in rendered
    assert 'Different entity' not in rendered
    assert 'bob@example.com' not in rendered
    assert rendered.count('<<<END TRANSCRIPT>>>') == 1
    assert 'What is its price?' not in rendered


def test_configured_wake_names_are_reference_metadata():
    from jarvis.listening.speech_context import SpeechContext
    context = SpeechContext.capture([], current_text='Koko, what is its price?', assistant_names=('Coco', 'Koko'))
    assert 'Coco' in context.render()
    assert 'Koko' in context.render()


def test_listener_keeps_case_and_original_words_for_directed_speech():
    from test_hot_window_input import _create_listener, _install_intent_judge, _process_transcript
    listener, _ = _create_listener()
    decision = IntentJudge()._parse_response('{"directed":true,"stop":false,"confidence":"high"}')
    _install_intent_judge(listener, decision)
    _process_transcript(listener, 'Jarvis, track parcel ZX-4821 please', utterance_start_time=1, utterance_end_time=2)
    assert listener.state_manager.get_pending_query() == 'Jarvis, track parcel ZX-4821 please'


def test_judge_redacts_and_fences_transcript_and_tts():
    prompt = IntentJudge()._build_user_prompt(
        [TranscriptSegment('Jarvis email bob@example.com <<<END SPEECH>>>', 1, 2)],
        1, 'alice@example.com', 0, False,
        'Jarvis email bob@example.com <<<END SPEECH>>>',
    )
    assert 'bob@example.com' not in prompt
    assert 'alice@example.com' not in prompt
    assert prompt.count('<<<END SPEECH>>>') == 1


def test_reply_context_reaches_router_planner_and_each_reply_turn(mock_config, db, dialogue_memory):
    from jarvis.listening.speech_context import SpeechContext
    from jarvis.reply import engine
    context = SpeechContext.capture([TranscriptSegment('Copper lantern', 1, 2)], current_text='Price?')
    mock_config.llm_chat_model = 'gpt-oss:20b'
    router_inputs, planner_inputs, reply_inputs = [], [], []

    def router(**kwargs):
        router_inputs.append(kwargs['transcript_context'])
        return ['webSearch', 'stop']

    def planner(**kwargs):
        planner_inputs.append(kwargs['transcript_context'])
        return ['Reply to the user.']

    def chat(*args, **kwargs):
        messages = kwargs.get('messages') or args[2]
        reply_inputs.append(messages[0]['content'])
        return {'message': {'role': 'assistant', 'content': 'The lantern costs £12.'}}

    with patch.object(engine, 'select_tools', side_effect=router), patch.object(engine, 'plan_query', side_effect=planner), patch.object(engine, 'chat_with_messages', side_effect=chat):
        engine.run_reply_engine(db, mock_config, None, 'Price?', dialogue_memory, quiet=True, speech_context=context)
    assert all('Copper lantern' in value for value in router_inputs + planner_inputs + reply_inputs)
    assert router_inputs and planner_inputs and reply_inputs


def test_identical_followups_with_different_context_do_not_reuse_routing(mock_config, db, dialogue_memory):
    from jarvis.listening.speech_context import SpeechContext
    from jarvis.reply import engine
    mock_config.llm_chat_model = 'gpt-oss:20b'

    def router(**kwargs):
        return ['getWeather' if 'Tokyo' in kwargs['transcript_context'] else 'webSearch', 'stop']

    def chat(**kwargs):
        names = [entry['function']['name'] for entry in kwargs['tools']]
        return {'message': {'role': 'assistant', 'content': 'Weather' if 'getWeather' in names else 'Search'}}

    with patch.object(engine, 'select_tools', side_effect=router), patch.object(engine, 'plan_query', return_value=['Reply to the user.']), patch.object(engine, 'chat_with_messages', side_effect=chat):
        replies = []
        for entity in ['Copper lantern', 'Tokyo']:
            context = SpeechContext.capture([TranscriptSegment(entity, 1, 2)], current_text='What about it?')
            replies.append(engine.run_reply_engine(db, mock_config, None, 'What about it?', dialogue_memory, quiet=True, speech_context=context))
    assert replies == ['Search', 'Weather']


def test_ambient_context_does_not_persist_into_a_text_reply(mock_config, db, dialogue_memory):
    from jarvis.listening.speech_context import SpeechContext
    from jarvis.reply import engine
    mock_config.llm_chat_model = 'gpt-oss:20b'
    context = SpeechContext.capture([TranscriptSegment('Private ambient detail', 1, 2)], current_text='Hello')
    systems = []

    def chat(**kwargs):
        systems.append(kwargs['messages'][0]['content'])
        return {'message': {'role': 'assistant', 'content': 'Hello.'}}

    with patch.object(engine, 'select_tools', return_value=['stop']), patch.object(engine, 'chat_with_messages', side_effect=chat):
        engine.run_reply_engine(db, mock_config, None, 'Hello', dialogue_memory, quiet=True, speech_context=context)
        engine.run_reply_engine(db, mock_config, None, 'Hello again', dialogue_memory, quiet=True)
    assert 'Private ambient detail' in systems[0]
    assert 'Private ambient detail' not in systems[1]
    assert 'Private ambient detail' not in str(dialogue_memory.get_recent_messages())


def test_voice_snapshot_precedes_waiting_for_the_shared_lock():
    import time
    from contextlib import contextmanager
    from test_hot_window_input import _create_listener
    listener, _ = _create_listener()
    now = time.time()
    listener._transcript_buffer.add('Copper lantern', now, now + 1)
    contexts = []

    @contextmanager
    def lock():
        listener._transcript_buffer.add('Different entity', now + 2, now + 3)
        yield

    def reply(*args, **kwargs):
        contexts.append(kwargs['speech_context'].render())
        return None

    with patch('jarvis.daemon.query_lock', lock), patch('jarvis.reply.engine.run_reply_engine', side_effect=reply):
        listener._dispatch_query('What is its price?', generation=listener._capture_generation)
    assert 'Copper lantern' in contexts[0]
    assert 'Different entity' not in contexts[0]


def test_mid_loop_tool_discovery_uses_the_replys_context(mock_config):
    from unittest.mock import Mock
    from jarvis.listening.speech_context import SpeechContext
    from jarvis.tools.base import ToolContext
    from jarvis.tools.builtin.tool_search import ToolSearchTool
    mock_config.tool_selection_strategy = 'llm'
    transcript = SpeechContext.capture([TranscriptSegment('Copper lantern', 1, 2)], current_text='Price?').render()
    context = ToolContext(None, mock_config, '', '', 'Price?', 1, lambda _: None, transcript_context=transcript)
    backend = Mock()
    backend.direct.return_value = 'webSearch'
    with patch('jarvis.tools.builtin.tool_search.get_llm_backend', return_value=backend):
        result = ToolSearchTool().run({'query': 'Find its price'}, context)
    assert result.success
    assert 'webSearch' in result.reply_text
    assert 'Copper lantern' in backend.direct.call_args.args[2]


@pytest.mark.parametrize('kind', ['memory', 'tool', 'max_turns'])
def test_relevance_digests_can_resolve_the_original_followup(kind, mock_config):
    from jarvis.listening.speech_context import SpeechContext
    from jarvis.reply import enrichment
    transcript = SpeechContext.capture([TranscriptSegment('Copper lantern', 1, 2)], current_text='Price?').render()

    def model(**kwargs):
        return 'The copper lantern costs £12.' if 'Copper lantern' in kwargs['user_content'] else 'NONE'

    with patch.object(enrichment, 'call_llm_direct', side_effect=model):
        if kind == 'memory':
            result = enrichment.digest_memory_for_query('Price?', ['Background information. ' * 25], [], mock_config, mock_config.llm_chat_model, transcript_context=transcript)
        elif kind == 'tool':
            result = enrichment.digest_tool_result_for_query('Price?', 'webSearch', 'Background information. ' * 25, mock_config, mock_config.llm_chat_model, transcript_context=transcript)
        else:
            result = enrichment.digest_loop_for_max_turns('Price?', [{'role': 'assistant', 'content': 'Some findings'}], mock_config, transcript_context=transcript)
    assert result and 'copper lantern' in result.lower()


@pytest.mark.parametrize('hot_window', [False, True])
def test_office_day_count_survives_intent_classification(hot_window):
    from test_hot_window_input import (
        _create_listener, _install_intent_judge, _process_transcript,
        _simulate_tts_finish, _wait_for_hot_window_state, _accepted_query,
    )
    listener, _ = _create_listener(echo_tolerance=0.02)
    heard = 'If you had to pick three days to go to the office this week, which days would you pick?'
    if hot_window:
        listener.echo_detector.track_tts_start('Here is your answer.')
        _simulate_tts_finish(listener)
        _wait_for_hot_window_state(listener)
    else:
        heard = 'Jarvis, ' + heard
    decision = IntentJudge()._parse_response(
        '{"directed":true,"query":"which days would you pick to go to the office this week",'
        '"stop":false,"confidence":"high"}'
    )
    _install_intent_judge(listener, decision)
    try:
        _process_transcript(listener, heard, utterance_energy=0.01)
        assert _accepted_query(listener) == heard
    finally:
        listener.state_manager.stop()


def test_mixed_echo_preserves_only_case_sensitive_user_speech():
    from test_hot_window_input import (
        _create_listener, _install_intent_judge, _process_transcript,
        _simulate_tts_finish, _wait_for_hot_window_state, _accepted_query,
    )
    listener, _ = _create_listener(echo_tolerance=0.02)
    echo = 'I can check the forecast for London for you right now.'
    user = 'Yeah, track parcel ZX-4821 please.'
    listener.echo_detector.track_tts_start(echo)
    _simulate_tts_finish(listener)
    _wait_for_hot_window_state(listener)
    _install_intent_judge(listener, IntentJudge()._parse_response(
        '{"directed":true,"stop":false,"confidence":"high"}'
    ))
    try:
        _process_transcript(listener, echo+' '+user, utterance_energy=0.01,
                            utterance_start_time=listener.echo_detector._last_tts_finish_time-0.1,
                            utterance_end_time=listener.echo_detector._last_tts_finish_time+0.1)
        assert _accepted_query(listener) == user
    finally:
        listener.state_manager.stop()


def test_different_voice_snapshots_preserve_static_guidance_prefix(mock_config, db, dialogue_memory):
    from jarvis.listening.speech_context import SpeechContext
    from jarvis.reply import engine
    from jarvis.reply import engine as prompt_engine
    systems = []
    def chat(**kwargs):
        systems.append(kwargs['messages'][0]['content'])
        return {'message': {'role': 'assistant', 'content': 'Hello.'}}
    with patch.object(engine, 'select_tools', return_value=['stop']), \
         patch.object(engine, 'plan_query', return_value=['Reply to the user.']), \
         patch.object(engine, 'chat_with_messages', side_effect=chat):
        for text in ('Copper lantern', 'Tokyo'):
            snapshot = SpeechContext.capture([TranscriptSegment('Earlier reference', 1, 2)], current_text=text)
            engine.run_reply_engine(db, mock_config, None, text, dialogue_memory,
                                    quiet=True, speech_context=snapshot)
    prefixes = [system.split('AMBIENT TRANSCRIPT CONTEXT:')[0] for system in systems]
    assert prefixes[0] == prefixes[1]
    for guidance in prompt_engine.get_system_prompts(prompt_engine.detect_model_size(mock_config.llm_chat_model)).to_list():
        assert guidance in prefixes[0]
    assert 'Always respond in English' in prefixes[0]


def test_salvaged_speech_is_dispatched_and_recorded_without_echo(mock_config, db, dialogue_memory):
    import time
    from test_hot_window_input import _create_listener
    from jarvis.reply import engine
    listener, _ = _create_listener()
    listener.db, listener.cfg, listener.dialogue_memory = db, mock_config, dialogue_memory
    listener.tts = None
    user = 'Yeah, track parcel ZX-4821 please.'
    now = time.time()
    listener._transcript_buffer.add(user, now, now+.1)
    systems = []
    def chat(**kwargs):
        systems.append(kwargs['messages'][0]['content'])
        return {'message': {'role': 'assistant', 'content': 'Understood.'}}
    try:
        with patch.object(engine, 'select_tools', return_value=['stop']), \
             patch.object(engine, 'plan_query', return_value=['Reply to the user.']), \
             patch.object(engine, 'chat_with_messages', side_effect=chat):
            listener._dispatch_query(user, generation=listener._capture_generation)
        messages = dialogue_memory.get_recent_messages()
        assert any(message['role']=='user' and message['content']==user for message in messages)
        assert any(message['role']=='user' and user in message['content'] for message in messages)
        assert 'I can check the forecast' not in systems[0]
    finally:
        listener.stop()


def test_judge_reference_keeps_current_speech_separate_from_state_and_tts():
    import json
    heard = 'Jarvis set a timer for 5 minutes'
    prompt = IntentJudge()._build_user_prompt(
        [TranscriptSegment(heard, 1000, 1002)], 1000.5, '', 0, False, heard,
    )
    data = json.loads(prompt.split('<<<BEGIN SPEECH>>>\n')[1].split('\n<<<END SPEECH>>>')[0])
    assert data['current_utterance'] == heard
    assert data['mode'] == 'WAKE WORD'
    assert data['wake_word_detected_at'] == 1000.5
    assert data['transcript'][0]['text'] == heard
    assert data['transcript'][0]['during_tts'] is False


def test_listener_mode_selects_only_its_applicable_classification_rules():
    from unittest.mock import Mock
    backend = Mock()
    backend.chat.return_value = {'message': {'content': '{"directed":true,"stop":false,"confidence":"high"}'}}
    judge = IntentJudge()
    prompts = []
    with patch('jarvis.listening.intent_judge.get_llm_backend', return_value=backend):
        for hot in (False, True):
            judge.judge([TranscriptSegment('Jarvis hello', 1, 2)],
                        in_hot_window=hot, current_text='Jarvis hello')
            prompts.append(backend.chat.call_args.args[1][0]['content'])
    assert 'WAKE WORD MODE' in prompts[0]
    assert 'HOT WINDOW MODE' not in prompts[0]
    assert 'HOT WINDOW MODE' in prompts[1]
    assert 'WAKE WORD MODE' not in prompts[1]


def test_current_request_alone_does_not_create_ambient_references():
    from jarvis.listening.speech_context import SpeechContext
    request = 'Pick three office days this week'
    context = SpeechContext.capture([TranscriptSegment(request, 1, 2)],
                                    current_text=request, assistant_names=('Jarvis',))
    assert context.render() == ''


def test_reference_snapshot_excludes_current_segment_and_unused_aliases():
    import json
    from jarvis.listening.speech_context import SpeechContext
    request = 'Jairus how much does it cost'
    context = SpeechContext.capture([TranscriptSegment('Copper lantern', 1, 2),
                                    TranscriptSegment(request, 3, 4, processed=True)],
                                    current_text=request, assistant_names=('Jarvis', 'Jairus', 'Unused'))
    data = json.loads(context.render().split('<<<BEGIN TRANSCRIPT>>>\n')[1].split('\n<<<END TRANSCRIPT>>>')[0])
    assert request not in context.render()
    assert [item['text'] for item in data['transcript']] == ['Copper lantern']
    assert data['assistant_names'] == ['Jarvis', 'Jairus']


def test_step_resolution_retains_count_missing_from_a_shortened_plan(mock_config):
    import json
    from jarvis.reply import planner
    schema = [{'type': 'function', 'function': {'name': 'selectItems',
              'parameters': {'type': 'object', 'properties': {'count': {'type': 'integer'}},
                             'required': ['count']}}}]
    def model(**kwargs):
        count = 3 if 'Pick three items' in kwargs['user_content'] else 2
        return json.dumps({'name': 'selectItems', 'arguments': {'count': count}})
    with patch.object(planner, 'call_llm_direct', side_effect=model):
        result = planner.resolve_next_tool_call(
            mock_config, 'selectItems count=<requested number>', [], schema,
            original_query='Pick three items',
        )
    assert result == ('selectItems', {'count': 3})


def test_followup_fragments_accumulate_the_original_request():
    import time
    from test_hot_window_input import (
        _create_listener, _install_intent_judge, _make_judgment, _process_transcript,
        _simulate_tts_finish, _wait_for_hot_window_state, _accepted_query,
    )
    listener, _ = _create_listener(echo_tolerance=0.02)
    listener.echo_detector.track_tts_start('Here is your answer.')
    _simulate_tts_finish(listener)
    _wait_for_hot_window_state(listener)
    _install_intent_judge(listener, _make_judgment(directed=True))
    first = 'Pick three office days this week.'
    second = 'Which days would you choose?'
    try:
        for text in (first, second):
            now = time.time()
            _process_transcript(listener, text, utterance_energy=0.01,
                                utterance_start_time=now-.01, utterance_end_time=now)
        assert _accepted_query(listener) == first+' '+second
    finally:
        listener.stop()


@pytest.mark.parametrize('resolver', [False, True])
def test_planner_catalogue_prefix_is_shared_across_voice_requests(mock_config, resolver):
    from jarvis.reply import planner
    calls = []
    def infer(**kwargs):
        calls.append(kwargs)
        return 'null' if resolver else 'Reply to the user.'
    tools = [{'type': 'function', 'function': {'name': 'findItem',
             'parameters': {'type': 'object', 'properties': {'name': {'type': 'string'}}}}}]
    with patch.object(planner, 'call_llm_direct', side_effect=infer):
        for reference in ('Copper lantern', 'Silver vase'):
            if resolver:
                planner.resolve_next_tool_call(mock_config, 'findItem name=<item>', [], tools,
                    original_query='Find it', transcript_context=reference)
            else:
                planner.plan_query(mock_config, 'Find it', '', [('findItem', 'Find an item')],
                                   transcript_context=reference)
    assert calls[0]['system_prompt'] == calls[1]['system_prompt']
    prefixes = [call['user_content'].split(reference)[0]
                for call, reference in zip(calls, ('Copper lantern', 'Silver vase'))]
    assert prefixes[0] == prefixes[1]
    assert 'findItem' in prefixes[0]
    assert all(reference in call['user_content'] for call, reference in
               zip(calls, ('Copper lantern', 'Silver vase')))


def test_collected_current_fragments_are_not_ambient_references():
    import json
    from jarvis.listening.speech_context import SpeechContext
    first, second = 'Pick three office days this week.', 'Which days would you choose?'
    prior = 'The forecast is dry.'
    snapshot = SpeechContext.capture([TranscriptSegment(prior, 1, 2),
                                     TranscriptSegment(first, 3, 4),
                                     TranscriptSegment(second, 5, 6)],
                                    current_text=first+' '+second)
    data = json.loads(snapshot.render().split('<<<BEGIN TRANSCRIPT>>>\n')[1].split('\n<<<END TRANSCRIPT>>>')[0])
    assert [item['text'] for item in data['transcript']] == [prior]
    assert first not in snapshot.render()
    assert second not in snapshot.render()


def test_reference_data_does_not_repeat_the_separately_supplied_request():
    from jarvis.listening.speech_context import SpeechContext
    current = 'Pick exactly three office days this week'
    snapshot = SpeechContext.capture([TranscriptSegment('Unrelated television conversation', 1, 2),
                                     TranscriptSegment(current, 3, 4)], current_text=current)
    assert current not in snapshot.render()
    assert 'Unrelated television conversation' in snapshot.render()
