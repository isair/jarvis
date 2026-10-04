"""Speech is preserved, with reference context scoped to one reply."""

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

from jarvis.listening.intent_judge import IntentJudge
from jarvis.listening.transcript_buffer import TranscriptSegment


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
    assert 'What is its price?' in rendered


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
        listener._dispatch_query('What is its price?')
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
