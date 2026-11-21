"""Empty diary preparation avoids inference without suppressing available recall."""
import pytest

from jarvis.reply import engine

pytestmark = pytest.mark.unit


@pytest.fixture
def reply(monkeypatch, mock_config, db, dialogue_memory):
    mock_config.llm_chat_model = 'gpt-oss:20b'
    mock_config.embedding_model = ''
    mock_config.planner_enabled = False
    mock_config.memory_digest_enabled = False
    mock_config.tool_result_digest_enabled = False
    monkeypatch.setattr(engine, 'select_tools', lambda **kwargs: ['stop'])
    monkeypatch.setattr(engine, 'plan_query', lambda **kwargs: [])
    monkeypatch.setattr(engine, '_live_time_location_string', lambda cfg: '')
    monkeypatch.setattr('jarvis.memory.graph_ops.build_warm_profile', lambda store: {})
    monkeypatch.setattr('jarvis.memory.graph_ops.format_warm_profile_block', lambda profile: '')

    def chat(cfg, messages, **kwargs):
        context = '\n'.join(message.get('content', '') for message in messages if message['role'] == 'system')
        return {'message': {'content': 'Orion' if 'RECALL_PROJECT_ORION' in context else 'Ready'}}

    monkeypatch.setattr(engine, 'chat_with_messages', chat)
    return lambda: engine.run_reply_engine(db, mock_config, None, 'What project did I mention?', dialogue_memory)


def test_empty_diary_reply_needs_no_memory_inference(monkeypatch, reply):
    def unavailable(*args, **kwargs):
        pytest.fail('An empty diary must not require memory preparation inference')

    monkeypatch.setattr(engine, 'extract_search_params_for_memory', unavailable)
    assert reply() == 'Ready'


def test_new_diary_data_is_recalled_after_an_empty_turn(monkeypatch, db, reply):
    monkeypatch.setattr(engine, 'extract_search_params_for_memory',
                        lambda *args, **kwargs: {'keywords': ['RECALL_PROJECT_ORION']})
    assert reply() == 'Ready'
    db.upsert_conversation_summary('2020-01-01', 'RECALL_PROJECT_ORION', 'project', source_app='fixture')
    assert reply() == 'Orion'


@pytest.mark.parametrize('source', ['all', 'graph'])
def test_empty_diary_does_not_suppress_other_sources(monkeypatch, mock_config, reply, source):
    mock_config.memory_enrichment_source = source

    # Raising a sentinel proves the preparation boundary remains reachable.
    class PreparationReached(BaseException):
        pass

    def prepare(*args, **kwargs):
        raise PreparationReached

    monkeypatch.setattr(engine, 'extract_search_params_for_memory', prepare)
    with pytest.raises(PreparationReached):
        reply()


def test_diary_presence_tracks_committed_rows(db):
    assert db.has_conversation_summaries() is False
    db.upsert_conversation_summary('2020-01-01', 'An older saved fact.', '', source_app='fixture')
    assert db.has_conversation_summaries() is True
    db.conn.execute('DELETE FROM conversation_summaries')
    db.conn.commit()
    assert db.has_conversation_summaries() is False


def test_unavailable_diary_check_preserves_preparation(monkeypatch, db, reply):
    def unavailable():
        raise OSError('Owned read failure')

    class PreparationReached(BaseException):
        pass

    def prepare(*args, **kwargs):
        raise PreparationReached

    monkeypatch.setattr(db, 'has_conversation_summaries', unavailable)
    monkeypatch.setattr(engine, 'extract_search_params_for_memory', prepare)
    with pytest.raises(PreparationReached):
        reply()
