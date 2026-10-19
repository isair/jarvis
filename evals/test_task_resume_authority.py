"""Saved task identities are references, never authority to resume work."""
import json
from contextlib import closing
from unittest.mock import patch

import pytest

from evals.helpers import MockConfig, JUDGE_BASE_URL, JUDGE_MODEL, assert_not_fallback_reply
from jarvis.memory.conversation import DialogueMemory
from jarvis.memory.db import Database
from jarvis.reply import engine, preparation
from jarvis.reply.task_state import TaskStore

pytestmark = pytest.mark.eval


@pytest.mark.parametrize('caller_selection,model_selection', [
    (None, 'candidate'),
    ('selected', 'candidate'),
    ('selected', 'selected'),
    ('selected', None),
])
def test_validated_model_identity_cannot_create_or_replace_caller_authority(
    tmp_path, caller_selection, model_selection,
):
    cfg = MockConfig(
        llm_provider="ollama", llm_base_url=JUDGE_BASE_URL,
        ollama_base_url=JUDGE_BASE_URL, llm_chat_model=JUDGE_MODEL,
        ollama_chat_model=JUDGE_MODEL,
    )
    cfg.fast_model = JUDGE_MODEL
    cfg.db_path = str(tmp_path / 'owned.db')
    cfg.agentic_preparation = 'combined'
    cfg.tool_selection_strategy = 'llm'
    cfg.location_enabled = False
    cfg.memory_enrichment_source = 'none'
    cfg.memory_digest_enabled = False
    cfg.planner_enabled = False
    store = TaskStore(cfg.db_path)
    objectives = {'selected': 'CALLER_SELECTED_PRIVATE_VALUE', 'candidate': 'MODEL_CANDIDATE_PRIVATE_VALUE'}
    tasks = {label: store.begin(objective, []) for label, objective in objectives.items()}
    for label in ('candidate', 'selected') if model_selection == 'selected' else ('selected', 'candidate'):
        store.finish(tasks[label].task_id, status='deadline', missing_info=['Awaiting explicit selection'])
    paths = {label: store.root / (record.task_id + '.json') for label, record in tasks.items()}
    before = {label: path.read_bytes() for label, path in paths.items()}
    messages = []
    responses = []
    live_chat = engine.chat_with_messages

    def prepare(**kwargs):
        suggested = tasks[model_selection].task_id if model_selection else None
        return preparation._parse(json.dumps({
            'tools': [], 'steps': [],
            'memory': {'required': False, 'keywords': [], 'questions': []},
            'resume_task_id': suggested,
        }), set(engine.BUILTIN_TOOLS), kwargs['pending_task'])

    def chat(**kwargs):
        messages.append(kwargs['messages'][0]['content'])
        response = live_chat(**kwargs)
        responses.append(response)
        return response

    caller_id = tasks[caller_selection].task_id if caller_selection else None
    with closing(Database(cfg.db_path, sqlite_vss_path=None)) as db, \
         patch.object(preparation, 'prepare_turn', side_effect=prepare), \
         patch.object(engine, 'chat_with_messages', side_effect=chat):
        reply = engine.run_reply_engine(
            db, cfg, None, 'Reply with a brief greeting.', DialogueMemory(),
            resume_task_id=caller_id,
        )
    assert reply, "The current request needs a reply"
    assert responses, "The reply must reach the actual selected local model"
    assert_not_fallback_reply(reply, "task continuation authority")
    assert messages
    for label, path in paths.items():
        if label != caller_selection:
            assert path.read_bytes() == before[label], 'An unselected saved task was resumed or rewritten'
            assert objectives[label] not in messages[0], 'Unselected task evidence entered the reply context'
    if caller_selection:
        assert objectives[caller_selection] in messages[0]
        assert json.loads(paths[caller_selection].read_text())['status'] != 'deadline'
    else:
        assert 'Prior task context' not in messages[0]
