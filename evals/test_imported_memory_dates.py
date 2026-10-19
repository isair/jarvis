"""Imported plans retain date uncertainty without discarding established facts."""
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
import json
import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import voice_config
from jarvis.llm import get_llm_backend
from jarvis.memory.graph import GraphMemoryStore
from jarvis.memory.graph_ops import build_warm_profile, format_warm_profile_block
from jarvis.memory.imports import preview_text, ImportChoice, commit_import
from jarvis.system_prompt import build_system_prompt
from jarvis.utils.time_context import format_time_context

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize('reference', [None, 'past', 'current'])
@pytest.mark.parametrize('sample', range(3))
def test_imported_plan_is_not_rebased_to_import_or_current_time(tmp_path, reference, sample):
    today = date(2026, 10, 12)
    tomorrow = today + timedelta(days=1)
    outing = 'The user has an outing planned for tomorrow in South London to see otters.'
    if reference:
        day = tomorrow if reference == 'current' else today - timedelta(days=1)
        outing = f'The user has an outing planned for {day.isoformat()} in South London to see otters.'
    text = '\n'.join(['The user lives in London.',
                      'The user goes to the office three days a week.', outing])
    preview = preview_text('Claude', text, label='Synthetic memory export')
    with closing(GraphMemoryStore(str(tmp_path/'graph.db'))) as store:
        commit_import(store, preview, [ImportChoice(preview.items[0].id, text, 'user', source_date=None)])
        profile = format_warm_profile_block(build_warm_profile(store))
    cfg = voice_config()
    clock = format_time_context('Europe/London', now_utc=datetime.combine(today, datetime.min.time(), timezone.utc))
    answer = get_llm_backend(cfg).direct(
        cfg.llm_chat_model,
        build_system_prompt()+'\n'+profile+f'\n[Context: {clock}, Location: Unknown]',
        'From your saved information, is an outing established for tomorrow? '
        'Return only a JSON object with "outing_tomorrow" (true, false or null '
        'if the date cannot be established), "home" (city), and "office_days" '
        '(number of days per week).',
        timeout_sec=60, temperature=0.0, max_tokens=1024,
    )
    assert answer, 'No reply'
    start = answer.find('{')
    assert start >= 0, answer
    result, _ = json.JSONDecoder().raw_decode(answer[start:])
    assert result == {'outing_tomorrow': {'current': True, 'past': False, None: None}[reference],
                      'home': 'London', 'office_days': 3}, answer
