"""Native reply generation schedules and cancels the requested countdowns."""
from dataclasses import replace

import pytest

from evals.helpers import isolated_voice_settings, judge_pass_fail, assert_not_fallback_reply
from jarvis.memory.conversation import DialogueMemory
from jarvis.memory.db import Database
from jarvis.reply import engine
from jarvis.tools.builtin import timer

pytestmark = pytest.mark.eval


@pytest.fixture
def countdowns(monkeypatch):
    manager = timer.TimerManager(announcer=lambda entry, on_terminal: None)
    monkeypatch.setattr(timer, '_manager_instance', manager)
    monkeypatch.setattr(timer, '_alarm_registry_instance', timer.AlarmRegistry())
    monkeypatch.setattr(timer, '_emit_alarm_event', lambda *args: None)
    monkeypatch.setattr(timer.AlarmRegistry, '_stdout_is_tty', staticmethod(lambda: False))
    # Isolate execution quality once the timer capability has been selected.
    # Routing accuracy remains a separate qualification.
    monkeypatch.setattr(engine, 'select_tools', lambda *args, **kwargs: ['timer', 'stop'])
    db = Database(':memory:')
    cfg = replace(isolated_voice_settings(), location_enabled=False, mcps={},
                  llm_chat_timeout_sec=60, planner_timeout_sec=15)
    yield manager, db, cfg, DialogueMemory()
    assert manager.shutdown(timeout_sec=2)
    db.close()


def reply_to(runtime, query, language):
    manager, db, cfg, dialogue = runtime
    reply = engine.run_reply_engine(db, cfg, None, query, dialogue, language=language)
    assert reply, '⏲️ A timer request must produce a reply'
    assert_not_fallback_reply(reply, 'timer workflow')
    return reply


@pytest.mark.parametrize('query,language,seconds', [
    ('Start a one hour and fifteen minute timer for tea.', 'en', 4500),
    ('Çay için on beş dakikalık bir zamanlayıcı başlat.', 'tr', 900),
    ('Stelle einen Timer für fünfundzwanzig Minuten für Tee.', 'de', 1500),
    ('Pon un temporizador de cuarenta y cinco minutos para el té.', 'es', 2700),
    ('Set a timer for twelve and a half minutes.', 'en', 750),
    ('Start a timer for six hundred seconds.', 'en', 600),
])
def test_timer_creation_preserves_duration_and_reports_a_countdown(countdowns, query, language, seconds, record_property):
    reply = reply_to(countdowns, query, language)
    entries = countdowns[0].list()
    record_property('reply', reply)
    assert len(entries) == 1, f'⏲️ Expected one countdown, received {len(entries)}'
    assert entries[0].duration_sec == seconds
    record_property('announcement', entries[0].announcement)
    if language != 'en':
        assert entries[0].announcement, '⏲️ A non-English request needs a localised expiry announcement'
        assert judge_pass_fail(
            f'The expiry announcement must be in language {language}, and say the timer has elapsed. '
            'Natural phrasing is valid; an English announcement fails.',
            entries[0].announcement,
        ) == 'PASS'
    assert judge_pass_fail(
        f'The reply must be in language {language} and acknowledge that a countdown was scheduled. '
        'The timer is still running, so claiming it has elapsed, finished or is already up fails. '
        'If a duration is mentioned, it must match the requested duration.',
        f'Request: {query}\nScheduled duration: {seconds} seconds\nReply: {reply}',
    ) == 'PASS'


@pytest.mark.parametrize('query,remaining', [
    ('Cancel only the tea timer and keep the soup timer running.', {'soup'}),
    ('Cancel every timer.', set()),
    ('Which timers are currently running?', {'tea', 'soup'}),
])
def test_timer_followup_preserves_requested_scope(countdowns, query, remaining, record_property):
    manager = countdowns[0]
    for label in ('tea', 'soup'):
        manager.start(3600, label)
    reply = reply_to(countdowns, query, 'en')
    record_property('reply', reply)
    assert {entry.label for entry in manager.list()} == remaining
    assert judge_pass_fail(
        'The reply must agree with the actual remaining countdowns. It must not claim a still-running '
        'timer was cancelled, or that a cancelled timer is still running. A listing request should '
        'identify both labels. Judge the response meaning, not particular wording.',
        f'Request: {query}\nRemaining labels: {sorted(remaining)}\nReply: {reply}',
    ) == 'PASS'
