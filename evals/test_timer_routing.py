"""Countdown requests reach the timer through the production tool router."""
import pytest
from evals.test_timer_workflow import countdowns, reply_to
from evals.helpers import judge_pass_fail
from jarvis.reply import engine
from jarvis.tools.selection import select_tools

pytestmark = pytest.mark.eval


@pytest.mark.parametrize('query,language,seconds', [
    ('Start a five minute timer for tea.', 'en', 300),
    ('Çay için beş dakikalık bir zamanlayıcı başlat.', 'tr', 300),
    ('Stelle einen Timer für fünf Minuten für Tee.', 'de', 300),
])
def test_timer_request_uses_production_routing(countdowns, monkeypatch, query, language, seconds):
    monkeypatch.setattr(engine, 'select_tools', select_tools)
    reply = reply_to(countdowns, query, language)
    entries = countdowns[0].list()
    assert len(entries) == 1
    assert entries[0].duration_sec == seconds
    assert judge_pass_fail(
        f'The reply must be in language {language}, acknowledge a running five-minute countdown, '
        'and must not claim the countdown has elapsed.', reply,
    ) == 'PASS'


def test_two_explicitly_requested_matching_timers_remain_distinct(countdowns, monkeypatch):
    monkeypatch.setattr(engine, 'select_tools', select_tools)
    reply = reply_to(countdowns, 'Start two separate timers for tea, each five minutes.', 'en')
    entries = countdowns[0].list()
    assert len(entries) == 2
    assert len({entry.id for entry in entries}) == 2
    assert all(entry.duration_sec == 300 for entry in entries)
    assert judge_pass_fail(
        'The reply must confirm two running five-minute timers, not one or an elapsed timer.', reply,
    ) == 'PASS'
