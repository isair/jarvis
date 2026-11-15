"""Text requests preserve timer identity and expiry language without Whisper metadata."""
import pytest
from evals.test_timer_workflow import countdowns, reply_to
from evals.helpers import judge_pass_fail

pytestmark = pytest.mark.eval


@pytest.mark.parametrize('query,language', [
    ('Start a five minute timer for tea.', 'en'),
    ('Çay için beş dakikalık bir zamanlayıcı başlat.', 'tr'),
])
def test_text_entry_creates_one_timer_with_localised_expiry(countdowns, query, language):
    reply = reply_to(countdowns, query, None)
    entries = countdowns[0].list()
    assert len(entries) == 1
    assert entries[0].duration_sec == 300
    assert entries[0].announcement
    assert judge_pass_fail(
        f'The expiry announcement must be in language {language}, and say the timer has elapsed. '
        'Natural phrasing is valid; an English announcement fails for a non-English request.',
        entries[0].announcement,
    ) == 'PASS'
    assert judge_pass_fail(
        f'The reply must be in language {language}, acknowledge a running five-minute countdown, '
        'and must not claim the countdown has elapsed.', reply,
    ) == 'PASS'


def test_written_override_does_not_change_the_source_language_of_expiry(countdowns):
    reply = reply_to(countdowns, 'Stelle einen Timer für fünf Minuten für Tee. Antworte auf Englisch.', None)
    entries = countdowns[0].list()
    assert len(entries) == 1
    assert entries[0].duration_sec == 300
    assert judge_pass_fail('The announcement must be in German and state that the countdown has elapsed.', entries[0].announcement) == 'PASS'
    assert judge_pass_fail('The reply must be in English and confirm a running five-minute countdown, without claiming expiry.', reply) == 'PASS'


def test_named_creation_can_be_cancelled_in_the_shared_conversation(countdowns):
    reply_to(countdowns, 'Çay için beş dakikalık bir zamanlayıcı başlat.', None)
    entries = countdowns[0].list()
    assert len(entries) == 1
    assert entries[0].label and entries[0].label.casefold() == 'çay'
    countdowns[0].start(3600, 'çorba')
    reply_to(countdowns, 'Çay zamanlayıcısını iptal et, çorba zamanlayıcısını çalışır bırak.', None)
    assert {entry.label for entry in countdowns[0].list()} == {'çorba'}
