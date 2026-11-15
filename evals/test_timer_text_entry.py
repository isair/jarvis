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
