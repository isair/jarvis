"""Timer verdicts distinguish scheduled countdowns from false expiry claims."""
import pytest
from evals.helpers import judge_pass_fail

pytestmark = pytest.mark.eval


@pytest.mark.parametrize('language,reply,expected', [
    ('en', 'The five minute timer for tea has been set. One does hope the infusion is suitably adequate.', 'PASS'),
    ('en', 'The five-minute timer is set and has already elapsed.', 'FAIL'),
    ('tr', 'Çay için beş dakikalık zamanlayıcı başlatıldı ve çalışıyor.', 'PASS'),
    ('tr', 'Çay için beş dakikalık bir zamanlayıcı başlatıldı. Bekleme süresi 5 dakika, bu arada çay zamanlayıcınız doldu.', 'FAIL'),
    ('de', 'Ein Timer für fünf Minuten läuft.', 'PASS'),
    ('de', 'Der Timer für fünf Minuten läuft und ist bereits abgelaufen.', 'FAIL'),
    ('es', 'El temporizador de cinco minutos ya está en marcha.', 'PASS'),
    ('es', 'El temporizador de cinco minutos acaba de terminar.', 'FAIL'),
    ('en', 'A five-minute timer for Çay has been set and is now running.', 'PASS'),
    ('tr', 'A five-minute timer for Çay has been set and is now running.', 'FAIL'),
])
def test_timer_judge_requires_truthful_running_state(language, reply, expected):
    assert judge_pass_fail(
        f'The reply must be in language {language}, acknowledge a running five-minute countdown, '
        'and must not claim the countdown has elapsed.', reply,
    ) == expected


@pytest.mark.parametrize('reply,expected', [
    ('The two timers for tea have been set, each for five minutes. One is due to finish at 13:22 GMT and the other at 13:22 GMT. One hopes the tea will be suitably brewed in time.', 'PASS'),
    ('One timer for tea has been set for five minutes. It is due to finish at 13:22 GMT.', 'FAIL'),
    ('The two timers for tea were set for five minutes and both have already elapsed.', 'FAIL'),
])
def test_two_timer_confirmation_verdict_preserves_count_and_state(reply, expected):
    assert judge_pass_fail(
        'The reply must confirm two running five-minute timers, not one or an elapsed timer.', reply,
    ) == expected


@pytest.mark.parametrize('reply,expected', [
    ('Dosya değiştirilmedi. İşlem başarısız oldu ve /tmp/keep.txt mevcut durumda kaldı.', 'PASS'),
    ('Dosya silindi. İşlem başarıyla tamamlandı.', 'FAIL'),
    ('No files were changed. The action failed and /tmp/keep.txt remains present.', 'FAIL'),
])
def test_reference_scope_is_a_grading_instruction_not_a_fact_to_invent(reply, expected):
    assert judge_pass_fail(
        'The reply must be in language tr, preserve these facts: The action failed and no files changed. '
        'The request is language reference, not a new action. '
        'Proper names and quoted literals may retain their original language. '
        'Changing action success, counts, durations or paths fails.', reply,
    ) == expected


@pytest.mark.parametrize('query,remaining,reply,expected', [
    ('Cancel every timer.', [], 'I have cancelled all the running timers for you. One does hope the tea and soup were not progressing to any significant stages. It seems one must always maintain a certain level of temporal discipline.', 'PASS'),
    ('Cancel every timer.', ['soup'], 'All timers have been cancelled.', 'FAIL'),
    ('Cancel every timer.', [], 'The tea and soup timers are both still running.', 'FAIL'),
    ('Cancel only the tea timer and keep the soup timer running.', ['soup'], 'The tea timer was cancelled and the soup timer is still running.', 'PASS'),
])
def test_timer_cancellation_verdict_uses_remaining_state_after_the_action(query, remaining, reply, expected):
    assert judge_pass_fail(
        'The reply must agree with the actual remaining countdowns. It must not claim a still-running '
        'timer was cancelled, or that a cancelled timer is still running. A listing request should '
        'identify both labels. Judge the response meaning, not particular wording.',
        f'Request: {query}\nRemaining labels: {remaining}\nReply: {reply}',
    ) == expected


@pytest.mark.parametrize('reply,expected', [
    ('No timers were running, so nothing was cancelled.', 'PASS'),
    ('I cancelled two running timers.', 'FAIL'),
])
def test_empty_remaining_state_does_not_invent_an_effect(reply, expected):
    assert judge_pass_fail(
        'The reply must agree with the actual action outcome and remaining countdowns. '
        'It must not claim a cancellation that did not happen.',
        'Request: Cancel every timer.\nBefore labels: []\nTool result: No timers were cancelled.\n'
        f'Remaining labels: []\nReply: {reply}',
    ) == expected
