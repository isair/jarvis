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
])
def test_timer_judge_requires_truthful_running_state(language, reply, expected):
    assert judge_pass_fail(
        f'The reply must be in language {language}, acknowledge a running five-minute countdown, '
        'and must not claim the countdown has elapsed.', reply,
    ) == expected
