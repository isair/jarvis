"""Native timer admission distinguishes new creations from preserved records."""
import pytest

from evals.helpers import isolated_voice_settings
from jarvis.reply.timer_delivery import prepare_timer_arguments

pytestmark = pytest.mark.eval


@pytest.mark.parametrize('query,count,label', [
    ('Cancel only the tea timer and keep the soup timer running.', 0, None),
    ('Çay zamanlayıcısını iptal et, çorba zamanlayıcısını çalışır bırak.', 0, None),
    ('Brich nur den Tee-Timer ab und lass den Suppen-Timer weiterlaufen.', 0, None),
    ('Cancela solo el temporizador del té y deja funcionando el de la sopa.', 0, None),
    ('List all my running timers.', 0, None),
    ('How much time remains on the soup timer?', 0, None),
    ('Keep the soup timer running and start a new five-minute tea timer.', 1, 'tea'),
    ('Çorba zamanlayıcısını çalışır bırak, çay için yeni bir beş dakikalık zamanlayıcı başlat.', 1, 'çay'),
])
def test_only_new_creations_receive_an_admission_budget(query, count, label):
    context = (
        'Earlier user request: Start a five-minute tea timer.\n'
        'Earlier assistant reply: The tea timer is running.\n'
        'Existing records: tea (Çay, Tee, té), five minutes; '
        'soup (çorba, Suppe, sopa), one hour. These are already running.'
    )
    proposal = {'action': 'set', 'hours': 0, 'minutes': 5, 'seconds': 0,
                'label': label or 'soup', 'expiry_announcement': 'The timer is up.'}
    decision = prepare_timer_arguments(
        isolated_voice_settings(), query, proposal, dialogue_context=context,
    )
    assert decision.requested_count == count, 'Only current requests to create new countdowns authorise admission'
    if count:
        assert decision.arguments['label'].casefold() == label
