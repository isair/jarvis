"""Native confirmation delivery preserves admitted counts and unrelated answers."""
import time

import pytest

from evals.helpers import isolated_voice_settings, judge_pass_fail
from jarvis.reply.delivery import render_written_reply
from jarvis.reply.timer_delivery import timer_reply_source

pytestmark = pytest.mark.eval


@pytest.mark.parametrize('query,draft,language,labels,additional', [
    ('Start a five-minute tea timer.', 'Your tea timer is up.', 'en', ['Tea'], ''),
    ('Stelle einen Timer für fünf Minuten für Tee. Antworte auf Englisch.', 'Your tea timer is up.', 'en', ['Tee'], ''),
    ('Start a five-minute tea timer and tell me the capital of France.', 'Your tea timer is up. Paris is the capital of France.', 'en', ['Tea'], 'The capital of France is Paris.'),
    ('Stelle einen Timer für fünf Minuten für Tee und nenne die Hauptstadt von Frankreich.', 'Your tea timer is up. Paris is the capital of France.', 'de', ['Tee'], 'The capital of France is Paris.'),
    ('Start a five-minute tea timer and tell me the capital of France.', 'Your tea timer is up; Paris is the capital of France.', 'en', ['Tea'], 'The capital of France is Paris.'),
    ('Start two separate five-minute timers for tea and soup.', 'One timer has already elapsed.', 'en', ['tea', 'soup'], ''),
])
def test_confirmation_uses_records_and_retains_other_requested_facts(query, draft, language, labels, additional):
    cfg = isolated_voice_settings()
    records = [{'id': f'private-{index}', 'label': label, 'duration_sec': 300, 'state': 'running'} for index, label in enumerate(labels)]
    deadline = time.monotonic() + cfg.llm_chat_timeout_sec
    source = timer_reply_source(cfg, query, draft, records, deadline)
    result = render_written_reply(cfg, query, source, protect_numbers=True, deadline=deadline)
    assert result
    assert all(record['id'] not in result for record in records)
    assert judge_pass_fail(
        f'The reply must be in language {language} and confirm {len(records)} running five-minute countdowns. '
        f'Each named purpose is preserved: {labels}. It must not claim expiry. {additional}', result,
    ) == 'PASS'
