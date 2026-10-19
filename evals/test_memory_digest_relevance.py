"""Memory relevance distinguishes engagement signals from identity facts."""
import pytest

from evals.memory_digest import digest_for_eval
from evals.tool_routing import requires_judge_llm

pytestmark = [pytest.mark.eval, requires_judge_llm]
BOOKS = [
    '[2026-04-12] The user asked about the novel Harbour Notes; the assistant summarised its plot.',
    '[2026-04-11] The user asked about The Clock Garden; the assistant described its characters.',
]
CASES = [
    ('Recommend a cafe for lunch.', [
        '[2026-04-12] The user asked where to get sourdough bagels and how to make rye bread.',
        '[2026-04-11] The user asked about local coffee roasters.',
    ], ('bagel', 'sourdough', 'coffee'), False, False),
    ('What book should I read next?', BOOKS, ('harbour notes', 'clock garden'), False, False),
    ('Suggest music for tonight.', [
        '[2026-04-12] The user said they listened to Amber Choir and Low Lantern this week.',
        '[2026-04-11] The user asked about Amber Choir albums.',
    ], ('amber choir', 'low lantern'), False, False),
    ('Akşam yemeği için ne önerirsin?', [
        '[2026-04-12] Kullanıcı mercimek çorbası tarifini sordu.',
        '[2026-04-11] Kullanıcı baklava yapmayı konuştu.',
    ], ('mercimek', 'lentil', 'baklava'), False, False),
    ('What do you know about me?', BOOKS, (), True, False),
    ('Recommend a book for me.', [
        '[2026-04-12] The user asked about the weather forecast and the assistant reported rain.',
        '[2026-04-11] The user asked for the area of a rectangle and the assistant calculated it.',
    ], (), True, False),
    ('Tell me more about the novel Harbour Notes.', [
        '[2026-04-12] The user asked about Harbour Notes; the assistant said its author is Ada North.',
    ], ('harbour notes', 'ada north'), False, True),
]


@pytest.mark.parametrize('query, entries, topics, irrelevant, attributed', CASES)
def test_memory_digest_preserves_query_specific_relevance(query, entries, topics, irrelevant, attributed):
    result = digest_for_eval(query, entries)
    if irrelevant:
        assert not result, f'🧠 Unrelated topics or past Q&A became personal facts: {result}'
        return
    assert result and result.strip(), '🧠 Relevant recorded engagement must survive digestion'
    lowered = result.casefold()
    if attributed:
        assert all(topic in lowered for topic in topics), f'📋 Historical claim lost its recorded details: {result}'
        assert 'assistant' in lowered, f'📋 Historical assistant claim lost its attribution: {result}'
    else:
        assert any(topic in lowered for topic in topics), f'🧠 Relevant recorded items are missing: {result}'
        assert not any(phrase in lowered for phrase in (
            'user loves', 'user likes', 'user prefers', 'favourite', 'favorite',
        )), f'🧠 Engagement was upgraded into an unstated preference: {result}'
