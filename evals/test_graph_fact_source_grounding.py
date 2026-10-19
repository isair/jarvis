"""Graph review rejects invented relationships before durable fact storage."""
import json

import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import voice_config

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize(('summary', 'branch', 'fact', 'supported'), [
    ('The user asked to translate "I live in Bristol".', 'USER', 'The user lives in Bristol.', False),
    ('Kullanıcı "Ankara şehrinde yaşıyorum" cümlesini çevirmemi istedi.', 'USER', 'The user lives in Ankara.', False),
    ('The user\'s brother lives in Bristol.', 'USER', 'The user lives in Bristol.', False),
    ('The user used to live in Bristol and currently lives in York.', 'USER', 'The user lives in Bristol.', False),
    ('The user requested a vegetarian dinner.', 'USER', 'The user is vegetarian.', False),
    ('The user said "I live in Bristol" and requested tomorrow\'s weather.', 'USER', 'The user lives in Bristol.', True),
    ('The user is vegetarian and requested dinner suggestions.', 'USER', 'The user is vegetarian.', True),
    ('The user told the assistant to reply briefly.', 'DIRECTIVES', 'Always reply briefly.', True),
    ('The user asked to explain the phrase "I am allergic to eggs".', 'USER', 'The user is allergic to eggs.', False),
    ('The user said "I am allergic to eggs" and requested dinner suggestions.', 'USER', 'The user is allergic to eggs.', True),
    ('La usuaria pidió corregir la frase "Tengo una gata llamada Miso".', 'USER', 'The user has a cat named Miso.', False),
    ('Kullanıcı "Öğretmen olarak çalışıyorum" cümlesini çevirmemi istedi.', 'USER', 'The user works as a teacher.', False),
    ('Der Benutzer bat um ein Rollenspiel als Lehrer.', 'USER', 'The user works as a teacher.', False),
    ('The user is visiting Bristol for the weekend.', 'USER', 'The user lives in Bristol.', False),
    ('The user asked about the difference between a dynamic camera system and a fixed camera setup in games.',
     'USER', 'The user asked about the difference between a dynamic camera system and a fixed camera setup in games', False),
    ('The user inquired about the difference between dynamic and fixed camera setups in games.',
     'USER', 'The user inquired about the difference between dynamic and fixed camera setups in games', False),
    ('The weather forecast for London for the week will range from 6.8 to 16.8 degrees Celsius.',
     'WORLD', 'The weather forecast for London for the week will range from 6.8 to 16.8 degrees Celsius', False),
    ('The user requested the weather forecast for London.', 'USER', 'The user is in London', False),
    ('The user requested the weather forecast for London.', 'USER', 'The user is located in London', False),
    ('The user follows an 1800 kcal daily meal plan.', 'USER', 'The user follows an 1800 kcal daily meal plan', True),
    ('A lookup established that Trenches Boxing Club in Hackney offers evening classes.',
     'WORLD', 'Trenches Boxing Club in Hackney offers evening classes', True),
    ('The user said they live in London and requested tomorrow\'s weather.', 'USER', 'The user lives in London', True),
])
def test_review_checks_candidate_relationship_against_source(monkeypatch, summary, branch, fact, supported):
    from jarvis.memory import graph_ops
    original = graph_ops.call_llm_direct
    def infer(**kwargs):
        if kwargs['user_content'].startswith('Extract'):
            return json.dumps([{'branch': branch, 'fact': fact}])
        return original(**kwargs)
    monkeypatch.setattr(graph_ops, 'call_llm_direct', infer)
    cfg = voice_config()
    result = graph_ops.extract_graph_memories(summary, cfg, cfg.llm_chat_model, timeout_sec=30)
    assert result == ([(branch.lower(), fact)] if supported else [])
