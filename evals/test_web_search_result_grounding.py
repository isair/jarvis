"""Local chat grounds answers in real search results rather than navigation."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import voice_config
from jarvis.llm import get_llm_backend
from jarvis.system_prompt import build_system_prompt
from jarvis.tools.builtin import web_search

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize('title', ['展覧会', 'فن', 'Privacy festival'])
def test_short_search_titles_ground_answers_in_article_not_navigation(title):
    html = ('<meta charset="utf-8">'
            '<a href="https://navigation.test/about">About the festival website</a>'
            f'<a class="result-link" href="https://source.test/article">{title}</a>').encode()
    instant = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {})
    response = SimpleNamespace(status_code=200, content=html)
    cfg = SimpleNamespace(web_search_enabled=True, voice_debug=False,
                          brave_search_api_key='', wikipedia_fallback_enabled=False)

    def fetch(urls, **kwargs):
        return ('The Kestrel festival takes place in Bristol.'
                if urls[0][1] == 'https://source.test/article'
                else 'The website editor works in London.')

    with patch.object(web_search.requests, 'get', side_effect=[instant, response]), patch.object(
            web_search, '_cascade_fetch', side_effect=fetch):
        result = web_search.WebSearchTool().run(
            {'search_query': 'Which city hosts the Kestrel festival?'},
            SimpleNamespace(cfg=cfg, language=None, user_print=Mock()),
        )
    assert result.success, result.reply_text
    assert 'https://source.test/article' in result.reply_text
    assert 'navigation.test' not in result.reply_text
    cfg = voice_config()
    answer = get_llm_backend(cfg).direct(
        cfg.llm_chat_model, build_system_prompt(),
        'Using only the search result below, which city hosts the Kestrel festival? '
        'Return only a JSON object with a city field.\n\n'+result.reply_text,
        timeout_sec=60, temperature=0.0, max_tokens=256,
    )
    assert answer, 'No reply'
    start = answer.find('{')
    assert start >= 0, answer
    parsed, _ = json.JSONDecoder().raw_decode(answer[start:])
    assert parsed == {'city': 'Bristol'}, answer
