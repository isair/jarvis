"""Search result structure supports short Unicode titles without navigation noise."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jarvis.tools.builtin import web_search

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('result_class', ['result-link', 'result__a'])
@pytest.mark.parametrize('title,query', [
    ('展覧会', '東京展示会'),
    ('ИИ', 'искусственный интеллект'),
    ('فن', 'معرض الفن'),
    ('Privacy law guide', 'privacy law guide'),
])
def test_short_and_localised_result_titles_are_not_navigation(monkeypatch, title, query, result_class):
    html = ('<a href="https://navigation.test/about">Informations sur notre service</a>'
            f'<a class="{result_class}" href="https://source.test/article">{title}</a>').encode()
    result = search_html(monkeypatch, html, query)
    assert result.success, result.reply_text
    assert title in result.reply_text
    assert 'https://source.test/article' in result.reply_text
    assert 'navigation.test' not in result.reply_text
    assert '<<<BEGIN UNTRUSTED WEB EXTRACT>>>' in result.reply_text


def search_html(monkeypatch, html, query):
    instant = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {})
    response = SimpleNamespace(status_code=200, content=html)
    monkeypatch.setattr(web_search.requests, 'get', lambda url, **kw:
                        instant if url.startswith('https://api.duckduckgo.com/') else response)
    monkeypatch.setattr(web_search, '_cascade_fetch', lambda *a, **kw: query+' synthetic source reading.')
    cfg = SimpleNamespace(web_search_enabled=True, voice_debug=False,
                          brave_search_api_key='', wikipedia_fallback_enabled=False)
    return web_search.WebSearchTool().run({'search_query': query},
        SimpleNamespace(cfg=cfg, language=None, user_print=Mock()))


@pytest.mark.parametrize('prefix', ['//duckduckgo.com/l/', 'https://duckduckgo.com/l/'])
def test_redirect_destinations_preserve_encoded_path_and_query(monkeypatch, prefix):
    from urllib.parse import quote

    destination = 'https://source.test/a%2Fb?q=one%26two&next=%252F'
    html = (f'<a class="result-link" href="{prefix}?uddg={quote(destination, safe="")}">'
            'Source</a>').encode()
    result = search_html(monkeypatch, html, 'Source')
    assert result.success
    assert f'Link: {destination}' in result.reply_text


def test_only_five_nonempty_http_results_are_returned_in_rank_order(monkeypatch):
    html = ('<a class="result-link" href="javascript:alert(1)">Invalid</a>'
            '<a class="result-link" href="https://empty.test"> </a>'
            '<a class="result-link" href="https://[invalid">Malformed</a>'
            + ''.join(f'<a class="result-link" href="https://source.test/{i}">'
                      f'Source {i}</a>' for i in range(7))).encode()
    result = search_html(monkeypatch, html, 'Source')
    assert result.success
    for i in range(5):
        assert f'{i+1}. **Source {i}**' in result.reply_text
    assert 'Source 5' not in result.reply_text
    assert 'Source 6' not in result.reply_text
    assert 'Invalid' not in result.reply_text
    assert 'Malformed' not in result.reply_text
    assert 'empty.test' not in result.reply_text
