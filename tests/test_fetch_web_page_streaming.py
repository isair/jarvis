"""Large pages expose useful bounded text without retaining their markup."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock
import tracemalloc

import pytest

from jarvis.tools.builtin import fetch_web_page

pytestmark = pytest.mark.unit


def read_page(monkeypatch, body, *, headers=None, include_links=False, chunk_bytes=None):
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        iter_content=lambda chunk_size: (body[i:i+(chunk_bytes or chunk_size)] for i in range(0, len(body), chunk_bytes or chunk_size)),
        encoding='utf-8', headers=headers or {'Content-Type': 'text/html; charset=utf-8'},
        url='https://example.test/final/page',
    )
    monkeypatch.setattr(fetch_web_page.requests, 'get', lambda *a, **kw: nullcontext(response))
    return fetch_web_page.FetchWebPageTool().run(
        {'url': 'https://example.test/start', 'include_links': include_links},
        SimpleNamespace(user_print=Mock()),
    )


@pytest.mark.parametrize('element', ['script', 'style'])
def test_large_non_content_sections_do_not_hide_the_article(monkeypatch, element):
    body = (b'<html><head><title>Large article</title><'+element.encode()+b'>'
            + b'x'*(3*1024*1024)
            + b'</'+element.encode()+b'></head><body><p>Useful article after the large payload.</p></body></html>')
    result = read_page(monkeypatch, body)
    assert result.success, result.reply_text
    assert 'Useful article after the large payload.' in result.reply_text
    assert 'Large article' in result.reply_text
    assert 'x'*100 not in result.reply_text


def test_large_visible_content_returns_an_explicit_bounded_excerpt(monkeypatch):
    body = b'<html><body><p>Opening article content.</p>' + b''.join(f'<p>Useful paragraph number {i}.</p>'.encode() for i in range(120000)) + b'</body></html>'
    result = read_page(monkeypatch, body)
    assert result.success, result.reply_text
    assert 'Opening article content.' in result.reply_text
    assert 'truncat' in result.reply_text.casefold() or 'excerpt' in result.reply_text.casefold()
    assert len(result.reply_text) < 52000


def test_large_markup_does_not_break_split_unicode_or_relative_links(monkeypatch):
    body = (b'<html><head><script>'+b'x'*(3*1024*1024)+b'</script></head><body>'
            + '<p>İstanbul, İzmir et Café déjà vu.</p><a href="child">Detailed source</a>'.encode()
            + b'</body></html>')
    result = read_page(monkeypatch, body, include_links=True)
    assert result.success, result.reply_text
    assert 'İstanbul, İzmir et Café déjà vu.' in result.reply_text
    assert 'https://example.test/final/child' in result.reply_text


def test_skipped_markup_does_not_allocate_a_page_sized_python_buffer(monkeypatch):
    body = b'<html><script>'+b'x'*(4*1024*1024)+b'</script><p>Useful content survives.</p></html>'
    tracemalloc.start()
    try:
        result = read_page(monkeypatch, body)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.success, result.reply_text
    assert 'Useful content survives.' in result.reply_text
    assert peak < 2*1024*1024, f'Page reading retained {peak:,} Python bytes'


def test_html_declared_charset_applies_without_http_charset(monkeypatch):
    body = '<html><head><meta charset="iso-8859-1"></head><body><p>Café déjà vu.</p></body></html>'.encode('iso-8859-1')
    result = read_page(monkeypatch, body, headers={'Content-Type': 'text/html'})
    assert result.success, result.reply_text
    assert 'Café déjà vu.' in result.reply_text


def test_encoding_metadata_survives_one_byte_transport_chunks(monkeypatch):
    text = 'İstanbul, İzmir et Café déjà vu.'
    body = f'<html><body><p>{text}</p></body></html>'.encode('utf-16')
    result = read_page(monkeypatch, body, headers={'Content-Type': 'text/html'}, chunk_bytes=1)
    assert result.success, result.reply_text
    assert text in result.reply_text


def test_download_deadline_labels_only_content_received_before_it(monkeypatch):
    ticks = iter((0, 0, 20))
    monkeypatch.setattr(fetch_web_page, 'time', SimpleNamespace(monotonic=lambda: next(ticks)))
    body = b'<p>Received article introduction.</p>'+b' '*65536+b'<p>Unread footer secret.</p>'
    result = read_page(monkeypatch, body)
    assert result.success, result.reply_text
    assert 'Received article introduction.' in result.reply_text
    assert 'Unread footer secret.' not in result.reply_text
    assert 'Partial page excerpt' in result.reply_text
