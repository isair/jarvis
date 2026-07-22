"""Phase 3A: safe Markdown renderer tests (security-focused, pure str->str)."""

from __future__ import annotations

import pytest

from jarvis.utils.markdown_render import render_markdown, is_safe_url


# --- escaping / injection ---------------------------------------------------

def test_script_tag_is_escaped_not_live():
    out = render_markdown("<script>alert(1)</script>")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_img_tag_is_escaped():
    out = render_markdown('<img src=x onerror="alert(1)">')
    assert "<img" not in out
    assert "&lt;img" in out


def test_event_handler_html_neutralised():
    out = render_markdown('<div onclick="evil()">x</div>')
    assert "onclick" in out  # visible as text
    assert "<div" not in out  # never live markup


def test_iframe_escaped():
    out = render_markdown("<iframe src=//evil></iframe>")
    assert "<iframe" not in out


# --- links: scheme allowlist ------------------------------------------------

@pytest.mark.parametrize("scheme", [
    "javascript:alert(1)", "data:text/html,<b>x</b>", "file:///etc/passwd",
    "vbscript:msgbox(1)", "//evil.com/x",
])
def test_unsafe_link_schemes_never_become_anchors(scheme):
    out = render_markdown(f"[click]({scheme})")
    assert "<a " not in out          # no live anchor of any kind
    assert 'href="javascript' not in out
    assert 'href="data' not in out
    assert 'href="file' not in out


def test_http_and_https_links_are_anchors():
    assert '<a href="http://a.io">x</a>' in render_markdown("[x](http://a.io)")
    assert '<a href="https://a.io">y</a>' in render_markdown("[y](https://a.io)")


def test_is_safe_url():
    assert is_safe_url("https://ok.com")
    assert is_safe_url("http://ok.com")
    assert not is_safe_url("javascript:alert(1)")
    assert not is_safe_url("data:text/html,x")
    assert not is_safe_url("file:///x")
    assert not is_safe_url("//protocol-relative")
    assert not is_safe_url("")
    assert not is_safe_url("https://ok.com\n<script>")  # control char rejected


# --- formatting -------------------------------------------------------------

def test_bold_and_italic():
    out = render_markdown("**b** and *i*")
    assert "<b>b</b>" in out and "<i>i</i>" in out


def test_inline_code_escapes_content():
    out = render_markdown("run `<script>` now")
    assert "<code>&lt;script&gt;</code>" in out


def test_fenced_code_block_escaped():
    out = render_markdown("before\n```\n<b>x</b> & y\n```\nafter")
    assert "<pre><code>" in out
    assert "&lt;b&gt;x&lt;/b&gt; &amp; y" in out
    assert "<b>x</b>" not in out  # not live inside code


def test_headings():
    out = render_markdown("# Title")
    assert "<h3>Title</h3>" in out


def test_unordered_list():
    out = render_markdown("- one\n- two")
    assert out.count("<li>") == 2 and "<ul>" in out


def test_ordered_list():
    out = render_markdown("1. a\n2. b")
    assert out.count("<li>") == 2 and "<ol>" in out


def test_blockquote():
    assert "<blockquote>" in render_markdown("> quoted")


def test_empty_input():
    assert render_markdown("") == ""
    assert render_markdown("   ") == ""
    assert render_markdown(None) == ""


def test_diacritics_preserved():
    out = render_markdown("Salut, cum merge cu ăîâșț?")
    assert "ăîâșț" in out


def test_link_inside_emphasis_still_safe():
    out = render_markdown("**[x](javascript:alert(1))**")
    assert "<a " not in out


# --- Phase 3A hardening (review remediation) --------------------------------

def test_userinfo_authority_rejected():
    # http://google.com@evil.com : visible host is evil.com -> phishing shape
    assert not is_safe_url("http://google.com@evil.com")
    out = render_markdown("[Google](http://google.com@evil.com)")
    assert "<a " not in out  # downgraded to plain text


def test_plain_https_still_allowed_after_userinfo_guard():
    assert is_safe_url("https://docs.python.org/3/")
    assert '<a href="https://docs.python.org/3/">' in render_markdown("[d](https://docs.python.org/3/)")


def test_length_cap_bounds_input():
    from jarvis.utils.markdown_render import MAX_RENDER_CHARS
    big = "[" * (MAX_RENDER_CHARS + 5000)  # pathological, would be O(N^2)
    out = render_markdown(big)  # must return quickly, not hang
    assert isinstance(out, str)


def test_href_not_corrupted_by_emphasis():
    # underscores in a URL must not be turned into <i>/<b> inside the href
    out = render_markdown("[x](https://a.io/a__b__c)")
    assert '<a href="https://a.io/a__b__c">' in out
    assert "<b>" not in out and "<i>" not in out


def test_forged_code_placeholder_is_inert():
    # untrusted text containing a placeholder-looking line cannot forge a code block
    out = render_markdown("@@deadbeefF0@@")
    assert "<pre>" not in out  # nonce mismatch -> treated as plain text
    assert "deadbeef" in out or "@@" in out  # rendered as escaped text, no crash


def test_inline_code_not_emphasised():
    out = render_markdown("use `a_b_c` please")
    assert "<code>a_b_c</code>" in out  # underscores inside code stay literal
    assert "<i>" not in out
