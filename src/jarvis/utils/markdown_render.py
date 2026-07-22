"""Safe Markdown -> HTML renderer for the modern chat UI (Phase 3A).

Assistant replies are UNTRUSTED text. This renderer produces a small, safe HTML
subset suitable for a read-only ``QTextBrowser``/``QLabel``. Security invariants:

  * ALL input is HTML-escaped FIRST, so the only tags in the output are ones this
    module inserts -- raw ``<script>``/``<img>``/``<iframe>``/event-handler HTML
    in the source can never survive as live markup;
  * NO JavaScript, NO remote resource loading (no ``<img>``, no CSS ``url()``),
    NO inline styles from the source;
  * links are rendered as anchors ONLY for ``http``/``https`` targets WITHOUT a
    ``user@host`` authority; every other scheme (``javascript:``/``data:``/
    ``file:``/``vbscript:``/protocol-relative ``//``) is downgraded to plain
    visible text. The caller must still keep ``setOpenExternalLinks(False)`` so
    even an http/https anchor never auto-opens without an explicit handler.

Hardening (Phase 3A review): input is length-capped to bound the O(N^2) regex
cost on the GUI thread; the internal code-block placeholder uses a per-render
nonce so untrusted text cannot forge/duplicate a code block; links and inline
code are lifted to sentinels before emphasis runs, so bold/italic can never
corrupt an href or a code span.

Supported Markdown (deliberately minimal): fenced code blocks, inline code,
bold, italic, headings, unordered/ordered lists, blockquotes, links, paragraphs
and line breaks. Unknown constructs degrade to escaped text. Pure stdlib; no Qt,
no third-party Markdown library -- fully unit-testable as ``str -> str``.
"""

from __future__ import annotations

import html
import re
import secrets
from typing import List, Tuple

__all__ = ["render_markdown", "is_safe_url", "SAFE_URL_SCHEMES", "MAX_RENDER_CHARS"]

# Only these schemes become clickable anchors. Everything else is shown as text.
SAFE_URL_SCHEMES = ("http://", "https://")

# Cap untrusted input before the (polynomial, GUI-thread) regex passes run.
MAX_RENDER_CHARS = 20000

_FENCE_RE = re.compile(r"^\s*```")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_ITALIC_RE = re.compile(r"(?<![\*_\w])([*_])(?=\S)(.+?)(?<=\S)\1(?![\*_\w])")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_ULIST_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
_OLIST_RE = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_QUOTE_RE = re.compile(r"^\s*>\s?(.*)$")


def is_safe_url(url: str) -> bool:
    """True only for http/https without a ``user@host`` authority. Rejects
    javascript:/data:/file:/vbscript:, protocol-relative //host, credentials in
    the authority, and any control char/whitespace."""
    if not url:
        return False
    u = url.strip()
    if not u or any(ord(c) < 0x20 for c in u):
        return False
    low = u.lower()
    if low.startswith("//"):  # protocol-relative
        return False
    if not low.startswith(SAFE_URL_SCHEMES):
        return False
    # Reject a userinfo (`user:pass@host`) authority: the visible host would be
    # after the '@', a classic phishing shape. Check only the authority segment.
    scheme_len = len("https://") if low.startswith("https://") else len("http://")
    authority = u[scheme_len:].split("/", 1)[0]
    if "@" in authority:
        return False
    return True


def _sentinel(nonce: str, kind: str, idx: int) -> str:
    # No spaces -> survives line.strip(); nonce -> unforgeable by untrusted text.
    return f"@@{nonce}{kind}{idx}@@"


def _extract_fenced_code(lines: List[str], nonce: str) -> Tuple[List[str], List[str]]:
    """Replace fenced ``` blocks with nonce placeholders; return (lines, html).

    Code content is HTML-escaped and never passed through inline formatting.
    An unterminated fence runs to end-of-text (best effort, bounded loop).
    """
    out: List[str] = []
    code_blocks: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if _FENCE_RE.match(lines[i]):
            i += 1
            buf: List[str] = []
            while i < n and not _FENCE_RE.match(lines[i]):
                buf.append(lines[i])
                i += 1
            if i < n:  # consume the closing fence
                i += 1
            escaped = html.escape("\n".join(buf))
            code_blocks.append(f"<pre><code>{escaped}</code></pre>")
            out.append(_sentinel(nonce, "F", len(code_blocks) - 1))
        else:
            out.append(lines[i])
            i += 1
    return out, code_blocks


def _render_inline(escaped: str, nonce: str) -> str:
    """Inline formatting on an ALREADY HTML-escaped string.

    Inline code and links are lifted to nonce sentinels BEFORE emphasis so that
    bold/italic passes can never run over a code span or an href attribute.
    """
    parts: List[str] = []  # restored last, in order

    def _stash(fragment: str) -> str:
        parts.append(fragment)
        return _sentinel(nonce, "I", len(parts) - 1)

    # 1) inline code -> sentinel (content already escaped; stays literal)
    escaped = _INLINE_CODE_RE.sub(lambda m: _stash(f"<code>{m.group(1)}</code>"), escaped)

    # 2) links -> sentinel (scheme-validated); label kept escaped
    def _link(m: "re.Match[str]") -> str:
        label = m.group(1)
        raw_url = html.unescape(m.group(2))  # was escaped with the rest of the line
        if is_safe_url(raw_url):
            safe_attr = html.escape(raw_url, quote=True)
            return _stash(f'<a href="{safe_attr}">{label}</a>')
        return f"{label} ({m.group(2)})"  # unsafe scheme -> plain (escaped) text

    escaped = _LINK_RE.sub(_link, escaped)

    # 3) emphasis on the remaining text only
    escaped = _BOLD_RE.sub(lambda m: f"<b>{m.group(2)}</b>", escaped)
    escaped = _ITALIC_RE.sub(lambda m: f"<i>{m.group(2)}</i>", escaped)

    # 4) restore sentinels (indices are single ints; safe substring replace)
    for idx, frag in enumerate(parts):
        escaped = escaped.replace(_sentinel(nonce, "I", idx), frag)
    return escaped


def _inline_full(line: str, nonce: str) -> str:
    return _render_inline(html.escape(line), nonce)


def render_markdown(text: str) -> str:
    """Render untrusted Markdown ``text`` to a safe HTML subset string."""
    if text is None:
        return ""
    text = str(text)
    if not text.strip():
        return ""
    if len(text) > MAX_RENDER_CHARS:  # bound GUI-thread regex cost
        text = text[:MAX_RENDER_CHARS]

    nonce = secrets.token_hex(6)  # per-render; untrusted text cannot predict it
    fence_re = re.compile(r"^@@" + re.escape(nonce) + r"F(\d+)@@$")

    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines, code_blocks = _extract_fenced_code(raw_lines, nonce)

    html_parts: List[str] = []
    list_stack: List[str] = []
    para_buf: List[str] = []

    def _close_lists() -> None:
        while list_stack:
            html_parts.append(f"</{list_stack.pop()}>")

    def _flush_para() -> None:
        if para_buf:
            html_parts.append("<p>" + "<br>".join(para_buf) + "</p>")
            para_buf.clear()

    for line in lines:
        stripped = line.strip()
        m_ph = fence_re.match(stripped)
        if m_ph:
            _flush_para()
            _close_lists()
            try:
                html_parts.append(code_blocks[int(m_ph.group(1))])
            except Exception:
                pass
            continue

        if not stripped:
            _flush_para()
            _close_lists()
            continue

        m = _HEADING_RE.match(line)
        if m:
            _flush_para()
            _close_lists()
            level = min(len(m.group(1)) + 2, 6)  # h1 -> h3, keep modest in a bubble
            html_parts.append(f"<h{level}>{_inline_full(m.group(2), nonce)}</h{level}>")
            continue

        m = _QUOTE_RE.match(line)
        if m:
            _flush_para()
            _close_lists()
            html_parts.append(f"<blockquote>{_inline_full(m.group(1), nonce)}</blockquote>")
            continue

        m = _ULIST_RE.match(line)
        if m:
            _flush_para()
            if not list_stack or list_stack[-1] != "ul":
                _close_lists()
                list_stack.append("ul")
                html_parts.append("<ul>")
            html_parts.append(f"<li>{_inline_full(m.group(1), nonce)}</li>")
            continue

        m = _OLIST_RE.match(line)
        if m:
            _flush_para()
            if not list_stack or list_stack[-1] != "ol":
                _close_lists()
                list_stack.append("ol")
                html_parts.append("<ol>")
            html_parts.append(f"<li>{_inline_full(m.group(1), nonce)}</li>")
            continue

        _close_lists()
        para_buf.append(_inline_full(line, nonce))

    _flush_para()
    _close_lists()
    return "".join(html_parts)
