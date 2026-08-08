"""The Memory Viewer must not phone home.

It renders the user's diary, personal facts and meal log — the most
sensitive data Jarvis holds. Any external resource on that page hands a
third party the user's IP, User-Agent and the time they opened it, and
silently degrades when the machine is offline, which is a supported way
to run Jarvis.

The README states the check this guards: `lsof -i` against a running
Jarvis "should only show 127.0.0.1 to Ollama".
"""

from __future__ import annotations

import re

import pytest

try:
    import flask  # noqa: F401

    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False

pytestmark = pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")


@pytest.fixture
def dashboard_html():
    from src.desktop_app import memory_viewer
    return memory_viewer.index()


@pytest.mark.unit
def test_dashboard_requests_nothing_from_the_network(dashboard_html):
    """No absolute http(s) URL may be fetched when the page renders."""
    fetched = re.findall(
        r'(?:href|src)\s*=\s*["\'](https?://[^"\']+)["\']',
        dashboard_html,
        re.IGNORECASE,
    )
    assert fetched == [], f"dashboard fetches external resources: {fetched}"


@pytest.mark.unit
def test_dashboard_does_not_preconnect_to_third_parties(dashboard_html):
    """`preconnect`/`dns-prefetch` leak the visit even without a fetch."""
    hints = re.findall(
        r'<link[^>]+rel\s*=\s*["\'](?:preconnect|dns-prefetch)["\'][^>]*>',
        dashboard_html,
        re.IGNORECASE,
    )
    assert hints == [], f"dashboard preconnects to third parties: {hints}"


@pytest.mark.unit
def test_no_css_import_of_remote_stylesheets(dashboard_html):
    """`@import url(https://…)` is a second way to pull a remote stylesheet."""
    imports = re.findall(r'@import\s+url\(["\']?https?://[^)]+\)', dashboard_html, re.IGNORECASE)
    assert imports == [], f"dashboard imports remote CSS: {imports}"


@pytest.mark.unit
def test_google_fonts_is_not_referenced_anywhere(dashboard_html):
    """Named explicitly: this was the regression, and it is easy to reintroduce."""
    assert "fonts.googleapis.com" not in dashboard_html
    assert "fonts.gstatic.com" not in dashboard_html


@pytest.mark.unit
def test_text_still_has_a_font_stack_to_fall_back_on(dashboard_html):
    """Dropping the webfont must not leave the page on browser defaults.

    The design uses a display face for UI text and a monospace face for
    data. With no CDN, each must resolve through a stack of faces that
    actually exist on Windows, macOS and Linux, ending in a generic
    family so there is always a last resort.
    """
    declared = dict(
        re.findall(r"(--font-[\w-]+):\s*([^;]+);", dashboard_html)
    )
    assert declared, "no font custom properties defined"

    for name, stack in declared.items():
        families = [f.strip() for f in stack.split(",") if f.strip()]
        assert len(families) >= 3, f"{name} has too thin a stack: {stack}"
        assert families[-1] in {"sans-serif", "serif", "monospace", "system-ui"}, (
            f"{name} must end in a generic family, got {families[-1]}"
        )

    # Every concrete declaration resolves through those properties or
    # names a stack of its own — never a single font with no fallback.
    for stack in re.findall(r"font-family:\s*([^;]+);", dashboard_html):
        stack = stack.strip()
        if "inherit" in stack:
            continue
        if stack.startswith("var("):
            referenced = re.match(r"var\((--[\w-]+)\)", stack)
            assert referenced and referenced.group(1) in declared, (
                f"font-family references an undefined property: {stack}"
            )
        else:
            assert "," in stack, f"font stack has no fallback: {stack}"
