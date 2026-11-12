"""Opening the local viewer does not load third-party styling resources."""

import re
from html.parser import HTMLParser
from urllib.parse import urlsplit

import pytest


class ViewerAssets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.resources = []
        self.css = []
        self.in_style = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "link" and set(attrs.get("rel", "").split()) & {"stylesheet", "preconnect", "dns-prefetch", "preload"}:
            self.resources.append(attrs.get("href", ""))
        elif tag in {"script", "img", "iframe", "source"} and attrs.get("src"):
            self.resources.append(attrs["src"])
        if tag == "style":
            self.in_style = True

    def handle_endtag(self, tag):
        if tag == "style":
            self.in_style = False

    def handle_data(self, data):
        if self.in_style:
            self.css.append(data)


def external_resources(html):
    assets = ViewerAssets()
    assets.feed(html)
    css = "".join(assets.css)
    assets.resources.extend(re.findall(r"url\(\s*['\"]?([^'\")\s]+)", css))
    assets.resources.extend(re.findall(r"@import\s+['\"]([^'\"]+)", css))
    return [resource for resource in assets.resources
            if urlsplit(resource).netloc or urlsplit(resource).scheme not in {"", "data"}]


@pytest.mark.unit
def test_viewer_style_and_static_resources_are_local():
    from desktop_app.memory_viewer import index

    assert external_resources(index()) == []


@pytest.mark.unit
@pytest.mark.parametrize("html,expected", [
    ('<link rel="stylesheet" href="/assets/theme.css">', []),
    ('<img src="data:image/png;base64,AAAA">', []),
    ('<a href="https://example.invalid">Explicit navigation</a>', []),
    ('<link rel="stylesheet" href="//example.invalid/theme.css">', ['//example.invalid/theme.css']),
    ('<style>@import "https://example.invalid/theme.css";</style>', ['https://example.invalid/theme.css']),
    ('<style>@font-face {src:url("https://example.invalid/font.woff2")}</style>', ['https://example.invalid/font.woff2']),
])
def test_asset_inventory_distinguishes_resources_from_explicit_links(html, expected):
    assert external_resources(html) == expected
