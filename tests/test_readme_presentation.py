"""Readable GitHub documentation, without thumbnail galleries or broken assets."""

from html.parser import HTMLParser
from pathlib import Path
import re
import struct

import pytest


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.unit


class ScreenshotParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []
        self.in_paragraph = False
        self.paragraph_images = 0

    def handle_starttag(self, tag, attrs):
        if tag == 'p':
            self.in_paragraph = True
            self.paragraph_images = 0
        if tag == 'img':
            self.images.append(dict(attrs))
            if self.in_paragraph:
                self.paragraph_images += 1
                assert self.paragraph_images == 1, 'Screenshots need their own row'

    def handle_endtag(self, tag):
        if tag == 'p':
            self.in_paragraph = False


def test_readme_screenshots_are_readable_standalone_assets():
    parser = ScreenshotParser()
    parser.feed((ROOT / 'README.md').read_text())
    assert parser.images
    for image in parser.images:
        assert image.get('alt'), 'Screenshots need descriptive alternative text'
        path = ROOT / image['src']
        assert path.is_file()
        data = path.read_bytes()
        assert data[:8] == b'\x89PNG\r\n\x1a\n'
        width, height = struct.unpack('>II', data[16:24])
        displayed = int(image['width'])
        assert displayed >= (720 if width > height else 420)
        assert width >= displayed, 'Do not upscale screenshots'


def test_readme_relative_file_links_exist():
    for document in (ROOT / 'README.md', ROOT / 'docs/CONFIGURATION.md'):
        for target in re.findall(r'\]\(([^)]+)\)', document.read_text()):
            if '://' in target or target.startswith(('mailto:', '#')):
                continue
            assert (document.parent / target.split('#')[0]).exists(), target


def test_readme_leads_with_voice_assistant_face():
    parser = ScreenshotParser()
    parser.feed((ROOT / 'README.md').read_text())
    assert parser.images[0]['src'] == 'docs/img/face.png'
