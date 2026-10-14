"""Bounded text collection from incremental HTML parser events."""
from urllib.parse import urljoin, urlsplit


class PageText:
    """Retain a readable excerpt, title and optional links without a DOM."""

    _HIDDEN = frozenset(('script', 'style', 'noscript', 'template'))
    _BLOCKS = frozenset(('p', 'div', 'br', 'hr', 'li', 'tr', 'section', 'article',
                         'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'pre', 'blockquote'))

    def __init__(self, url: str, include_links: bool, max_chars: int = 48000):
        self.url = url
        self.include_links = include_links
        self.max_chars = max_chars
        self.parts = []
        self.chars = 0
        self.title = ''
        self.links = []
        self.truncated = False
        self.invalid_links = 0
        self._hidden_depth = 0
        self._in_title = False
        self._link = None
        self._link_text = ''

    def _append(self, text):
        if not text:
            return
        if text.isspace():
            if not self.parts or self.parts[-1].endswith((' ', '\n')):
                return
            text = '\n' if '\n' in text else ' '
        remaining = self.max_chars-self.chars
        if len(text) > remaining:
            self.truncated = True
        retained = text[:remaining]
        if retained:
            self.parts.append(retained)
            self.chars += len(retained)

    def start(self, tag, attrs):
        if self._hidden_depth:
            self._hidden_depth += 1
            return
        if tag in self._HIDDEN:
            self._hidden_depth = 1
            return
        if tag in self._BLOCKS:
            self._append('\n')
        if tag == 'title':
            self._in_title = True
        if tag == 'a' and self.include_links and len(self.links) < 20:
            href = attrs.get('href', '').strip()
            try:
                resolved = urljoin(self.url, href)
                if href and len(resolved) <= 2048 and urlsplit(resolved).scheme in ('http', 'https', 'mailto', 'tel'):
                    self._link, self._link_text = resolved, ''
            except ValueError:
                self.invalid_links += 1

    def end(self, tag):
        if self._hidden_depth:
            self._hidden_depth -= 1
            return
        if tag == 'title':
            self._in_title = False
        if tag == 'a' and self._link is not None:
            label = self._link_text.strip()
            if len(label) > 3:
                self.links.append((label, self._link))
            self._link, self._link_text = None, ''
        if tag in self._BLOCKS:
            self._append('\n')

    def data(self, text):
        if self._hidden_depth:
            return
        if self._in_title:
            self.title += text[:max(0, 512-len(self.title))]
            return
        self._append(text)
        if self._link is not None:
            self._link_text += text[:max(0, 512-len(self._link_text))]

    def close(self):
        return self

    def text(self):
        seen, lines = set(), []
        for line in ''.join(self.parts).splitlines():
            line = line.strip()
            if len(line) > 3 and line not in seen:
                if len(lines) == 500:
                    self.truncated = True
                    break
                lines.append(line)
                seen.add(line)
        return '\n'.join(lines)
