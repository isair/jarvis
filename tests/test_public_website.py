"""Public pages stay crawlable, accurate and safe to publish."""
import importlib.util
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]
ROOT = Path(__file__).resolve().parents[1]
ROUTES = tuple(page['route'] for page in json.loads((ROOT / 'website/site.json').read_text())['pages'])


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.elements, self.visible, self.structured = [], [], []
        self.body = False
        self.hidden = None
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        self.elements.append((tag, values))
        if tag == 'body':
            self.body = True
        if tag in ('script', 'style'):
            self.hidden = tag
        if tag == 'script' and values.get('type') == 'application/ld+json':
            self.hidden = 'json'

    def handle_endtag(self, tag):
        if tag == 'body':
            self.body = False
        if tag in ('script', 'style'):
            self.hidden = None

    def handle_data(self, data):
        if self.hidden == 'json':
            self.structured.append(data)
        if self.body and self.hidden is None:
            self.visible.append(data)

    def tags(self, tag):
        return [attrs for name, attrs in self.elements if name == tag]


@pytest.fixture
def builder():
    spec = importlib.util.spec_from_file_location('public_site_builder', ROOT / 'website/build.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('base', [
    'https://barissencan.com/jarvis/',
    'https://isair.github.io/jarvis/',
    'https://jarvis.example.org/',
])
def test_pages_have_readable_content_and_consistent_discovery_metadata(builder, tmp_path, base):
    output = tmp_path / 'site'
    builder.build_site(output, base_url=base)
    urls, titles = [], []
    for route in ROUTES:
        source = (output / route / 'index.html').read_text()
        page = Page(source)
        canonical = [link['href'] for link in page.tags('link') if link.get('rel') == 'canonical']
        assert canonical == [base + route]
        urls.extend(canonical)
        assert len(page.tags('h1')) == 1
        assert int(page.tags('main')[0].get('tabindex', '0')) < 0, 'Skip navigation must focus main without another tab stop'
        assert page.tags('html')[0]['lang'] == 'en-GB'
        assert any(meta.get('name') == 'description' and meta.get('content') for meta in page.tags('meta'))
        assert any(meta.get('property') == 'og:url' and meta.get('content') == canonical[0] for meta in page.tags('meta'))
        assert len(' '.join(page.visible).split()) > 80, 'Useful content must be present without JavaScript'
        assert not [script for script in page.tags('script') if script.get('type') != 'application/ld+json']
        assert not [link for link in page.tags('link') if link.get('rel') == 'stylesheet' and urlsplit(link['href']).netloc]
        titles.append(source.split('<title>')[1].split('</title>')[0])
    assert len(set(titles)) == len(ROUTES)
    assert [node.text for node in ET.parse(output / 'sitemap.xml').findall('.//{*}loc')] == urls
    home = Page((output / 'index.html').read_text())
    app = json.loads(''.join(home.structured))
    assert app['@type'] == 'SoftwareApplication'
    assert app['url'] == base
    assert app['description'] in ' '.join(' '.join(home.visible).split())
    config = json.loads((ROOT / 'website/site.json').read_text())
    assert app['license'] == config['repository'] + '/blob/' + config['source_branch'] + '/LICENSE'
    assert app['sameAs'] == config['repository']
    assert 'codeRepository' not in app
    assert app['author']['name'] == config['author']['name']
    assert 'aggregateRating' not in app and 'review' not in app
    assert 'non-commercial' in (output / 'privacy/index.html').read_text()


def test_navigation_images_and_social_previews_resolve_inside_the_project_path(builder, tmp_path):
    base = 'https://barissencan.com/jarvis/'
    output = tmp_path / 'site'
    builder.build_site(output, base_url=base)
    prefix = urlsplit(base).path
    for route in ROUTES:
        page = Page((output / route / 'index.html').read_text())
        targets = [a['href'] for a in page.tags('a') if 'href' in a]
        targets += [image['src'] for image in page.tags('img')]
        targets += [link['href'] for link in page.tags('link') if link.get('rel') in ('stylesheet', 'icon')]
        targets += [meta['content'] for meta in page.tags('meta') if meta.get('property') == 'og:image']
        for target in targets:
            parsed = urlsplit(target)
            if parsed.scheme == 'mailto' or (parsed.netloc and parsed.netloc != urlsplit(base).netloc):
                continue
            if not parsed.path:
                assert any(attrs.get('id') == parsed.fragment for _, attrs in page.elements)
                continue
            assert parsed.path.startswith(prefix)
            destination = output / parsed.path.removeprefix(prefix)
            assert destination.is_file() or (destination / 'index.html').is_file(), target
        assert all(image.get('alt') for image in page.tags('img'))
        assert any(a.get('href') == '#main' for a in page.tags('a'))


@pytest.mark.parametrize('base', [
    '', 'http://example.org/', 'https://name:secret@example.org/',
    'https://example.org/?token=secret', 'https://example.org/#fragment',
    'https://example.org/../other/',
])
def test_invalid_public_urls_are_rejected_without_publication(builder, tmp_path, base):
    output = tmp_path / 'site'
    with pytest.raises(ValueError):
        builder.build_site(output, base_url=base)
    assert not output.exists()


def test_build_refuses_to_overwrite_or_publish_unrelated_files(builder, tmp_path):
    output = tmp_path / 'site'
    output.mkdir()
    unrelated = output / 'unrelated.txt'
    unrelated.write_text('Keep this private fixture unchanged')
    with pytest.raises(ValueError):
        builder.build_site(output, base_url='https://barissencan.com/jarvis/')
    assert unrelated.read_text() == 'Keep this private fixture unchanged'
    assert not (output / 'index.html').exists()


def test_rebuild_updates_all_canonical_urls_and_protects_added_files(builder, tmp_path):
    output = tmp_path / 'site'
    builder.build_site(output, base_url='https://isair.github.io/jarvis/')
    builder.build_site(output, base_url='https://barissencan.com/jarvis/')
    assert 'https://isair.github.io' not in (output / 'sitemap.xml').read_text()
    unrelated = output / 'unrelated.txt'
    unrelated.write_text('Keep this private fixture unchanged')
    original = (output / 'index.html').read_bytes()
    with pytest.raises(ValueError):
        builder.build_site(output, base_url='https://jarvis.example.org/')
    assert unrelated.read_text() == 'Keep this private fixture unchanged'
    assert (output / 'index.html').read_bytes() == original


@pytest.mark.parametrize('fault', ['parent-route', 'external-template', 'duplicate-route'])
def test_invalid_page_configuration_cannot_publish_outside_content(builder, monkeypatch, tmp_path, fault):
    import shutil
    source = tmp_path / 'website'
    shutil.copytree(ROOT / 'website', source)
    private = tmp_path / 'private.html'
    private.write_text('Private fixture must not be published')
    config_path = source / 'site.json'
    config = json.loads(config_path.read_text())
    if fault == 'parent-route':
        config['pages'][1]['route'] = '../escaped/'
    elif fault == 'external-template':
        config['pages'][1]['template'] = '../private.html'
    else:
        config['pages'][1]['route'] = config['pages'][0]['route']
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(builder, 'SOURCE', source)
    output = tmp_path / 'site'
    with pytest.raises(ValueError):
        builder.build_site(output, base_url='https://barissencan.com/jarvis/')
    assert not output.exists()
    assert not (tmp_path / 'escaped').exists()
    assert private.read_text() == 'Private fixture must not be published'


def test_output_symlink_does_not_modify_its_target(builder, tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    private = outside / 'unrelated.txt'
    private.write_text('Keep this private fixture unchanged')
    output = tmp_path / 'site'
    output.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        builder.build_site(output, base_url='https://barissencan.com/jarvis/')
    assert private.read_text() == 'Keep this private fixture unchanged'
    assert not (outside / 'index.html').exists()


def test_configured_routes_control_navigation_and_install_metadata(builder, monkeypatch, tmp_path):
    import shutil
    source = tmp_path / 'website'
    shutil.copytree(ROOT / 'website', source)
    config_path = source / 'site.json'
    config = json.loads(config_path.read_text())
    config['pages'][1]['route'] = 'start/'
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(builder, 'SOURCE', source)
    base = 'https://example.org/jarvis/'
    output = tmp_path / 'site'
    builder.build_site(output, base_url=base)
    home = Page((output / 'index.html').read_text())
    assert any(link.get('href') == '/jarvis/start/' for link in home.tags('a'))
    assert not any('/getting-started/' in link.get('href', '') for link in home.tags('a'))
    assert json.loads(''.join(home.structured))['installUrl'] == base + 'start/'
    assert (output / 'start/index.html').is_file()
    assert base + 'start/' in (output / 'sitemap.xml').read_text()
