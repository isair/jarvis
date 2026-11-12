"""Build the public product pages without importing the assistant runtime."""
import argparse
from html import escape
import json
from pathlib import Path
import shutil
from string import Template
import tempfile
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parent
MANIFEST = '.jarvis-site.json'
IMAGES = ('face.png', 'chat-window.png', 'setup-provider.png')


def canonical_base(value):
    parts = urlsplit(value)
    if (parts.scheme != 'https' or not parts.hostname or parts.username or parts.password
            or parts.query or parts.fragment or '\\' in value
            or any(part in ('.', '..') for part in unquote(parts.path).split('/'))):
        raise ValueError('Canonical URL must be HTTPS without credentials, query, fragment or relative segments')
    return value.rstrip('/') + '/'


def _owned_output(destination):
    if destination.is_symlink():
        raise ValueError('Output must not be a symbolic link')
    if destination.exists() and not destination.is_dir():
        raise ValueError('Output must be a directory')
    if not destination.exists() or not any(destination.iterdir()):
        return
    manifest = destination / MANIFEST
    if not manifest.is_file():
        raise ValueError('Output contains files that do not belong to this site build')
    entries = list(destination.rglob('*'))
    if any(entry.is_symlink() for entry in entries):
        raise ValueError('Output must not contain symbolic links')
    try:
        expected = set(json.loads(manifest.read_text())['files']) | {MANIFEST}
    except (ValueError, KeyError, TypeError):
        raise ValueError('Output manifest is invalid; use a fresh output directory') from None
    actual = {entry.relative_to(destination).as_posix() for entry in entries if entry.is_file()}
    if actual != expected:
        raise ValueError('Output contains added or missing files; use a fresh output directory')


def _validate_pages(pages):
    routes = []
    for page in pages:
        route = page['route']
        if route and (route.startswith('/') or not route.endswith('/')
                      or any(part in ('', '.', '..') for part in route[:-1].split('/'))
                      or any(char in route for char in '\\?#%')
                      or any(char.isspace() for char in route)):
            raise ValueError('Page routes must be relative directory paths')
        template = (SOURCE / page['template']).resolve()
        try:
            template.relative_to(SOURCE.resolve())
        except ValueError:
            raise ValueError('Page templates must be inside the website source directory') from None
        if not template.is_file():
            raise ValueError('Every page must have a local template file')
        routes.append(route)
    if len(set(routes)) != len(routes):
        raise ValueError('Every page must have a unique route')
    keys = [page['key'] for page in pages]
    if len(set(keys)) != len(keys) or not {'home', 'getting_started', 'privacy'} <= set(keys):
        raise ValueError('Page keys must be unique and include the shared navigation pages')


def build_site(destination, *, base_url=None):
    """Render a complete static bundle, refusing unrelated output files."""
    destination = Path(destination)
    site = json.loads((SOURCE / 'site.json').read_text())
    base = canonical_base(site['base_url'] if base_url is None else base_url)
    _validate_pages(site['pages'])
    _owned_output(destination)
    prefix = urlsplit(base).path
    routes = {page['key']: page['route'] for page in site['pages']}
    source_url = site['repository'] + '/blob/' + site['source_branch']
    common = {
        **{key: escape(prefix + route, quote=True) for key, route in routes.items()},
        'source': escape(source_url, quote=True),
        'license': escape(source_url + '/LICENSE', quote=True),
        'assets': escape(prefix + 'assets/', quote=True),
        'repository': escape(site['repository'], quote=True),
        'summary': escape(site['summary']),
        'author_name': escape(site['author']['name']),
        'author_url': escape(site['author']['url'], quote=True),
    }
    layout = Template((SOURCE / 'layout.html').read_text())
    namespace = 'http://www.sitemaps.org/schemas/sitemap/0.9'
    ET.register_namespace('', namespace)
    sitemap = ET.Element(f'{{{namespace}}}urlset')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='jarvis-site-', dir=destination.parent) as staging:
        bundle = Path(staging)
        for page in site['pages']:
            url = base + page['route']
            data = {
                '@context': 'https://schema.org',
                '@type': 'SoftwareApplication' if page['key'] == 'home' else 'WebPage',
                'name': site['name'] if page['key'] == 'home' else page['title'],
                'url': url,
                'description': site['summary'] if page['key'] == 'home' else page['description'],
            }
            if page['key'] == 'home':
                data.update({
                    'applicationCategory': 'ProductivityApplication',
                    'operatingSystem': ['macOS', 'Windows', 'Linux'],
                    'sameAs': site['repository'],
                    'license': source_url + '/LICENSE',
                    'installUrl': base + routes['getting_started'],
                    'author': dict({'@type': 'Person'}, **site['author']),
                })
            values = dict(common, title=escape(page['title']), description=escape(page['description'], quote=True),
                          canonical=escape(url, quote=True), social_image=escape(base + 'assets/face.png', quote=True),
                          schema=json.dumps(data, ensure_ascii=False).replace('<', '\\u003c'))
            values['body'] = Template((SOURCE / page['template']).read_text()).substitute(common)
            target = bundle / page['route'] / 'index.html'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(layout.substitute(values), encoding='utf-8')
            ET.SubElement(ET.SubElement(sitemap, f'{{{namespace}}}url'), f'{{{namespace}}}loc').text = url
        assets = bundle / 'assets'
        assets.mkdir()
        shutil.copyfile(SOURCE / 'style.css', assets / 'style.css')
        for name in IMAGES:
            shutil.copyfile(ROOT / 'docs/img' / name, assets / name)
        ET.ElementTree(sitemap).write(bundle / 'sitemap.xml', encoding='utf-8', xml_declaration=True)
        (bundle / '.nojekyll').touch()
        files = sorted(path.relative_to(bundle).as_posix() for path in bundle.rglob('*') if path.is_file())
        (bundle / MANIFEST).write_text(json.dumps({'files': files}), encoding='utf-8')
        _owned_output(destination)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(bundle, destination)
    return len(site['pages']), base


def main():
    parser = argparse.ArgumentParser(description='🏗️ Build the Jarvis public product site')
    parser.add_argument('--output', type=Path, default=ROOT / '_site')
    parser.add_argument('--base-url')
    args = parser.parse_args()
    try:
        count, base = build_site(args.output, base_url=args.base_url)
    except ValueError as error:
        parser.error(str(error))
    print(f'✅ Built {count} public pages in {args.output}\n   🌐 Canonical base: {base}')


if __name__ == '__main__':
    main()
