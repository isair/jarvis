"""Reply-owned discovery and progressive loading of authorised local skills."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import unicodedata

import yaml

from .debug import debug_log
from .memory.document_files import open_document_file, scan_document_directory


SKILL_TOOLS = frozenset(('searchSkills', 'readSkill'))


def _tokens(text):
    return set(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def without_skill_messages(messages):
    """Remove ephemeral skill traffic while preserving other native call pairs."""
    removed_ids = {
        call.get('id') for msg in messages for call in (msg.get('tool_calls') or [])
        if call.get('function', {}).get('name') in SKILL_TOOLS and call.get('id')
    }
    uses_skills = bool(removed_ids) or any(msg.get('tool_name') in SKILL_TOOLS for msg in messages)
    kept = []
    for msg in messages:
        if msg.get('tool_name') in SKILL_TOOLS or msg.get('tool_call_id') in removed_ids:
            continue
        if msg.get('tool_calls'):
            calls = [call for call in msg['tool_calls']
                     if call.get('function', {}).get('name') not in SKILL_TOOLS]
            if calls:
                # Later ordinary calls can also quote loaded guidance in reasoning.
                if uses_skills:
                    clean = {key: value for key, value in msg.items() if key != 'thinking'}
                    kept.append({**clean, 'tool_calls': calls, 'content': ''})
                else:
                    kept.append(msg)
        else:
            kept.append(msg)
    return kept


class SkillSession:
    """A local metadata index and bounded instruction set for one reply."""

    HEADER_BYTES = 8192
    METADATA_BYTES = 4096
    FILE_BYTES = 12288
    GUIDANCE_BYTES = 16384

    def __init__(self, cfg, cancel_event=None):
        self.enabled = cfg.skills_enabled is True
        self.cancel_event = cancel_event
        self._catalogue = {}
        self._index = defaultdict(set)
        self._loaded = {}
        self._metadata_bytes = 0
        self._guidance_bytes = 0
        if not self.enabled:
            return
        skipped = 0
        for raw in cfg.skills_paths:
            if self._cancelled():
                break
            try:
                root = Path(raw).expanduser().resolve(strict=True)
                with scan_document_directory(root) as (_, entries):
                    for entry in entries:
                        if self._cancelled():
                            break
                        if not entry.directory:
                            continue
                        path = root / entry.name
                        key = hashlib.sha256(str(path).encode('utf-8')).hexdigest()
                        if key in self._catalogue:
                            continue
                        try:
                            meta = self._metadata(path)
                            self._catalogue[key] = {**meta, 'id': key, 'path': path}
                            for token in _tokens(meta['name'] + ' ' + meta['description']):
                                self._index[token].add(key)
                        except (OSError, ValueError, UnicodeError, yaml.YAMLError, RecursionError):
                            skipped += 1
            except (OSError, ValueError):
                skipped += 1
        debug_log(f'skills: indexed={len(self._catalogue)}, skipped={skipped}', 'tools')

    @property
    def catalogue_size(self):
        return len(self._catalogue)

    def _cancelled(self):
        return self.cancel_event is not None and self.cancel_event.is_set()

    def _metadata(self, path):
        with open_document_file(path / 'SKILL.md') as stream:
            if stream.readline(5).strip() != b'---':
                raise ValueError('Missing skill frontmatter')
            header = bytearray()
            while len(header) <= self.HEADER_BYTES:
                line = stream.readline(self.HEADER_BYTES + 1 - len(header))
                if line.strip() == b'---':
                    break
                if not line:
                    raise ValueError('Incomplete skill frontmatter')
                header.extend(line)
            else:
                raise ValueError('Skill frontmatter is too large')
        text = header.decode('utf-8')
        # Scanning does not construct or expand alias-backed merge mappings.
        if any(isinstance(token, yaml.tokens.AliasToken) for token in yaml.scan(text)):
            raise ValueError('Skill frontmatter cannot contain YAML aliases')
        meta = yaml.safe_load(text)
        if not isinstance(meta, dict):
            raise ValueError('Skill frontmatter must be a mapping')
        name, description = meta.get('name'), meta.get('description')
        if (not isinstance(name, str) or name != path.name or not 1 <= len(name) <= 64
                or name != name.lower() or name.startswith('-') or name.endswith('-')
                or '--' in name or not all(c.isalnum() or c == '-' for c in name)):
            raise ValueError('Invalid skill name')
        if not isinstance(description, str) or not 1 <= len(description.strip()) <= 1024:
            raise ValueError('Invalid skill description')
        return {'name': name, 'description': description.strip()}

    def search(self, query):
        if not self.enabled or self._cancelled() or not isinstance(query, str):
            return []
        scores = Counter()
        for token in _tokens(query):
            for key in self._index.get(token, ()):
                scores[key] += 1
        normalised = unicodedata.normalize('NFKC', query).casefold()
        ranked = sorted(scores, key=lambda key: (
            -int(unicodedata.normalize('NFKC', self._catalogue[key]['name']).casefold() in normalised),
            -scores[key], self._catalogue[key]['name'], key,
        ))
        hits = []
        for key in ranked:
            item = {field: self._catalogue[key][field] for field in ('id', 'name', 'description')}
            item['path'] = key + '/SKILL.md'
            description = item['description'].encode('utf-8')
            if len(description) > 512:
                item['description'] = description[:512].decode('utf-8', errors='ignore')
                item['description_truncated'] = True
            candidate = hits + [item]
            size = len(json.dumps(candidate, ensure_ascii=False).encode('utf-8'))
            if size + self._metadata_bytes <= self.METADATA_BYTES:
                hits = candidate
            if len(hits) == 5:
                break
        if hits:
            self._metadata_bytes += len(json.dumps(hits, ensure_ascii=False).encode('utf-8'))
        debug_log(f'skills search: matches={len(hits)}, metadata_bytes={self._metadata_bytes}', 'tools')
        return hits

    def read(self, skill, relative='SKILL.md'):
        if not self.enabled:
            raise ValueError('Local skills are disabled')
        if self._cancelled():
            raise ValueError('Skill loading was cancelled')
        if not isinstance(skill, str) or not isinstance(relative, str):
            raise ValueError('Skill and file must be strings')
        matches = [item for key, item in self._catalogue.items()
                   if key == skill or item['name'] == skill]
        if len(matches) != 1:
            raise ValueError('Unknown or ambiguous skill; search and use its exact ID')
        path = PurePosixPath(relative)
        if (not relative or path.is_absolute() or '..' in path.parts
                or '\\' in relative or ':' in relative):
            raise ValueError('Skill files must be relative paths within the package')
        item = matches[0]
        identity = (item['id'], str(path))
        if identity in self._loaded:
            return {'skill': item['name'], 'file': str(path), 'already_loaded': True,
                    'note': 'Use the complete instructions in the earlier readSkill result.'}
        with open_document_file(item['path'] / path) as stream:
            raw = stream.read(self.FILE_BYTES + 1)
        if len(raw) > self.FILE_BYTES:
            raise ValueError('Skill file is too large; use smaller reference files')
        text = raw.decode('utf-8')
        if self._cancelled():
            raise ValueError('Skill loading was cancelled')
        result = {'skill': item['name'], 'file': str(path), 'instructions': text}
        block = json.dumps(result, ensure_ascii=False)
        size = len(block.encode('utf-8'))
        if size + self._guidance_bytes > self.GUIDANCE_BYTES:
            raise ValueError('The skill instruction budget for this reply is exhausted')
        self._loaded[identity] = True
        self._guidance_bytes += size
        debug_log(f'skills load: files={len(self._loaded)}, guidance_bytes={self._guidance_bytes}', 'tools')
        return result
