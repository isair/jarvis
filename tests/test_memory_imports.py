"""Local sources become reviewed memories without inference or source writes."""
import json
from contextlib import closing
from pathlib import Path
import pytest
from jarvis.memory.graph import GraphMemoryStore
from jarvis.memory import imports

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


def source(path, *, scope='personal', provider='ChatGPT'):
    return imports.MemorySource(provider, 'Saved memories', scope, path)


def test_claude_discovery_retains_project_scope_and_respects_disabled_memory(tmp_path):
    root = tmp_path/'.claude'
    notes = root/'projects'/'test-project'/'memory'
    notes.mkdir(parents=True)
    (root/'CLAUDE.md').write_text('Use British English.', encoding='utf-8')
    (notes/'MEMORY.md').write_text('Project build commands.', encoding='utf-8')
    found = imports.discover_sources(tmp_path)
    assert {(s.path.name, s.scope) for s in found.sources} == {('CLAUDE.md', 'instructions'), ('MEMORY.md', 'project')}
    (root/'settings.json').write_text(json.dumps({'autoMemoryEnabled': False}), encoding='utf-8')
    assert [(s.path.name, s.scope) for s in imports.discover_sources(tmp_path).sources] == [('CLAUDE.md', 'instructions')]


def test_custom_claude_directory_and_memory_directory_are_used(tmp_path):
    root, memory = tmp_path/'configured-claude', tmp_path/'notes'
    root.mkdir(); memory.mkdir()
    (root/'settings.json').write_text(json.dumps({'autoMemoryDirectory': str(memory)}), encoding='utf-8')
    (memory/'preferences.md').write_text('Keep answers concise.', encoding='utf-8')
    assert [s.path for s in imports.discover_sources(tmp_path, claude_config_dir=root).sources] == [memory/'preferences.md']


@pytest.mark.parametrize('text', ['I live in London.\n\nKeep replies concise.', 'Londra’da yaşıyorum.\n\n短い回答が好きです。'])
def test_preview_preserves_wording_without_saving_or_changing_source(tmp_path, text):
    path = tmp_path/'memories.txt'; path.write_text(text, encoding='utf-8')
    preview = imports.read_source(source(path))
    assert [i.text for i in preview.items] == text.split('\n\n')
    assert path.read_text(encoding='utf-8') == text
    assert not list(tmp_path.glob('*.db'))


def test_secrets_and_rich_markup_are_data_in_the_preview():
    preview = imports.preview_text('Claude', '<img src="https://example.com/x">\n\nContact user@example.com', label='Pasted memories')
    assert preview.items[0].text == '<img src="https://example.com/x">'
    assert 'user@example.com' not in preview.items[1].text


@pytest.mark.parametrize('mode', ['symlink', 'oversized', 'binary'])
def test_unsafe_source_is_rejected_without_partial_preview(tmp_path, mode):
    path = tmp_path/'memories.txt'
    if mode == 'symlink':
        target = tmp_path/'target.txt'; target.write_text('Private text')
        path.symlink_to(target)
    elif mode == 'oversized':
        path.write_bytes(b'x'*(imports.MAX_SOURCE_BYTES+1))
    else:
        path.write_bytes(b'\x00\xff')
    with pytest.raises(imports.MemoryImportError):
        imports.read_source(source(path))


def test_import_is_explicit_and_destinations_remain_distinct(tmp_path):
    preview = imports.preview_text('ChatGPT', 'I live in London.\n\nUse British English.\n\nProject build notes.', label='Saved memories')
    with closing(GraphMemoryStore(str(tmp_path/'test.db'))) as store:
        choices = [imports.ImportChoice(preview.items[0].id, preview.items[0].text, 'user', source_date=None),
                   imports.ImportChoice(preview.items[1].id, preview.items[1].text, 'directives', source_date=None)]
        result = imports.commit_import(store, preview, choices)
        assert result.created == 2
        assert [n.data for n in store.get_children('user')] == ['I live in London.']
        assert [n.data for n in store.get_children('directives')] == ['Use British English.']
        assert not store.get_children('world')
        assert not store.search_nodes('Project build notes')


def test_repeat_import_preserves_human_edits_and_source_quote(tmp_path):
    preview = imports.preview_text('Claude', 'I prefer concise replies.', label='Memory export')
    choice = imports.ImportChoice(preview.items[0].id, 'I prefer brief replies.', 'user', source_date=None)
    with closing(GraphMemoryStore(str(tmp_path/'test.db'))) as store:
        first = imports.commit_import(store, preview, [choice])
        node = store.get_children('user')[0]
        store.update_node(node.id, data='A manually edited preference.')
        second = imports.commit_import(store, preview, [choice])
        assert first.created == 1 and second.created == 0
        assert store.get_node(node.id).data == 'A manually edited preference.'
        records = store.get_import_provenance(node.id)
        assert len(records) == 1
        assert records[0]['original_text'] == preview.items[0].text
        assert records[0]['reviewed_text'] == choice.text
        assert records[0]['source_date'] is None


def test_source_date_is_preserved_and_is_not_an_import_date(tmp_path):
    preview = imports.preview_text('ChatGPT', 'An outing was planned for tomorrow.', label='Old memories')
    with closing(GraphMemoryStore(str(tmp_path/'test.db'))) as store:
        imports.commit_import(store, preview, [imports.ImportChoice(preview.items[0].id, preview.items[0].text, 'world', source_date='2020-12-31')])
        node = store.get_children('world')[0]
        record = store.get_import_provenance(node.id)[0]
        assert record['source_date'] == '2020-12-31'
        assert node.data == preview.items[0].text
        assert record['imported_at'] != record['source_date']


def test_invalid_selection_rolls_back_the_whole_batch(tmp_path):
    preview = imports.preview_text('Claude', 'One memory.\n\nAnother memory.', label='Export')
    with closing(GraphMemoryStore(str(tmp_path/'test.db'))) as store:
        choices = [imports.ImportChoice(preview.items[0].id, 'One memory.', 'user', source_date=None),
                   imports.ImportChoice(preview.items[1].id, 'Another memory.', 'root', source_date=None)]
        with pytest.raises(imports.MemoryImportError):
            imports.commit_import(store, preview, choices)
        assert not store.get_children('user')


def test_failed_sql_write_rolls_back_nodes_provenance_and_profile_revision(tmp_path):
    import sqlite3
    preview = imports.preview_text('Claude', 'First memory.\n\nSecond memory.', label='Export')
    choices = [imports.ImportChoice(item.id, item.text, 'user', source_date=None) for item in preview.items]
    with closing(GraphMemoryStore(str(tmp_path/'test.db'))) as store:
        revision = store.get_warm_profile_revision()
        store.conn.execute("CREATE TRIGGER reject_second BEFORE INSERT ON memory_nodes WHEN NEW.data = 'Second memory.' BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END")
        store.conn.commit()
        with pytest.raises(sqlite3.DatabaseError):
            imports.commit_import(store, preview, choices)
        assert not store.get_children('user')
        assert not store.get_import_provenance()
        assert store.get_warm_profile_revision() == revision
        store.conn.execute('DROP TRIGGER reject_second'); store.conn.commit()
        assert imports.commit_import(store, preview, choices).created == 2


def test_concurrent_imports_have_one_saved_memory_and_source_record(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    preview = imports.preview_text('ChatGPT', 'I prefer Celsius.', label='Memories')
    choices = [imports.ImportChoice(preview.items[0].id, preview.items[0].text, 'user', source_date=None)]
    path = str(tmp_path/'test.db')
    with closing(GraphMemoryStore(path)) as first, closing(GraphMemoryStore(path)) as second:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda store: imports.commit_import(store, preview, choices), [first, second]))
        assert sum(result.created for result in results) == 1
        assert len(first.get_children('user')) == 1
        assert len(first.get_import_provenance()) == 1


def test_provenance_survives_split_and_deletion_without_resurrecting_memories(tmp_path):
    preview = imports.preview_text('Claude', 'Original reference text.', label='Project notes')
    choice = imports.ImportChoice(preview.items[0].id, preview.items[0].text, 'world', source_date=None)
    with closing(GraphMemoryStore(str(tmp_path/'test.db'))) as store:
        imports.commit_import(store, preview, [choice])
        original = store.get_children('world')[0]
        assert store.split_node(original, children=[('Part', 'Reference part', 'Selected reference detail.')], summary='Reference categories')
        child = store.get_children(original.id)[0]
        assert store.get_import_provenance(child.id)[0]['original_text'] == original.data
        store.delete_node(original.id)
        record = store.get_import_provenance()[0]
        assert record['node_id'] is None and record['original_text'] == original.data
        assert imports.commit_import(store, preview, [choice]).created == 0
        assert not store.get_children('world')


def test_duplicate_text_reuses_existing_node_but_retains_separate_providers(tmp_path):
    with closing(GraphMemoryStore(str(tmp_path/'test.db'))) as store:
        node = store.create_node('Preference', 'User preference', 'I prefer Celsius.', 'user')
        for provider in ('ChatGPT', 'Claude'):
            preview = imports.preview_text(provider, 'I prefer Celsius.', label='Memories')
            result = imports.commit_import(store, preview, [imports.ImportChoice(preview.items[0].id, preview.items[0].text, 'user', source_date=None)])
            assert result.created == 0
        assert [n.id for n in store.get_children('user')] == [node.id]
        assert {r['provider'] for r in store.get_import_provenance(node.id)} == {'ChatGPT', 'Claude'}


def test_large_edited_selection_is_rejected_before_writing(tmp_path):
    preview = imports.preview_text('Claude', '\n\n'.join(f'Memory {i}' for i in range(40)), label='Export')
    choices = [imports.ImportChoice(item.id, 'x'*imports.MAX_ENTRY_CHARS, 'user', source_date=None) for item in preview.items]
    with closing(GraphMemoryStore(str(tmp_path/'test.db'))) as store:
        with pytest.raises(imports.MemoryImportError):
            imports.commit_import(store, preview, choices)
        assert not store.get_children('user')


def test_claude_topic_metadata_is_provenance_and_not_memory_text(tmp_path):
    path = tmp_path/'preference.md'
    path.write_text('---\nname: Response style\ndescription: User preference\ntype: feedback\nmodified: 2024-03-12T10:20:00+00:00\n---\n\nKeep replies concise.', encoding='utf-8')
    preview = imports.read_source(source(path, provider='Claude Code', scope='project'))
    assert [item.text for item in preview.items] == ['Keep replies concise.']
    assert dict(preview.source_metadata)['modified'] == '2024-03-12T10:20:00+00:00'
    with closing(GraphMemoryStore(str(tmp_path/'test.db'))) as store:
        imports.commit_import(store, preview, [imports.ImportChoice(preview.items[0].id, preview.items[0].text, 'world', source_date=None)])
        record = store.get_import_provenance()[0]
        assert record['source_date'] is None
        assert record['source_metadata']['type'] == 'feedback'
        assert record['source_metadata']['modified'] == '2024-03-12T10:20:00+00:00'


def test_disabled_auto_memory_inventory_keeps_user_instructions(tmp_path):
    root = tmp_path/'.claude'; notes = root/'projects'/'test'/'memory'
    notes.mkdir(parents=True)
    (root/'CLAUDE.md').write_text('Use British English.')
    (notes/'MEMORY.md').write_text('Project memory.')
    found = imports.discover_sources(tmp_path, disable_auto_memory=True)
    assert [item.scope for item in found.sources] == ['instructions']


def test_equal_text_with_different_source_dates_keeps_distinct_memories(tmp_path):
    text = 'An outing is planned for tomorrow.'
    with closing(GraphMemoryStore(str(tmp_path/'test.db'))) as store:
        for day in ('2020-01-01', '2026-10-12'):
            preview = imports.preview_text('Claude', text, label='Memories')
            assert imports.commit_import(store, preview, [imports.ImportChoice(preview.items[0].id, text, 'world', source_date=day)]).created == 1
        nodes = store.get_children('world')
        assert len(nodes) == 2
        assert {store.get_import_provenance(node.id)[0]['source_date'] for node in nodes} == {'2020-01-01', '2026-10-12'}


def test_child_keeps_source_date_when_its_imported_parent_is_deleted(tmp_path):
    preview = imports.preview_text('Claude', 'Original note.', label='Notes')
    with closing(GraphMemoryStore(str(tmp_path/'test.db'))) as store:
        imports.commit_import(store, preview, [imports.ImportChoice(preview.items[0].id, preview.items[0].text, 'world', source_date='2020-01-01')])
        parent = store.get_children('world')[0]
        assert store.split_node(parent, children=[('Part', 'Part', 'Original note.')], summary='Parts')
        child = store.get_children(parent.id)[0]
        store.delete_node(parent.id)
        assert store.get_import_provenance(child.id)[0]['source_date'] == '2020-01-01'


def test_recall_context_carries_original_dates_without_import_timestamp(tmp_path):
    preview = imports.preview_text('Claude', 'An outing is planned for tomorrow.', label='Notes')
    with closing(GraphMemoryStore(str(tmp_path/'test.db'))) as store:
        imports.commit_import(store, preview, [imports.ImportChoice(preview.items[0].id, preview.items[0].text, 'world', source_date='2020-01-01')])
        node = store.get_children('world')[0]
        context = store.get_context_data(node, max_chars=800)
        assert '2020-01-01' in context and node.data in context
        assert store.get_import_provenance(node.id)[0]['imported_at'] not in context
        assert len(store.get_context_data(node, max_chars=300)) <= 300


def test_metadata_edits_keep_dates_but_data_writes_invalidate_the_binding(tmp_path):
    preview = imports.preview_text('Claude', 'An outing is planned for tomorrow.', label='Notes')
    with closing(GraphMemoryStore(str(tmp_path/'test.db'))) as store:
        imports.commit_import(store, preview, [imports.ImportChoice(preview.items[0].id, preview.items[0].text, 'user', source_date='2020-01-01')])
        node = store.get_children('user')[0]
        store.update_node(node.id, name='Renamed note')
        assert store.get_import_provenance(node.id)[0]['current_binding'] == 1
        store.conn.execute('UPDATE memory_nodes SET data = data WHERE id = ?', (node.id,)); store.conn.commit()
        assert store.get_import_provenance(node.id)[0]['current_binding'] == 0
        frame = store.get_context_data(store.get_node(node.id), max_chars=800)
        data = json.loads(frame.split(imports.IMPORT_CONTEXT_BEGIN+'\n')[1].split('\n'+imports.IMPORT_CONTEXT_END)[0])
        assert data['date'] is None
        assert data['archived_quotes'][0]['date'] == '2020-01-01'


def test_multiline_claude_header_is_provenance_not_personal_memory(tmp_path):
    path=tmp_path/'topic.md'
    path.write_text('---\nname: Project instructions\ndescription: |\n  Notes for this repository\npaths:\n  - "src/**"\nmodified: 2026-10-12T01:00:00Z\n---\n\nUse the documented build command.',encoding='utf-8')
    preview=imports.read_source(source(path,provider='Claude Code',scope='instructions'))
    assert [item.text for item in preview.items]==['Use the documented build command.']
    assert preview.source.scope=='project'
    assert '2026-10-12' in dict(preview.source_metadata)['header']
