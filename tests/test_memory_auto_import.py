"""Source consent and incremental imports use owned local files and graphs."""
from contextlib import closing
import pytest
import sqlite3
from jarvis.memory.graph import GraphMemoryStore
from jarvis.memory.imports import MemorySource, ImportChoice, read_source, commit_import
from jarvis.memory import auto_import

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]

@pytest.fixture
def imported(tmp_path):
    path = tmp_path/'memories.md'
    path.write_text('I live in London.\n\nI cycle to work.', encoding='utf-8')
    preview = read_source(MemorySource('ChatGPT', 'Memories', 'personal', path))
    with closing(GraphMemoryStore(str(tmp_path/'graph.db'))) as store:
        choices = [ImportChoice(i.id, i.text, 'user', None) for i in preview.items]
        result = commit_import(store, preview, choices)
        yield store, path, preview, choices, result

def offer(imported):
    store, path, preview, choices, result = imported
    return auto_import.record_success(store, preview, choices, result)

def enable(imported):
    grant = offer(imported)
    return auto_import.decide(imported[0], grant.source_id, True)

def texts(store):
    return {node.data for node in store.get_children('user')}

def test_declined_offer_is_persistent_and_not_repeated(imported):
    store, _, preview, choices, result = imported
    grant = offer(imported)
    auto_import.decide(store, grant.source_id, False)
    assert auto_import.record_success(store, preview, choices, result) is None
    with closing(GraphMemoryStore(store.db_path)) as reopened:
        assert auto_import.load_sources(reopened) == []
        assert auto_import.record_success(reopened, preview, choices, result) is None

def test_unchanged_source_has_no_database_mutations(imported):
    store, _, preview, _, _ = imported
    grant = enable(imported)
    before = store.conn.total_changes
    assert auto_import.sync_source(store, grant.source_id, grant.generation, preview) == 'unchanged'
    assert store.conn.total_changes == before

def test_changed_source_replaces_only_its_pristine_entries(imported):
    store, path, preview, _, _ = imported
    grant = enable(imported)
    path.write_text('I live in Bath.\n\nI cycle to work.', encoding='utf-8')
    assert auto_import.sync_source(store, grant.source_id, grant.generation, read_source(preview.source)) == 'updated'
    assert texts(store) == {'I live in Bath.', 'I cycle to work.'}
    records = store.get_import_provenance()
    assert any(r['reviewed_text']=='I live in London.' and r['node_id'] is None for r in records)
    assert all(r['source_date'] is None for r in records)
    assert auto_import.sync_source(store, grant.source_id, grant.generation, read_source(preview.source)) == 'unchanged'

@pytest.mark.parametrize('edit', ['text', 'move', 'delete', 'share', 'child'])
def test_source_refresh_preserves_human_or_shared_work_and_requires_review(imported, edit):
    store, path, preview, _, result = imported
    grant = enable(imported)
    node_id = result.node_ids[0]
    if edit == 'text': store.update_node(node_id, data='My own correction.')
    elif edit == 'move': store.update_node(node_id, parent_id='world')
    elif edit == 'delete': store.delete_node(node_id)
    elif edit == 'child': store.create_node('Child', 'Human addition', 'Keep this', node_id)
    else:
        other = MemorySource('Claude', 'Another source', 'personal', None)
        from jarvis.memory.imports import preview_text
        p = preview_text('Claude', 'I live in London.', label=other.label)
        commit_import(store, p, [ImportChoice(p.items[0].id, p.items[0].text, 'user', None)])
    before = [dict(row) for row in store.conn.execute('SELECT * FROM memory_nodes ORDER BY id')]
    path.write_text('I live in Bath.\n\nI cycle to work.', encoding='utf-8')
    assert auto_import.sync_source(store, grant.source_id, grant.generation, read_source(preview.source)) == 'review'
    assert [dict(row) for row in store.conn.execute('SELECT * FROM memory_nodes ORDER BY id')] == before
    assert auto_import.load_sources(store) == []

def test_revocation_blocks_an_already_read_update(imported):
    store, path, preview, _, _ = imported
    grant = enable(imported)
    path.write_text('I live in Bath.', encoding='utf-8')
    pending = read_source(preview.source)
    auto_import.decide(store, grant.source_id, False)
    assert auto_import.sync_source(store, grant.source_id, grant.generation, pending) == 'revoked'
    assert texts(store) == {'I live in London.', 'I cycle to work.'}

def test_partial_or_edited_review_cannot_silently_approve_the_full_source(imported):
    store, _, preview, choices, result = imported
    assert not auto_import.record_success(store, preview, choices[:1], result).can_enable

def test_failed_refresh_rolls_back_graph_and_checkpoint(imported):
    store, path, preview, _, _ = imported
    grant = enable(imported)
    store.conn.execute("CREATE TEMP TRIGGER fail_import BEFORE INSERT ON memory_imports BEGIN SELECT RAISE(ABORT, 'test failure'); END")
    path.write_text('I live in Bath.', encoding='utf-8')
    with pytest.raises(sqlite3.IntegrityError, match='test failure'):
        auto_import.sync_source(store, grant.source_id, grant.generation, read_source(preview.source))
    assert texts(store) == {'I live in London.', 'I cycle to work.'}
    assert auto_import.load_sources(store)[0].digest == preview.source_digest
    assert store.conn.execute('SELECT COUNT(*) FROM memory_import_retired').fetchone()[0]==0

def test_reverting_a_source_reactivates_its_old_text_without_duplicates(imported):
    store, path, preview, _, _ = imported
    grant = enable(imported)
    for text in ('I live in Bath.', 'I live in London.\n\nI cycle to work.', 'I live in Bath.', 'I live in London.\n\nI cycle to work.'):
        path.write_text(text, encoding='utf-8')
        assert auto_import.sync_source(store, grant.source_id, grant.generation, read_source(preview.source)) == 'updated'
        assert texts(store) == set(text.split('\n\n'))


def test_auto_refresh_does_not_bind_new_text_to_a_different_human_edit(imported):
    store, path, preview, _, _ = imported
    grant = enable(imported)
    from jarvis.memory.imports import preview_text
    p = preview_text('Claude', 'I live in Bath.', label='Another source')
    other = commit_import(store, p, [ImportChoice(p.items[0].id,p.items[0].text,'user',None)])
    store.update_node(other.node_ids[0], data='Human edit unrelated to Bath.')
    path.write_text('I live in Bath.', encoding='utf-8')
    assert auto_import.sync_source(store, grant.source_id, grant.generation, read_source(preview.source)) == 'updated'
    assert texts(store) == {'I live in Bath.', 'Human edit unrelated to Bath.'}


def test_setup_generations_never_reuse_an_old_permission(imported):
    store,path,preview,choices,result=imported
    original=enable(imported)
    auto_import.decide(store,original.source_id,False)
    auto_import.begin_setup(store,original.source_id)
    renewed=auto_import.record_success(store,preview,choices,result)
    current=auto_import.decide(store,renewed.source_id,True)
    assert current.generation>original.generation
    path.write_text('I live in Bath.')
    assert auto_import.sync_source(store,original.source_id,original.generation,read_source(preview.source))=='revoked'
    assert texts(store)=={'I live in London.','I cycle to work.'}


def test_unchanged_entry_keeps_its_explicit_date_when_other_entries_change(tmp_path):
    path=tmp_path/'source.md';path.write_text('I visited York.\n\nI cycle to work.')
    p=read_source(MemorySource('Claude','Memory','personal',path))
    choices=[ImportChoice(i.id,i.text,'user','2026-10-01' if index==0 else None)
             for index,i in enumerate(p.items)]
    with closing(GraphMemoryStore(str(tmp_path/'graph.db'))) as store:
        result=commit_import(store,p,choices)
        grant=auto_import.record_success(store,p,choices,result)
        grant=auto_import.decide(store,grant.source_id,True)
        path.write_text('I visited York.\n\nI walk to work.')
        auto_import.sync_source(store,grant.source_id,grant.generation,read_source(p.source))
        assert store.get_import_provenance(result.node_ids[0])[0]['source_date']=='2026-10-01'
        latest=[r for r in store.get_import_provenance() if r['reviewed_text']=='I walk to work.']
        assert latest[0]['source_date'] is None

def test_reviewed_setup_can_restore_an_importer_retired_version(imported):
    store,path,preview,choices,result=imported
    grant=enable(imported)
    path.write_text('I live in Bath.')
    auto_import.sync_source(store,grant.source_id,grant.generation,read_source(preview.source))
    auto_import.decide(store,grant.source_id,False)
    auto_import.begin_setup(store,grant.source_id)
    path.write_text('I live in London.\n\nI cycle to work.')
    p=read_source(preview.source)
    result=commit_import(store,p,choices)
    renewed=auto_import.record_success(store,p,choices,result)
    assert renewed.can_enable
    assert {'I live in London.','I cycle to work.'} <= texts(store)
    before=len(store.get_import_provenance())
    commit_import(store,p,choices)
    assert len(store.get_import_provenance())==before


def test_archived_versions_do_not_increase_live_candidate_work(tmp_path):
    steps=[]
    for count in (0,5000):
        directory=tmp_path/str(count);directory.mkdir()
        path=directory/'source.md';path.write_text('I live in Rome.')
        p=read_source(MemorySource('ChatGPT','Memory','personal',path))
        with closing(GraphMemoryStore(str(directory/'graph.db'))) as store:
            choices=[ImportChoice(i.id,i.text,'user',None) for i in p.items]
            result=commit_import(store,p,choices)
            grant=auto_import.record_success(store,p,choices,result)
            grant=auto_import.decide(store,grant.source_id,True)
            store.conn.execute('''WITH RECURSIVE sequence(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM sequence WHERE n<?)
                INSERT INTO memory_imports SELECT 'archived-'||n,branch,'I live in Bath.',NULL,
                provider,source_id,source_digest,source_label,source_scope,source_date,source_metadata,
                'I live in Bath.','I live in Bath.',imported_at FROM memory_imports,sequence WHERE n<=?''',(count,count))
            store.conn.commit()
            path.write_text('I live in Bath.')
            updated=read_source(p.source)
            work=[];store.conn.set_progress_handler(lambda:(work.append(1),0)[1],10)
            auto_import.sync_source(store,grant.source_id,grant.generation,updated)
            store.conn.set_progress_handler(None,0)
            assert texts(store)=={'I live in Bath.'}
            steps.append(len(work))
    assert steps[1] <= 2*steps[0]+200,steps


def test_declined_sources_do_not_consume_active_watch_capacity(tmp_path):
    with closing(GraphMemoryStore(str(tmp_path/'graph.db'))) as store:
        for index in range(auto_import.MAX_AUTO_SOURCES+1):
            path=tmp_path/f'{index}.md';path.write_text(f'Reference number {index}.')
            p=read_source(MemorySource('Claude','Memory','personal',path))
            choices=[ImportChoice(i.id,i.text,'world',None) for i in p.items]
            result=commit_import(store,p,choices)
            grant=auto_import.record_success(store,p,choices,result)
            assert grant is not None
            auto_import.decide(store,grant.source_id,False)
        auto_import.begin_setup(store,grant.source_id)
        renewed=auto_import.record_success(store,p,choices,result)
        auto_import.decide(store,renewed.source_id,True)
        assert len(auto_import.load_sources(store))==1


def test_active_watch_capacity_is_explicit_and_disabling_releases_it(tmp_path):
    grants=[]
    with closing(GraphMemoryStore(str(tmp_path/'graph.db'))) as store:
        for index in range(auto_import.MAX_AUTO_SOURCES+1):
            path=tmp_path/f'{index}.md';path.write_text(f'Selected reference {index}.')
            p=read_source(MemorySource('Claude','Memory','personal',path))
            choices=[ImportChoice(i.id,i.text,'world',None) for i in p.items]
            result=commit_import(store,p,choices)
            grant=auto_import.record_success(store,p,choices,result)
            grants.append(grant)
            if index<auto_import.MAX_AUTO_SOURCES:
                auto_import.decide(store,grant.source_id,True)
        with pytest.raises(auto_import.MemoryImportError,match='Disable another source'):
            auto_import.decide(store,grants[-1].source_id,True)
        auto_import.decide(store,grants[0].source_id,False)
        auto_import.decide(store,grants[-1].source_id,True)
        assert len(auto_import.load_sources(store))==auto_import.MAX_AUTO_SOURCES
        shown=auto_import.load_sources(store,include_disabled=True)
        assert sum(g.state=='enabled' for g in shown)==auto_import.MAX_AUTO_SOURCES


def test_a_human_deleted_entry_cannot_be_recreated_by_setup(imported):
    store,_,preview,choices,result=imported
    store.delete_node(result.node_ids[0])
    repeated=commit_import(store,preview,choices)
    grant=auto_import.record_success(store,preview,choices,repeated)
    assert not grant.can_enable
    with pytest.raises(auto_import.MemoryImportError,match='complete local file'):
        auto_import.decide(store,grant.source_id,True)
    assert 'I live in London.' not in texts(store)

def test_consent_history_does_not_increase_monitor_work(tmp_path):
    steps=[]
    for count in (0,5000):
        directory=tmp_path/str(count);directory.mkdir()
        path=directory/'source.md';path.write_text('Owned reference.')
        p=read_source(MemorySource('Claude','Reference','personal',path))
        with closing(GraphMemoryStore(str(directory/'graph.db'))) as store:
            choices=[ImportChoice(i.id,i.text,'world',None) for i in p.items]
            result=commit_import(store,p,choices)
            grant=auto_import.record_success(store,p,choices,result)
            auto_import.decide(store,grant.source_id,True)
            store.conn.execute('''WITH RECURSIVE sequence(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM sequence WHERE n<?)
                INSERT INTO memory_auto_sources SELECT 'declined-'||n,source_json,'disabled',0,digest,branch,bindings,can_enable,'' FROM memory_auto_sources,sequence WHERE n<=?''',(count,count))
            store.conn.commit()
            work=[];store.conn.set_progress_handler(lambda:(work.append(1),0)[1],10)
            active=auto_import.load_sources(store)
            managed=auto_import.load_sources(store,include_disabled=True)
            store.conn.set_progress_handler(None,0)
            assert len(active)==1
            assert any(g.source_id==grant.source_id for g in managed)
            assert len(managed)<=2*auto_import.MAX_AUTO_SOURCES
            steps.append(len(work))
    assert steps[1]<=2*steps[0]+200,steps
