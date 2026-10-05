"""Hybrid diary retrieval uses relevance order, not incompatible score scales."""

import json
import subprocess
import sys
import textwrap

import pytest

from jarvis.memory.db import Database
from jarvis.utils.vector_store import PythonVectorStore

pytestmark = pytest.mark.unit


@pytest.fixture
def diary(tmp_path, monkeypatch):
    monkeypatch.setattr('jarvis.utils.vector_store.get_best_vector_store', lambda path, dimension: PythonVectorStore(path))
    db = Database(str(tmp_path / 'diary.db'))
    db._python_vector_store = PythonVectorStore(db.db_path)
    yield db
    db.close()


def test_keyword_match_beats_equal_distance_nonmatch(diary):
    missing = diary.upsert_conversation_summary('2026-01-01', 'Ordinary daily notes')
    hit = diary.upsert_conversation_summary('2026-01-02', 'quasar quasar quasar')
    for day in range(3, 25):
        diary.upsert_conversation_summary(f'2026-01-{day:02}', 'Ordinary daily notes')
    for sid in (missing, hit):
        diary.upsert_summary_embedding(sid, [1., 0.], diary.get_summary_embedding_text(sid))
    results = diary.search_hybrid('quasar', json.dumps([1., 0.]))
    assert results[0]['id'] == hit
    assert all(row['score'] > 0 for row in results)


def test_stronger_keyword_match_wins_with_equal_vectors(diary):
    weak = diary.upsert_conversation_summary('2026-01-01', 'quasar and many ordinary daily notes')
    strong = diary.upsert_conversation_summary('2026-01-02', 'quasar quasar quasar')
    for day in range(3, 25):
        diary.upsert_conversation_summary(f'2026-01-{day:02}', 'Ordinary daily notes')
    for sid in (weak, strong):
        diary.upsert_summary_embedding(sid, [1., 0.], diary.get_summary_embedding_text(sid))
    assert diary.search_hybrid('quasar', '[1, 0]')[0]['id'] == strong


def test_keyword_only_candidate_and_semantic_only_candidate_are_retained(diary):
    keyword = diary.upsert_conversation_summary('2026-01-01', 'quasar')
    semantic = diary.upsert_conversation_summary('2026-01-02', 'distant star')
    diary.upsert_summary_embedding(semantic, [1., 0.], diary.get_summary_embedding_text(semantic))
    results = diary.search_hybrid('quasar', '[1, 0]')
    assert {row['id'] for row in results} == {keyword, semantic}
    assert len(diary.search_hybrid('quasar', '[1, 0]', top_k=1)) == 1


def test_no_embeddings_preserves_fts_order(diary):
    diary.upsert_conversation_summary('2026-01-01', 'quasar and ordinary daily notes')
    diary.upsert_conversation_summary('2026-01-02', 'quasar quasar quasar')
    lexical = diary.search_hybrid('quasar', None)
    hybrid = diary.search_hybrid('quasar', '[1, 0]')
    assert [r['id'] for r in hybrid] == [r['id'] for r in lexical]


def test_sqlite_vss_keyword_ranking(tmp_path):
    pytest.importorskip('sqlite_vss')
    # Isolate the extension's bundled Faiss from the Python Faiss shared library.
    result = subprocess.run([sys.executable, '-c', textwrap.dedent('''
        import json, sys, sqlite_vss
        from unittest.mock import patch
        from jarvis.memory.db import Database
        from jarvis.utils.vector_store import PythonVectorStore
        with patch('jarvis.utils.vector_store.get_best_vector_store', side_effect=lambda path, dimension: PythonVectorStore(path)):
            diary = Database(sys.argv[1])
        diary.conn.enable_load_extension(True)
        sqlite_vss.load(diary.conn)
        diary.is_vss_enabled = True
        diary._init_schema()
        missing = diary.upsert_conversation_summary('2026-01-01', 'Ordinary notes')
        hit = diary.upsert_conversation_summary('2026-01-02', 'quasar quasar quasar')
        for day in range(3, 25):
            diary.upsert_conversation_summary(f'2026-01-{day:02}', 'Ordinary notes')
        vector = [1.] + [0.] * 767
        for sid in (missing, hit):
            diary.upsert_summary_embedding(sid, vector, diary.get_summary_embedding_text(sid))
        assert diary.search_hybrid('quasar', json.dumps(vector))[0]['id'] == hit
        diary.close()
    '''), str(tmp_path / 'vss.db')], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('word', ['京都', 'велосипед', 'θερμόμετρο', 'नमस्ते', 'قهوة', 'café', 'cafe\u0301', 'İstanbul', '한글'])
def test_unicode_keyword_query_finds_the_matching_diary_not_recent_noise(diary, word):
    target = diary.upsert_conversation_summary('2026-01-01', word)
    diary.upsert_conversation_summary('2026-02-01', 'Unrelated recent notes')
    rows = diary.search_hybrid(word, None, top_k=1)
    assert [row['id'] for row in rows] == [target]


@pytest.mark.parametrize('word', ['京都', 'велосипед', 'θερμόμετρο', 'قهوة'])
def test_unicode_keyword_match_participates_in_hybrid_ranking(diary, word):
    target = diary.upsert_conversation_summary('2026-01-01', word)
    noise = diary.upsert_conversation_summary('2026-02-01', 'Unrelated recent notes')
    diary.upsert_summary_embedding(target, [1., 1.], diary.get_summary_embedding_text(target))
    diary.upsert_summary_embedding(noise, [1., 0.], diary.get_summary_embedding_text(noise))
    assert diary.search_hybrid(word, '[1, 0]', top_k=1)[0]['id'] == target


def test_multilingual_keywords_keep_or_recall_without_embeddings(diary):
    from types import SimpleNamespace
    from jarvis.memory.conversation import search_conversation_memory_by_keywords
    diary.upsert_conversation_summary('2026-01-01', '京都')
    diary.upsert_conversation_summary('2026-01-02', 'cycling')
    diary.upsert_conversation_summary('2026-02-01', 'Unrelated recent notes')
    contexts = search_conversation_memory_by_keywords(
        diary, ['京都', 'cycling'], SimpleNamespace(embedding_model=''),
    )
    assert len(contexts) == 2
    assert any('京都' in text for text in contexts)
    assert any('cycling' in text for text in contexts)
