"""Offline recall@3 check with labelled, controlled embeddings (no model download).

This measures fusion and multilingual keyword normalisation, not embedding-model
quality. Keyword queries and paraphrases share a corpus with weaker distractors.
"""

import json
import sqlite3

import pytest

from jarvis.memory.db import Database
from jarvis.utils.vector_store import PythonVectorStore, get_python_vector_store

pytestmark = pytest.mark.eval

CASES = [
    ('京都', '日本の古都'), ('велосипед', 'двухколёсный транспорт'),
    ('θερμόμετρο', 'μέτρηση θερμοκρασίας'), ('قهوة', 'مشروب الصباح'),
    ('한글', '한국 문자'), ('नमस्ते', 'अभिवादन'),
    ('picnic', 'outdoor lunch'), ('guitar', 'string instrument'),
    ('passport', 'travel document'), ('dentist', 'tooth appointment'),
    ('marathon', 'long distance race'), ('allergy', 'pollen reaction'),
    ('mortgage', 'house loan'), ('bicycle', 'two wheeled transport'),
    ('birthday', 'annual celebration'),
]


@pytest.mark.parametrize('mixed_dimensions', [False, True], ids=['current-model', 'mixed-model-history'])
@pytest.mark.parametrize('failed_refresh', [False, True], ids=['healthy-index', 'rejected-refresh'])
@pytest.mark.parametrize('updated_text', [False, True], ids=['initial-text', 'updated-text'])
def test_hybrid_recall_at_three(tmp_path, mixed_dimensions, failed_refresh, updated_text):
    db = Database(str(tmp_path / 'recall.db'))
    db._python_vector_store = PythonVectorStore(db.db_path)
    try:
        # Broad distractors are less similar than the query-specific target.
        for index in range(15):
            sid = db.upsert_conversation_summary(f'2026-01-{index + 1:02}', 'Ordinary daily notes')
            db.upsert_summary_embedding(sid, [1.] * len(CASES), db.get_summary_embedding_text(sid))
        targets = []
        for index, (keyword, _) in enumerate(CASES):
            sid = db.upsert_conversation_summary(f'2026-02-{index + 1:02}', ' '.join([keyword] * 3))
            db.upsert_summary_embedding(sid, [1.1 if i == index else 1. for i in range(len(CASES))], db.get_summary_embedding_text(sid))
            targets.append(sid)
        if mixed_dimensions:
            sid = db.upsert_conversation_summary('2026-03-01', 'Unrelated model history')
            db.upsert_summary_embedding(sid, [1.] * (len(CASES) + 1), db.get_summary_embedding_text(sid))
        if failed_refresh:
            with sqlite3.connect(db.db_path) as conn:
                conn.execute("CREATE TRIGGER reject_embedding BEFORE INSERT ON python_vector_store "
                             "BEGIN SELECT RAISE(ABORT, 'embedding rejected'); END")
            try:
                db.upsert_summary_embedding(targets[0], [1.] * len(CASES), db.get_summary_embedding_text(targets[0]))
            except sqlite3.IntegrityError:
                pass
        if updated_text:
            for index, (keyword, _) in enumerate(CASES):
                targets[index] = db.upsert_conversation_summary(
                    f'2026-02-{index + 1:02}', ' '.join([keyword] * 3) + ' Additional diary note.',
                )
        results = []
        for subset, query_index in (('lexical', 0), ('semantic', 1)):
            hits = {'fts': 0, 'hybrid': 0}
            for index, case in enumerate(CASES):
                vector = json.dumps([1.1 if i == index else 1. for i in range(len(CASES))])
                for mode, embedding in (('fts', None), ('hybrid', vector)):
                    rows = db.search_hybrid(case[query_index], embedding, top_k=3)
                    hits[mode] += targets[index] in [row['id'] for row in rows]
            print(f"📊 {subset} recall@3: FTS {hits['fts']}/{len(CASES)}, hybrid {hits['hybrid']}/{len(CASES)}")
            results.append(hits)
        for hits in results:
            assert hits['hybrid'] >= hits['fts']
            assert hits['hybrid'] == len(CASES)
    finally:
        db.close()


def test_superseded_refresh_preserves_semantic_recall(tmp_path, monkeypatch):
    monkeypatch.setattr('jarvis.utils.vector_store.get_best_vector_store',
                        lambda path, dimension: get_python_vector_store(path))
    first = Database(str(tmp_path / 'freshness.db'))
    second = Database(first.db_path)
    try:
        ident = first.upsert_conversation_summary('2026-01-01', 'The user likes coffee.', 'drinks')
        source = first.get_summary_embedding_text(ident)
        first.upsert_summary_embedding(ident, [0., 1.], source)
        competitor = first.upsert_conversation_summary('2026-01-02', 'The user enjoys cycling.', 'sports')
        first.upsert_summary_embedding(competitor, [0.6, 0.8], first.get_summary_embedding_text(competitor))
        second.upsert_conversation_summary('2026-01-01', 'The user renews a passport.', 'travel')
        second.upsert_summary_embedding(ident, [1., 0.], second.get_summary_embedding_text(ident))
        first.upsert_summary_embedding(ident, [0., 1.], source)
        hits = second.search_hybrid('identity document', json.dumps([1., 0.]), top_k=1)
        recalled = bool(hits and hits[0]['id'] == ident)
        print(f'📊 Superseded-refresh semantic recall@1: {int(recalled)}/1')
        assert recalled
    finally:
        first.close()
        second.close()
