"""Offline recall@3 check with labelled, controlled embeddings (no model download).

This measures fusion and multilingual keyword normalisation, not embedding-model
quality. Keyword queries and paraphrases share a corpus with weaker distractors.
"""

import json
import sqlite3

import pytest

from jarvis.memory.db import Database
from jarvis.utils.vector_store import PythonVectorStore

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
def test_hybrid_recall_at_three(tmp_path, mixed_dimensions, failed_refresh):
    db = Database(str(tmp_path / 'recall.db'))
    db._python_vector_store = PythonVectorStore(db.db_path)
    try:
        # Broad distractors are less similar than the query-specific target.
        for index in range(15):
            sid = db.upsert_conversation_summary(f'2026-01-{index + 1:02}', 'Ordinary daily notes')
            db.upsert_summary_embedding(sid, [1.] * len(CASES))
        targets = []
        for index, (keyword, _) in enumerate(CASES):
            sid = db.upsert_conversation_summary(f'2026-02-{index + 1:02}', ' '.join([keyword] * 3))
            db.upsert_summary_embedding(sid, [1.1 if i == index else 1. for i in range(len(CASES))])
            targets.append(sid)
        if mixed_dimensions:
            sid = db.upsert_conversation_summary('2026-03-01', 'Unrelated model history')
            db.upsert_summary_embedding(sid, [1.] * (len(CASES) + 1))
        if failed_refresh:
            with sqlite3.connect(db.db_path) as conn:
                conn.execute("CREATE TRIGGER reject_embedding BEFORE INSERT ON python_vector_store "
                             "BEGIN SELECT RAISE(ABORT, 'embedding rejected'); END")
            try:
                db.upsert_summary_embedding(targets[0], [1.] * len(CASES))
            except sqlite3.IntegrityError:
                pass
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
