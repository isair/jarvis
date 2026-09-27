"""Offline recall@3 check with labelled, controlled embeddings (no model download).

This measures fusion, not embedding-model quality. Nine keyword queries and nine
paraphrases share a 24-summary corpus with semantically weaker distractors.
"""

import json

import pytest

from jarvis.memory.db import Database
from jarvis.utils.vector_store import PythonVectorStore

pytestmark = pytest.mark.eval

CASES = [
    ('picnic', 'outdoor lunch'), ('guitar', 'string instrument'),
    ('passport', 'travel document'), ('dentist', 'tooth appointment'),
    ('marathon', 'long distance race'), ('allergy', 'pollen reaction'),
    ('mortgage', 'house loan'), ('bicycle', 'two wheeled transport'),
    ('birthday', 'annual celebration'),
]


def test_hybrid_recall_at_three(tmp_path):
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
