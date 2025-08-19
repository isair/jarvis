from __future__ import annotations
import json
from .db import Database
from .embed import get_embedding


def retrieve_top_chunks(db: Database, query: str, ollama_base_url: str, embed_model: str,
                         top_k: int = 8) -> list[tuple[int, float, str]]:
    try:
        vec = get_embedding(query, ollama_base_url, embed_model)
        vec_json = json.dumps(vec) if vec is not None else None
    except Exception:
        vec_json = None
    rows = db.search_hybrid(query, vec_json, top_k=top_k)
    return [(int(r[0]), float(r[1]), str(r[2])) for r in rows]
