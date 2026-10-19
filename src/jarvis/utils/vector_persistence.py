"""Atomic persistence for local vector-index refreshes."""
import sqlite3
from typing import Literal, Optional, Union

from ..debug import debug_log


def persist_vector(db_path: str, backend: Literal['python', 'faiss'], summary_id: int,
                   value: Union[str, bytes], source_text: Optional[str]) -> bool:
    """Store a vector only while its optional diary source is current."""
    table, column = {'python': ('python_vector_store', 'vector_json'),
                     'faiss': ('faiss_vector_store', 'vector_blob')}[backend]
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        if source_text is None:
            cursor = conn.execute(
                f'INSERT OR REPLACE INTO {table}(summary_id, {column}) VALUES (?, ?)',
                (summary_id, value),
            )
        else:
            cursor = conn.execute(
                f'''INSERT OR REPLACE INTO {table}(summary_id, {column})
                    SELECT ?, ? FROM conversation_summaries
                    WHERE id=? AND COALESCE(summary, '') || ' ' || COALESCE(topics, '') = ?''',
                (summary_id, value, summary_id, source_text),
            )
        accepted = cursor.rowcount == 1
        conn.commit()
        return accepted
    except Exception:
        debug_log(f'{backend} embedding persistence failed', 'memory')
        raise
    finally:
        if conn is not None:
            conn.close()
