"""
Pure Python vector store implementation for out-of-the-box vector search.
Falls back to this when sqlite-vss is not available.
"""

import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import sqlite3
from pathlib import Path
import threading
from weakref import WeakValueDictionary
from .embedding_vector import normalise_embedding
from ..debug import debug_log


class PythonVectorStore:
    """Simple in-memory vector store with SQLite persistence."""
    
    def __init__(self, db_path: str):
        """Initialize the vector store with a database path."""
        self.db_path = db_path
        self.vectors: Dict[int, np.ndarray] = {}  # summary_id -> vector
        self._lock = threading.RLock()
        self._load_vectors()
    
    def _load_vectors(self) -> None:
        """Load vectors from SQLite database."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            # Create table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS python_vector_store (
                    summary_id INTEGER PRIMARY KEY,
                    vector_json TEXT NOT NULL
                )
            """)
            
            # Load existing vectors
            rows = cur.execute("SELECT summary_id, vector_json FROM python_vector_store").fetchall()
            for summary_id, vector_json in rows:
                try:
                    self.vectors[summary_id] = normalise_embedding(json.loads(vector_json))
                except (TypeError, ValueError, OverflowError):
                    debug_log('Skipping an unusable persisted Python embedding', 'memory')
        except Exception:
            # If anything fails, just start with empty vectors
            pass
        finally:
            if conn is not None:
                conn.close()
    
    def _save_vector(self, summary_id: int, vector: np.ndarray) -> None:
        """Persist a single vector to SQLite."""
        if self.db_path == ':memory:':
            return
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            vector_json = json.dumps(vector.tolist())
            cur.execute(
                "INSERT OR REPLACE INTO python_vector_store (summary_id, vector_json) VALUES (?, ?)",
                (summary_id, vector_json)
            )
            conn.commit()
        except Exception:
            debug_log('Python embedding persistence failed', 'memory')
            raise
        finally:
            if conn is not None:
                conn.close()
    
    def add_vector(self, summary_id: int, vector: List[float]) -> None:
        """Add or update a vector for a summary."""
        with self._lock:
            vec_array = normalise_embedding(vector)
            self._save_vector(summary_id, vec_array)
            self.vectors[summary_id] = vec_array
    
    def search(self, query_vector: List[float], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for similar vectors using cosine similarity.
        Returns list of (summary_id, distance) tuples sorted by similarity.
        """
        with self._lock:
            if not self.vectors:
                return []
            
            try:
                query_array = normalise_embedding(query_vector)
            except ValueError:
                debug_log('Skipping semantic search for an unusable query embedding', 'memory')
                return []
            
            # Calculate cosine similarities
            similarities = []
            for summary_id, vector in self.vectors.items():
                if vector.size != query_array.size:
                    continue
                # Cosine similarity = dot product of normalized vectors
                similarity = np.dot(query_array, vector)
                # Convert to distance (lower is better, like sqlite-vss)
                distance = 1.0 - similarity
                similarities.append((summary_id, distance))
            
            # Sort by distance (ascending) and return top k
            similarities.sort(key=lambda x: x[1])
            return similarities[:top_k]
    
    def delete_vector(self, summary_id: int) -> None:
        """Remove a vector from the store."""
        with self._lock:
            if summary_id in self.vectors:
                del self.vectors[summary_id]
                try:
                    conn = sqlite3.connect(self.db_path)
                    cur = conn.cursor()
                    cur.execute("DELETE FROM python_vector_store WHERE summary_id = ?", (summary_id,))
                    conn.commit()
                    conn.close()
                except Exception:
                    pass


_python_stores: WeakValueDictionary[str, PythonVectorStore] = WeakValueDictionary()
_python_stores_lock = threading.RLock()


def get_python_vector_store(db_path: str) -> PythonVectorStore:
    """Share an index only between active owners of the same database file."""
    if str(db_path) == ':memory:':
        return PythonVectorStore(db_path)
    key = str(Path(db_path).resolve())
    with _python_stores_lock:
        store = _python_stores.get(key)
        if store is None:
            store = PythonVectorStore(key)
            _python_stores[key] = store
        return store


def get_best_vector_store(db_path: str, dimension: int = 768):
    """Get the best available vector store (FAISS if available, otherwise Python fallback)."""
    # Try FAISS first (much faster)
    try:
        from .fast_vector_store import get_faiss_vector_store
        faiss_store = get_faiss_vector_store(db_path, dimension)
        if faiss_store is not None:
            return faiss_store
    except ImportError:
        pass
    
    # Fallback to Python implementation
    return get_python_vector_store(db_path)
