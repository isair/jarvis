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
                self.vectors[summary_id] = np.array(json.loads(vector_json), dtype=np.float32)
            
            conn.close()
        except Exception:
            # If anything fails, just start with empty vectors
            pass
    
    def _save_vector(self, summary_id: int, vector: np.ndarray) -> None:
        """Persist a single vector to SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            vector_json = json.dumps(vector.tolist())
            cur.execute(
                "INSERT OR REPLACE INTO python_vector_store (summary_id, vector_json) VALUES (?, ?)",
                (summary_id, vector_json)
            )
            conn.commit()
            conn.close()
        except Exception:
            # Fail silently - in-memory still works
            pass
    
    def add_vector(self, summary_id: int, vector: List[float]) -> None:
        """Add or update a vector for a summary."""
        with self._lock:
            vec_array = np.array(vector, dtype=np.float32)
            # Normalize vector for cosine similarity
            norm = np.linalg.norm(vec_array)
            if norm > 0:
                vec_array = vec_array / norm
            self.vectors[summary_id] = vec_array
            self._save_vector(summary_id, vec_array)
    
    def search(self, query_vector: List[float], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for similar vectors using cosine similarity.
        Returns list of (summary_id, distance) tuples sorted by similarity.
        """
        with self._lock:
            if not self.vectors:
                return []
            
            # Normalize query vector
            query_array = np.array(query_vector, dtype=np.float32)
            query_norm = np.linalg.norm(query_array)
            if query_norm > 0:
                query_array = query_array / query_norm
            
            # Calculate cosine similarities
            similarities = []
            for summary_id, vector in self.vectors.items():
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


# Global instance
_python_vector_store: Optional[PythonVectorStore] = None


def get_python_vector_store(db_path: str) -> PythonVectorStore:
    """Get or create the global Python vector store instance."""
    global _python_vector_store
    if _python_vector_store is None:
        _python_vector_store = PythonVectorStore(db_path)
    return _python_vector_store


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
