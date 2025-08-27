"""
High-performance vector store implementation using FAISS for fast vector search.
This replaces the slow pure Python vector store with a much faster C++ implementation.
"""

import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import sqlite3
from pathlib import Path
import threading
import logging

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None


class FAISSVectorStore:
    """High-performance vector store using FAISS for fast similarity search."""
    
    def __init__(self, db_path: str, dimension: int = 768):
        """Initialize the FAISS vector store with a database path."""
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        self.db_path = db_path
        self.dimension = dimension
        self.index = None
        self.summary_id_to_index = {}  # Maps summary_id -> FAISS index position
        self.index_to_summary_id = {}  # Maps FAISS index position -> summary_id
        self._lock = threading.RLock()
        self._needs_rebuild = False
        
        self._load_vectors()
    
    def _load_vectors(self) -> None:
        """Load vectors from SQLite database and build FAISS index."""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            # Create table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS faiss_vector_store (
                    summary_id INTEGER PRIMARY KEY,
                    vector_blob BLOB NOT NULL
                )
            """)
            
            # Load existing vectors
            rows = cur.execute("SELECT summary_id, vector_blob FROM faiss_vector_store").fetchall()
            
            if rows:
                vectors = []
                summary_ids = []
                
                for summary_id, vector_blob in rows:
                    # Convert blob back to numpy array
                    vector = np.frombuffer(vector_blob, dtype=np.float32)
                    if len(vector) == self.dimension:
                        vectors.append(vector)
                        summary_ids.append(summary_id)
                
                if vectors:
                    # Build FAISS index
                    self._build_index(np.array(vectors), summary_ids)
            
            conn.close()
        except Exception as e:
            logging.warning(f"Failed to load FAISS vectors: {e}")
            # Start with empty index
            self._build_empty_index()
    
    def _build_empty_index(self) -> None:
        """Build an empty FAISS index."""
        with self._lock:
            # Use IndexFlatIP for cosine similarity (with normalized vectors)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.summary_id_to_index = {}
            self.index_to_summary_id = {}
    
    def _build_index(self, vectors: np.ndarray, summary_ids: List[int]) -> None:
        """Build FAISS index from vectors and summary IDs."""
        with self._lock:
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(vectors)
            
            # Use IndexFlatIP for exact cosine similarity search
            # For larger datasets, consider IndexHNSWFlat for approximate but faster search
            if len(vectors) > 10000:
                # Use HNSW for large datasets (approximate but much faster)
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 50
            else:
                # Use exact search for smaller datasets
                self.index = faiss.IndexFlatIP(self.dimension)
            
            # Add vectors to index
            self.index.add(vectors)
            
            # Build mapping between summary IDs and FAISS indices
            self.summary_id_to_index = {summary_id: i for i, summary_id in enumerate(summary_ids)}
            self.index_to_summary_id = {i: summary_id for i, summary_id in enumerate(summary_ids)}
    
    def _save_vector(self, summary_id: int, vector: np.ndarray) -> None:
        """Persist a single vector to SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            # Convert numpy array to blob
            vector_blob = vector.astype(np.float32).tobytes()
            cur.execute(
                "INSERT OR REPLACE INTO faiss_vector_store (summary_id, vector_blob) VALUES (?, ?)",
                (summary_id, vector_blob)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logging.warning(f"Failed to save vector to database: {e}")
    
    def add_vector(self, summary_id: int, vector: List[float]) -> None:
        """Add or update a vector for a summary."""
        with self._lock:
            vec_array = np.array(vector, dtype=np.float32)
            
            # Normalize vector for cosine similarity
            norm = np.linalg.norm(vec_array)
            if norm > 0:
                vec_array = vec_array / norm
            
            # If summary already exists, mark for rebuild
            if summary_id in self.summary_id_to_index:
                self._needs_rebuild = True
            
            # Save to database
            self._save_vector(summary_id, vec_array)
            
            # If index is empty or needs rebuild, rebuild from database
            if self.index is None or self.index.ntotal == 0 or self._needs_rebuild:
                self._load_vectors()
                self._needs_rebuild = False
            else:
                # Add new vector to existing index
                vec_array = vec_array.reshape(1, -1)
                faiss.normalize_L2(vec_array)
                
                index_pos = self.index.ntotal
                self.index.add(vec_array)
                self.summary_id_to_index[summary_id] = index_pos
                self.index_to_summary_id[index_pos] = summary_id
    
    def search(self, query_vector: List[float], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for similar vectors using FAISS.
        Returns list of (summary_id, distance) tuples sorted by similarity.
        """
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return []
            
            # Prepare query vector
            query_array = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            
            # Normalize query vector
            faiss.normalize_L2(query_array)
            
            # Search with FAISS
            k = min(top_k, self.index.ntotal)
            similarities, indices = self.index.search(query_array, k)
            
            # Convert to (summary_id, distance) format
            # FAISS IndexIP returns similarities (higher = better), convert to distances (lower = better)
            results = []
            for i in range(len(indices[0])):
                faiss_idx = indices[0][i]
                similarity = similarities[0][i]
                
                if faiss_idx >= 0 and faiss_idx in self.index_to_summary_id:
                    summary_id = self.index_to_summary_id[faiss_idx]
                    # Convert similarity to distance (1 - similarity)
                    distance = 1.0 - similarity
                    results.append((summary_id, float(distance)))
            
            return results
    
    def delete_vector(self, summary_id: int) -> None:
        """Remove a vector from the store."""
        with self._lock:
            if summary_id in self.summary_id_to_index:
                # Mark for rebuild (FAISS doesn't support efficient deletion)
                self._needs_rebuild = True
                
                # Remove from database
                try:
                    conn = sqlite3.connect(self.db_path)
                    cur = conn.cursor()
                    cur.execute("DELETE FROM faiss_vector_store WHERE summary_id = ?", (summary_id,))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    logging.warning(f"Failed to delete vector from database: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        with self._lock:
            return {
                "total_vectors": self.index.ntotal if self.index else 0,
                "dimension": self.dimension,
                "index_type": type(self.index).__name__ if self.index else None,
                "needs_rebuild": self._needs_rebuild,
                "faiss_available": FAISS_AVAILABLE,
            }


# Global instance
_faiss_vector_store: Optional[FAISSVectorStore] = None


def get_faiss_vector_store(db_path: str, dimension: int = 768) -> Optional[FAISSVectorStore]:
    """Get or create the global FAISS vector store instance."""
    global _faiss_vector_store
    
    if not FAISS_AVAILABLE:
        return None
    
    if _faiss_vector_store is None:
        try:
            _faiss_vector_store = FAISSVectorStore(db_path, dimension)
        except Exception as e:
            logging.warning(f"Failed to create FAISS vector store: {e}")
            return None
    
    return _faiss_vector_store
