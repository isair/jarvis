"""Validate and normalise local semantic-index vectors."""
import numpy as np


def normalise_embedding(vector, dimension=None) -> np.ndarray:
    """Return a finite unit vector, rejecting unusable embeddings."""
    try:
        values = np.asarray(vector, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Embedding must contain numeric values') from exc
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError('Embedding must be a non-empty finite vector')
    if dimension is not None and values.size != dimension:
        raise ValueError('Embedding dimension does not match the index')
    scale = np.max(np.abs(values))
    if scale == 0:
        raise ValueError('Embedding must have a non-zero direction')
    # Scaling first avoids overflow and underflow in the squared norm.
    values = values / scale
    return (values / np.linalg.norm(values)).astype(np.float32)
