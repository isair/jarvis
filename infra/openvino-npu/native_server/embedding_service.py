from __future__ import annotations

from pathlib import Path

from native_embedding_runtime import NpuEmbeddingRuntime
from compiled_model_registry import cache_root


class EmbeddingService:
    model_name = "Qwen3-Embedding-0.6B-int4"

    def __init__(self, model_dir: Path, cache_dir: Path) -> None:
        self.runtime = NpuEmbeddingRuntime(
            model_dir,
            max_length=2048,
            cache_dir=cache_root(cache_dir, "2026.2.1", "32.0.100.4778", model_dir, "embedding-b1-s2048"),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.runtime.embed([text])[0].tolist() for text in texts]
