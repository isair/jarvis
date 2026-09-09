from __future__ import annotations

from pathlib import Path

from native_reranker_runtime import NpuRerankerRuntime
from compiled_model_registry import cache_root as versioned_cache_root


class RerankerService:
    model_name = "Qwen3-Reranker-0.6B-int8"

    def __init__(self, model_dir: Path, cache_root: Path) -> None:
        self.profiles = {
            512: NpuRerankerRuntime(model_dir, 512, versioned_cache_root(cache_root, "2026.2.1", "32.0.100.4778", model_dir, "reranker-b1-s512")),
            1024: NpuRerankerRuntime(model_dir, 1024, versioned_cache_root(cache_root, "2026.2.1", "32.0.100.4778", model_dir, "reranker-b1-s1024")),
        }

    @staticmethod
    def bucket_length(query: str, document: str) -> int:
        # Tokenizer-aware truncation is performed by the runtime. Bucketing by
        # characters keeps routing deterministic before tokenization.
        return 512 if len(query) + len(document) <= 1800 else 1024

    def score(self, query: str, document: str) -> tuple[float, str]:
        length = self.bucket_length(query, document)
        return self.profiles[length].score(query, document), f"b1-s{length}"
