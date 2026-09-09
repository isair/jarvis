from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class EmbeddingRequest:
    model: str
    inputs: list[str]
    operation: Literal["query", "index"]

    @staticmethod
    def parse(payload: dict[str, Any]) -> "EmbeddingRequest":
        values = payload.get("input")
        if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
            raise ValueError("input must be a non-empty array of strings")
        if len(values) > 500:
            raise ValueError("input exceeds the 500-item request limit")
        operation = payload.get("operation", "query")
        if operation not in ("query", "index"):
            raise ValueError("operation must be query or index")
        return EmbeddingRequest(str(payload.get("model", "Qwen3-Embedding-0.6B-int4")), values, operation)


@dataclass(frozen=True)
class RerankRequest:
    model: str
    query: str
    documents: list[dict[str, Any]]
    top_n: int

    @staticmethod
    def parse(payload: dict[str, Any]) -> "RerankRequest":
        documents = payload.get("documents")
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip() or not isinstance(documents, list):
            raise ValueError("query and documents are required")
        if len(documents) > 50:
            raise ValueError("documents exceeds the 50-candidate limit")
        normalized = [doc if isinstance(doc, dict) and isinstance(doc.get("text"), str) else None for doc in documents]
        if any(doc is None for doc in normalized):
            raise ValueError("each document must be an object containing text")
        top_n = min(max(int(payload.get("top_n", 8)), 1), len(documents))
        return RerankRequest(str(payload.get("model", "Qwen3-Reranker-0.6B-int8")), query, normalized, top_n)  # type: ignore[arg-type]
