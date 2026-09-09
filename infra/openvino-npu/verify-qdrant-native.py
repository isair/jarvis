"""Native NPU embedding -> Qdrant acceptance check."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

from native_embedding_runtime import NpuEmbeddingRuntime


QDRANT = "http://127.0.0.1:6333"
COLLECTION = "kelvin_qwen3_native_acceptance"


def request(method: str, path: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        QDRANT + path,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.loads(response.read())


def main() -> None:
    runtime = NpuEmbeddingRuntime(
        Path(__file__).parent / "models" / "Qwen3-Embedding-0.6B-int4-cw-ov" / "1"
    )
    texts = [
        "Qdrant stores normalized 1024-dimensional code embeddings.",
        "The settings panel uses a dark theme.",
    ]
    vectors = [runtime.embed([text])[0].tolist() for text in texts]
    try:
        request("DELETE", f"/collections/{COLLECTION}")
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
    request(
        "PUT",
        f"/collections/{COLLECTION}",
        {"vectors": {"size": 1024, "distance": "Cosine"}},
    )
    request(
        "PUT",
        f"/collections/{COLLECTION}/points?wait=true",
        {"points": [{"id": index + 1, "vector": vector, "payload": {"text": text}} for index, (vector, text) in enumerate(zip(vectors, texts))]},
    )
    query = runtime.embed(["Find Qdrant embedding configuration."])[0].tolist()
    result = request(
        "POST",
        f"/collections/{COLLECTION}/points/search",
        {"vector": query, "limit": 2, "with_payload": True},
    )
    top = result["result"][0]
    if top["payload"]["text"] != texts[0]:
        raise AssertionError(f"Unexpected top result: {top}")
    print(f"Qdrant native acceptance passed: collection_size=1024, distance=Cosine, top_score={top['score']:.6f}")


if __name__ == "__main__":
    main()
