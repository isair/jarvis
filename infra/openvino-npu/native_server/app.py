from __future__ import annotations

import json
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1]))

from contracts import EmbeddingRequest, RerankRequest
from embedding_service import EmbeddingService
from inference_queue import InferenceQueue
from metrics import Metrics
from reranker_service import RerankerService


ROOT = Path(__file__).parents[1]
MODEL_ROOT = ROOT / "models"
EMBED_MODEL = MODEL_ROOT / "Qwen3-Embedding-0.6B-int4-cw-ov" / "1"
RERANK_MODEL = MODEL_ROOT / "Qwen3-Reranker-0.6B-int8-ov" / "1"
CACHE_ROOT = ROOT / "cache"
PORT = int(os.environ.get("OPENVINO_NPU_PORT", "8010"))


class NativeNpuApplication:
    def __init__(self) -> None:
        self.ready = False
        self.initializing = False
        self.initialization_error: str | None = None
        self.queue = InferenceQueue()
        self.metrics = Metrics()
        self.embedding: EmbeddingService | None = None
        self.reranker: RerankerService | None = None

    def initialize(self) -> None:
        self.initializing = True
        self.initialization_error = None
        try:
            print(json.dumps({"event": "native_retrieval_initialization_started", "service": "native-npu-retrieval", "embedding_model": str(EMBED_MODEL), "reranker_model": str(RERANK_MODEL), "device": "NPU"}), flush=True)
            self.embedding = EmbeddingService(EMBED_MODEL, CACHE_ROOT)
            self.reranker = RerankerService(RERANK_MODEL, CACHE_ROOT)
            for profile, runtime in (("embedding-b1-s2048", self.embedding.runtime), ("reranker-b1-s512", self.reranker.profiles[512]), ("reranker-b1-s1024", self.reranker.profiles[1024])):
                print(json.dumps({"event": "compiled_model_ready", "profile": profile, "cache_status": runtime.cache_status, "compile_or_import_ms": runtime.compile_or_import_ms, "cache_path": str(runtime.cache_path), "execution_devices": ["NPU"]}), flush=True)
            self.embedding.embed(["native service warmup"])
            self.reranker.score("warmup query", "warmup document")
            self.ready = True
            print(json.dumps({"event": "native_retrieval_ready", "service": "native-npu-retrieval", "device": "NPU"}), flush=True)
        except Exception as error:
            self.initialization_error = f"{type(error).__name__}: {error}"
            print(json.dumps({"event": "native_retrieval_initialization_failed", "service": "native-npu-retrieval", "error": self.initialization_error}), flush=True)
            traceback.print_exc()
        finally:
            self.initializing = False

    def close(self) -> None:
        self.queue.close()


app = NativeNpuApplication()


class Handler(BaseHTTPRequestHandler):
    def _json(self, status: int, payload: Any) -> None:
        data = json.dumps(payload).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except ConnectionError:
            self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/health", "/healthz"):
            return self._json(200, {"status": "ok", "service": "native-npu-retrieval"})
        if self.path == "/readyz":
            return self._json(200 if app.ready else 503, {"ready": app.ready, "initializing": app.initializing, "error": app.initialization_error})
        if self.path == "/v1/capabilities":
            if not app.ready:
                return self._json(503, {"error": app.initialization_error or "Native retrieval service is still initializing", "ready": False})
            return self._json(200, {"device": "NPU", "embedding_dimensions": 1024, "embedding_max_length": 2048, "reranker_profiles": ["b1-s512", "b1-s1024"], "retrieval_profiles": {"interactive": {"candidates": 12, "rerank_top_n": 8}, "deep_research": {"candidates": 20, "rerank_top_n": 8}, "background": {"candidates": 40, "rerank_top_n": 20}}})
        if self.path == "/metrics":
            data = app.metrics.prometheus().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not app.ready:
            return self._json(503, {"error": app.initialization_error or "Native retrieval service is still initializing", "ready": False})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            if self.path == "/v1/embeddings":
                request = EmbeddingRequest.parse(payload)
                started = time.perf_counter()
                priority = 1 if request.operation == "query" else 2
                vectors, queue_seconds, inference_seconds = app.queue.submit_timed(priority, lambda: app.embedding.embed(request.inputs))  # type: ignore[union-attr]
                app.metrics.observe("embedding", time.perf_counter() - started, queue_seconds, inference_seconds)
                return self._json(200, {"data": [{"index": i, "embedding": vector} for i, vector in enumerate(vectors)], "dimensions": 1024, "normalized": True, "device": "NPU", "model": request.model, "operation": request.operation, "queue_wait_seconds": queue_seconds, "inference_seconds": inference_seconds, "request_seconds": time.perf_counter() - started})
            if self.path == "/v1/rerank":
                request = RerankRequest.parse(payload)
                started = time.perf_counter()
                scored, queue_seconds, inference_seconds = app.queue.submit_timed(0, lambda: [(i, *app.reranker.score(request.query, doc["text"])) for i, doc in enumerate(request.documents)])  # type: ignore[union-attr]
                app.metrics.observe("rerank", time.perf_counter() - started, queue_seconds, inference_seconds)
                ranked = sorted(scored, key=lambda item: item[1], reverse=True)[: request.top_n]
                return self._json(200, {"results": [{"id": request.documents[i].get("id", str(i)), "index": i, "score": score} for i, score, _profile in ranked], "profile": ranked[0][2] if ranked else "b1-s512", "device": "NPU", "model": request.model})
            self._json(404, {"error": "not found"})
        except Exception as error:
            self._json(400, {"error": str(error)})


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Native OpenVINO NPU service listening on http://127.0.0.1:{PORT}", flush=True)
    Thread(target=app.initialize, name="native-retrieval-initializer", daemon=True).start()
    try:
        server.serve_forever()
    finally:
        app.close()


if __name__ == "__main__":
    main()
