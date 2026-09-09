"""Bounded native OpenVINO NPU embedding runtime for Qwen3 embeddings.

This is intentionally independent from OVMS so the model contract can be
validated before adding HTTP and Qdrant integration. The model emits token
embeddings; this module performs masked mean pooling and L2 normalization to
produce one 1024-dimensional vector per input text.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import openvino as ov
from transformers import AutoTokenizer


DEFAULT_MODEL = Path(__file__).parent / "models" / "Qwen3-Embedding-0.6B-int4-cw-ov" / "1"
QUERY_INSTRUCTION = (
    "Instruct: Given a software-engineering task, retrieve code passages, symbols, tests, "
    "configuration, and documentation relevant to implementing, debugging, or verifying "
    "the requested change.\nQuery: "
)


class NpuEmbeddingRuntime:
    def __init__(self, model_dir: Path, max_length: int = 2048, cache_dir: Path | None = None) -> None:
        self.model_dir = model_dir
        self.max_length = max_length
        self.cache_path = cache_dir
        self.cache_status = "unknown"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="right", fix_mistral_regex=True)
        core = ov.Core()
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            had_cache = any(path.name != "manifest.json" for path in cache_dir.iterdir())
            core.set_property({"CACHE_DIR": str(cache_dir)})
        else:
            had_cache = False
        model = core.read_model(model_dir / "openvino_model.xml")
        model.reshape({"input_ids": [1, max_length], "attention_mask": [1, max_length]})
        compile_started = time.perf_counter()
        self.compiled = core.compile_model(model, "NPU")
        self.compile_or_import_ms = (time.perf_counter() - compile_started) * 1000
        self.cache_status = "warm-inferred" if had_cache else "cold"
        self.request = self.compiled.create_infer_request()

    def embed(self, texts: list[str], query: bool = False) -> np.ndarray:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Embedding input must contain non-empty text")
        if len(texts) != 1:
            raise ValueError("The NPU runtime is intentionally serialized; call embed once per text")

        values = [QUERY_INSTRUCTION + texts[0] if query else texts[0]]
        encoded = self.tokenizer(
            values,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        outputs = self.request.infer(
            {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            }
        )
        token_embeddings = np.asarray(next(iter(outputs.values())), dtype=np.float32)
        mask = encoded["attention_mask"].astype(np.float32)[..., None]
        pooled = (token_embeddings * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
        norm = np.linalg.norm(pooled, axis=-1, keepdims=True)
        if not np.isfinite(norm).all() or (norm <= 0).any():
            raise RuntimeError("Embedding output has an invalid norm")
        result = pooled / norm
        if result.shape != (1, 1024) or not np.isfinite(result).all():
            raise RuntimeError(f"Expected finite pooled shape (1, 1024), got {result.shape}")
        return result


def run_acceptance(model_dir: Path) -> None:
    cold_started = time.perf_counter()
    runtime = NpuEmbeddingRuntime(model_dir)
    cold_compile_ms = (time.perf_counter() - cold_started) * 1000
    same_a = runtime.embed(["Qdrant retrieves relevant code chunks."])[0]
    same_b = runtime.embed(["Qdrant retrieves relevant code chunks."])[0]
    related = runtime.embed(["Qdrant performs vector search over code chunks."])[0]
    unrelated = runtime.embed(["A lemon tree grows in a sunny garden."])[0]

    shorter_runtime = NpuEmbeddingRuntime(model_dir, max_length=512)
    shorter = shorter_runtime.embed(["Qdrant retrieves relevant code chunks."])[0]

    same_score = float(same_a @ same_b)
    related_score = float(same_a @ related)
    unrelated_score = float(same_a @ unrelated)
    if same_score < 0.999:
        raise AssertionError(f"Identical text cosine similarity too low: {same_score}")
    if related_score <= unrelated_score:
        raise AssertionError(f"Related text did not outrank unrelated text: {related_score} <= {unrelated_score}")
    padding_score = float(same_a @ shorter)
    if padding_score < 0.999:
        raise AssertionError(f"Padding changed the embedding: cosine similarity={padding_score}")

    try:
        runtime.embed([""])
    except ValueError:
        pass
    else:
        raise AssertionError("Empty embedding input was accepted")

    try:
        runtime.embed(["first", "second"])
    except ValueError:
        pass
    else:
        raise AssertionError("Serialized NPU runtime accepted a multi-item batch")

    started = time.perf_counter()
    for _ in range(100):
        runtime.embed(["Sequential NPU embedding stability probe."])
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"NPU embedding acceptance passed: dimension=1024, norm={np.linalg.norm(same_a):.6f}")
    print(f"cosine identical={same_score:.6f}, related={related_score:.6f}, unrelated={unrelated_score:.6f}")
    print(f"padding invariance cosine={padding_score:.6f}")
    print(f"cold compile/load: {cold_compile_ms:.1f} ms")
    print(f"100 sequential requests: {elapsed_ms:.1f} ms, mean={elapsed_ms / 100:.1f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    run_acceptance(args.model_dir)
