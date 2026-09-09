"""Smallest bounded native NPU gate for Qwen3-Reranker.

The prompt and score extraction follow the official OpenVINO model README:
the final-token logits for the tokenizer's ``yes`` and ``no`` tokens are
converted to a two-class log-softmax probability. This deliberately starts at
batch 1 x sequence 512 after the NPU driver update.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import openvino as ov
from transformers import AutoTokenizer


DEFAULT_MODEL = Path(__file__).parent / "models" / "Qwen3-Reranker-0.6B-int8-ov" / "1"
MAX_LENGTH = 512
INSTRUCTION = "Given a software-engineering task, retrieve relevant code passages."
PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class NpuRerankerRuntime:
    def __init__(self, model_dir: Path, max_length: int = MAX_LENGTH, cache_dir: Path | None = None) -> None:
        self.max_length = max_length
        self.cache_path = cache_dir
        self.cache_status = "unknown"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left", fix_mistral_regex=True)
        self.false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.true_id = self.tokenizer.convert_tokens_to_ids("yes")
        if self.false_id is None or self.true_id is None:
            raise RuntimeError("The reranker tokenizer does not define yes/no tokens")
        self.prefix_tokens = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(SUFFIX, add_special_tokens=False)

        core = ov.Core()
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            had_cache = any(path.name != "manifest.json" for path in cache_dir.iterdir())
            core.set_property({"CACHE_DIR": str(cache_dir)})
        else:
            had_cache = False
        model = core.read_model(model_dir / "openvino_model.xml")
        model.reshape(
            {
                "input_ids": [1, max_length],
                "attention_mask": [1, max_length],
                "position_ids": [1, max_length],
            }
        )
        compile_started = time.perf_counter()
        self.compiled = core.compile_model(model, "NPU")
        self.compile_or_import_ms = (time.perf_counter() - compile_started) * 1000
        self.cache_status = "warm-inferred" if had_cache else "cold"
        self.request = self.compiled.create_infer_request()

    def _format(self, query: str, document: str) -> str:
        return (
            f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"
        )

    def score(self, query: str, document: str) -> float:
        if not query.strip() or not document.strip():
            raise ValueError("Reranker query and document must be non-empty")
        raw = self.tokenizer(
            [self._format(query, document)],
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        token_ids = self.prefix_tokens + raw["input_ids"][0] + self.suffix_tokens
        encoded = self.tokenizer.pad(
            {"input_ids": [token_ids]},
            padding="max_length",
            max_length=self.max_length,
            return_tensors="np",
        )
        attention_mask = encoded["attention_mask"].astype(np.int64)
        position_ids = np.maximum(np.cumsum(attention_mask, axis=1) - 1, 0).astype(np.int64)
        outputs = self.request.infer(
            {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )
        logits = np.asarray(next(iter(outputs.values())), dtype=np.float32)
        last_logits = logits[:, -1, :]
        selected = last_logits[:, [self.false_id, self.true_id]]
        selected -= selected.max(axis=1, keepdims=True)
        probabilities = np.exp(selected)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        score = float(probabilities[0, 1])
        if not np.isfinite(score):
            raise RuntimeError("Reranker produced a non-finite score")
        return score


def run_acceptance(model_dir: Path, max_length: int = MAX_LENGTH) -> None:
    started = time.perf_counter()
    runtime = NpuRerankerRuntime(model_dir, max_length=max_length)
    compile_ms = (time.perf_counter() - started) * 1000
    query = "Find the Qdrant embedding configuration."
    positive = runtime.score(query, "The code index configures Qdrant with a 1024-dimensional cosine vector.")
    negative = runtime.score(query, "The application renders a settings panel with a dark theme.")
    if not positive > negative:
        raise AssertionError(f"Positive reranker pair did not outrank negative: {positive} <= {negative}")
    print(f"NPU reranker acceptance passed: shape=1x{runtime.max_length}, compile/load={compile_ms:.1f} ms")
    print(f"positive={positive:.6f}, negative={negative:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()
    run_acceptance(args.model_dir, args.max_length)
