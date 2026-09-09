# OpenVINO NPU embedding runtime

This stack runs the English `OpenVINO/Qwen3-Embedding-0.6B-int4-cw-ov` embedding model and `OpenVINO/Qwen3-Reranker-0.6B-int8-ov` through OpenVINO Model Server (OVMS) and Qdrant.

The model is intentionally separate from the primary coding LLM. The NPU is used for indexing and query embeddings while the RTX 4090 remains available for generation.

## Model layout

Convert/export the model into this layout:

```text
infra/openvino-npu/models/Qwen3-Embedding-0.6B-int4-cw-ov/1/
  openvino_model.xml
  openvino_model.bin
```

The embedding model exposes 1,024-dimensional vectors. The reranker is loaded resident but is not executed in the first phase.

Before starting OVMS, validate the model contract directly on the physical NPU:

```powershell
.\infra\openvino-npu\verify-native-embedding.ps1
```

The native acceptance runner statically bounds the sequence to 2,048 tokens, performs attention-mask mean pooling, L2-normalizes the result, rejects empty/invalid outputs, checks identical/related/unrelated cosine scores, and runs 100 serialized requests.

## Start

From the repository root:

```powershell
.\infra\openvino-npu\start-local-infra.ps1
```

The script starts OVMS and Qdrant, waits for readiness, sends a real embedding request to OVMS, checks Qdrant, and prints a compact status block. NPU execution is required; the script does not silently relabel CPU execution as NPU execution.

The native launcher uses the repository-local `.venv-openvino-npu` by default. On first start it creates the venv and installs the pinned dependencies from [`requirements-openvino-npu.txt`](requirements-openvino-npu.txt). This avoids accidentally probing a system or pyenv interpreter that does not contain OpenVINO. Use `-Python` only when intentionally selecting an already provisioned interpreter; that override is validated and is never silently replaced.

Useful commands:

```powershell
.\infra\openvino-npu\start-local-infra.ps1 -SkipModelProbe
.\infra\openvino-npu\stop-local-infra.ps1
docker compose -f .\infra\openvino-npu\docker-compose.yml logs -f ovms
```

## Kelvin Clyne provider settings

The existing [`OpenAICompatibleEmbedder`](../../src/services/code-index/embedders/openai-compatible.ts:35) uses the OpenAI SDK with `baseURL + /embeddings` for a base URL. Configure:

```text
Provider: openai-compatible
Base URL: http://127.0.0.1:8000/v3
Model:    Qwen3-Embedding-0.6B-int4-cw-ov
Qdrant:  http://127.0.0.1:6333
```

OVMS does not currently support OpenAI’s optional `dimensions` request property. Kelvin’s existing OpenAI-compatible embedding request does not send that property; keep it omitted for this provider.

Optimum Intel is not needed for this OVMS deployment because both downloaded artifacts are already OpenVINO IR. It is useful only for direct Python inference or re-export workflows. OpenVINO 2026.0+ is required.

The embedding runtime is deliberately bounded:

```text
NPU max active inferences: 1
Scheduler embedding batch: 8
Endpoint batch limit: 500 inputs/request
Maximum search results: 50 candidates
Maximum injected results: 8 chunks
Context injection budget: 8,000 tokens
```

The 50-result search limit is not the injection limit. Retrieve broadly, rerank or score candidates, then inject only the best eight chunks within the 8,000-token budget.

## Optional reranking phase

Reranking is not enabled by this first vertical slice. The future provider contract can target an OVMS Cohere-compatible endpoint such as:

```text
Endpoint: http://127.0.0.1:8000/v3
Model: Qwen3-Reranker-0.6B-int8-ov
Candidates: 50
Results: 8
Context budget: 8,000 tokens
```

Keep reranking behind a provider abstraction; do not couple Qdrant or the code index to one Intel model. The reranker requires a real compile and inference benchmark on the target NPU before enabling it. Embedding and reranking are resident together but scheduled with one active NPU workload; interactive reranking has priority over query embedding, which has priority over background indexing.
