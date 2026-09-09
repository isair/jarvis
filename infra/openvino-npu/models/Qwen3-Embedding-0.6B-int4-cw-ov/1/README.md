---
license: apache-2.0
base_model:
    - Qwen/Qwen3-Embedding-0.6B
library_name: transformers
pipeline_tag: feature-extraction
---

# Qwen3-Embedding-0.6B-int4-cw-ov

- Model creator: [Qwen](https://huggingface.co/Qwen)
- Original model: [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

## Description

This is [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) model converted to the [OpenVINO™ IR](https://docs.openvino.ai/2026/documentation/openvino-ir-format.html) (Intermediate Representation) format with weights compressed to INT4 by [NNCF](https://github.com/openvinotoolkit/nncf).

> The model is optimized for inference on NPU using these [instructions.](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html#export-an-llm-model-via-hugging-face-optimum-intel)

## Quantization Parameters

Weight compression was performed using `nncf.compress_weights` with the following parameters:

- mode: **INT4_SYM**
- ratio: **1.0**

For more information on quantization, check the [OpenVINO model optimization guide](https://docs.openvino.ai/2026/openvino-workflow/model-optimization-guide/weight-compression.html).

## Compatibility

The provided OpenVINO™ IR model is compatible with:

- OpenVINO version 2026.0.0 and higher
- Optimum Intel 1.27.0 and higher

## Running Model Inference with [Optimum Intel](https://huggingface.co/docs/optimum/intel/index)

1. Install packages required for using [Optimum Intel](https://huggingface.co/docs/optimum/intel/index) integration with the OpenVINO backend:

```
pip install "git+https://github.com/huggingface/optimum-intel.git" "torch==2.8" --extra-index-url https://download.pytorch.org/whl/cpu
```

2. Run model inference:

```
import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer

from optimum.intel import OVModelForFeatureExtraction

device = "NPU"
model_id = "OpenVINO/Qwen3-Embedding-0.6B-int4-cw-ov"

model = OVModelForFeatureExtraction.from_pretrained(model_id, export=False, device=device)


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


# Each query must come with a one-sentence instruction that describes the task
task = "Given a web search query, retrieve relevant passages that answer the query"

queries = [get_detailed_instruct(task, "What is the capital of China?"), get_detailed_instruct(task, "Explain gravity")]
# No need to add instruction for retrieval documents
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]
input_texts = queries + documents

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

max_length = 8192

# Tokenize the input texts
batch_dict = tokenizer(
    input_texts,
    padding=True,
    truncation=True,
    max_length=max_length,
    return_tensors="pt",
)
batch_dict.to(model.device)
outputs = model(**batch_dict)
embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
# normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = embeddings[:2] @ embeddings[2:].T
print(scores.tolist())
```

For more examples and possible optimizations, refer to the [Inference with Optimum Intel](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-optimum-intel.html).

## Limitations

Check the original [model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) for limitations.

## Legal information

The original model is distributed under [Apache License Version 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) license. More details can be found in [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).

## Disclaimer

Intel is committed to respecting human rights and avoiding causing or contributing to adverse impacts on human rights. See [Intel’s Global Human Rights Principles](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/policy-human-rights.pdf). Intel’s products and software are intended only to be used in applications that do not cause or contribute to adverse impacts on human rights.
