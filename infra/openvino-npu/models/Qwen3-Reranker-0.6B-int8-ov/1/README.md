---
license: apache-2.0
base_model:
    - Qwen/Qwen3-Reranker-0.6B
library_name: transformers
pipeline_tag: text-ranking
---

# Qwen3-Reranker-0.6B-int8-ov

- Model creator: [Qwen](https://huggingface.co/Qwen)
- Original model: [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)

## Description

This is [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) model converted to the [OpenVINO™ IR](https://docs.openvino.ai/2026/documentation/openvino-ir-format.html) (Intermediate Representation) format with weights compressed to INT8 by [NNCF](https://github.com/openvinotoolkit/nncf).

## Quantization Parameters

Weight compression was performed using `nncf.compress_weights` with the following parameters:

- mode: **INT8_ASYM**
- group_size: **-1**

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
from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM


model_id = "OpenVINO/Qwen3-Reranker-0.6B-int8-ov"

model = OVModelForCausalLM.from_pretrained(model_id, use_cache=False, export=False)


def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction, query=query, doc=doc)
    return output


def process_inputs(pairs):
    inputs = tokenizer(
        pairs, padding=False, truncation="longest_first", return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs


def compute_logits(inputs, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores


tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
max_length = 8192

prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

task = "Given a web search query, retrieve relevant passages that answer the query"

queries = [
    "What is the capital of China?",
    "Explain gravity",
]

documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

pairs = [format_instruction(task, query, doc) for query, doc in zip(queries, documents)]

# Tokenize the input texts
inputs = process_inputs(pairs)
scores = compute_logits(inputs)

print("scores: ", scores)
```

For more examples and possible optimizations, refer to the [Inference with Optimum Intel](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-optimum-intel.html).

## Limitations

Check the original [model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) for limitations.

## Legal information

The original model is distributed under [Apache License Version 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) license. More details can be found in [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B).

## Disclaimer

Intel is committed to respecting human rights and avoiding causing or contributing to adverse impacts on human rights. See [Intel’s Global Human Rights Principles](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/policy-human-rights.pdf). Intel’s products and software are intended only to be used in applications that do not cause or contribute to adverse impacts on human rights.
