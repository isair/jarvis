# Bonsai 2 as an experimental local chat model

Bonsai 2 offers 27B-class chat with compact weights through Jarvis's existing OpenAI-compatible provider. It is an experimental alternative for people comfortable running a model server, outside automatic recommendations. Keep a small fast model for voice intent and tool routing, and a separate embedding model for memory search.

## Runtime and memory

Use the [Prism llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b10772-e8fd209) and the [publisher's Bonsai 2 GGUF](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf). All Bonsai 2 GGUF formats require the fork's model transformations; successful loading in stock llama.cpp does not establish correct inference. The original Bonsai 27B is a different model. See [publisher backend support](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/BACKEND-SUPPORT.md) for your hardware.

The PQ2_0 language weights occupy about 7.21 GB on disk. Allow extra memory for model state, context, runtime, Whisper, fast and embedding models. Process memory on Apple Silicon is not a portable VRAM requirement. An 8,192-token context and disabled prompt cache limit this example's memory use; longer contexts and caches require more.

Download `Ternary-Bonsai-2-27B-PQ2_0.gguf` from revision `b072e1d3b35a0a630cece372c2127528e0994386`. Its SHA-256 is `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`. Select the `prism-b10772-e8fd209` runtime archive for your platform and follow its installation instructions.

## Serve chat, fast and embeddings locally

Create `models.ini`, replacing every model path with your own absolute path. The fast Gemma and Nomic files must be compatible GGUF files from a trusted distribution of those models. If they are already installed in Ollama, `ollama show gemma4:e4b --modelfile` and `ollama show nomic-embed-text --modelfile` identify their local `FROM` files. Use those paths without renaming or modifying the files. These model IDs are endpoint aliases, not download commands.

```ini
version = 1

[*]
ctx-size = 8192
parallel = 1
cache-ram = 0
jinja = true
reasoning = off
load-on-startup = true

[prism-ml/Ternary-Bonsai-2-27B-gguf]
model = /absolute/path/Ternary-Bonsai-2-27B-PQ2_0.gguf
temp = 0.7
top-p = 0.8
top-k = 20
min-p = 0
presence-penalty = 1.5
repeat-penalty = 1

[google/gemma-4-E4B-it]
model = /absolute/path/gemma-4-E4B-it.gguf

[nomic-ai/nomic-embed-text-v1.5]
model = /absolute/path/nomic-embed-text-v1.5.gguf
embedding = true
pooling = mean
```

Run the fork's `llama-server` from its installed directory:

```bash
./llama-server --models-preset /absolute/path/models.ini --models-max 3 \
  --host 127.0.0.1 --port 18965 --no-webui
```

`reasoning = off` matters for Jarvis's short classification and planning deadlines. Bonsai's default extended reasoning can consume the output budget before producing an answer. The server-level setting is required for this recipe; a client flag alone does not establish that the template disables reasoning.

In Jarvis's Setup Wizard, select **OpenAI-compatible**, connect to `http://127.0.0.1:18965/v1`, and select Bonsai for chat, Gemma for fast and Nomic for embeddings. Review the memory budget and run the connection checks. The warning for an experimental chat model offers **Continue Anyway**; this does not certify the server or downloaded weights. All three models remain loaded in this example, so budget their combined memory. This recipe is qualified on Apple Silicon with PQ2_0; publisher support for another backend does not establish Jarvis qualification on that hardware.

## Suitability

This combination is for experimentation, not an automatic replacement for Gemma or Qwen recommendations. Bonsai can perform native tool calls, but its current Jarvis qualification includes planning deadline failures and strict weather-response failures where correct temperatures are written as words rather than retained as numeric measurements. A passed setup probe does not establish long-term accuracy or latency. Keep the recommended defaults if you need the most thoroughly qualified path.

The configuration keeps inference and memory search on your machine. Downloading weights and runtime binaries needs an internet connection; conversation data is sent only to the local endpoint.
