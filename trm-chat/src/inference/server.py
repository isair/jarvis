"""TRM-Chat Inference Server - Ollama-compatible API.

This server mimics Ollama's /api/chat endpoint, allowing seamless
integration with Jarvis without any code changes.

Usage:
    uvicorn trm_chat.inference.server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import time
from typing import Optional, List, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import mlx.core as mx

# Will be initialized on startup
_model = None
_tokenizer = None
_config = None


class Message(BaseModel):
    """Chat message."""
    role: str  # "system", "user", "assistant"
    content: str
    tool_calls: Optional[List[Dict]] = None


class ChatRequest(BaseModel):
    """Ollama-compatible chat request."""
    model: str
    messages: List[Message]
    stream: bool = False
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    format: Optional[str] = None  # "json" for JSON mode


class ChatResponse(BaseModel):
    """Ollama-compatible chat response."""
    model: str
    created_at: str
    message: Message
    done: bool = True
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class StreamChunk(BaseModel):
    """Streaming response chunk."""
    model: str
    created_at: str
    message: Message
    done: bool = False


def get_model_path() -> str:
    """Get model path from environment or default."""
    import os
    return os.environ.get("TRM_MODEL_PATH", "./models/qwen2.5-3b")


def get_trm_heads_path() -> Optional[str]:
    """Get TRM heads path from environment."""
    import os
    return os.environ.get("TRM_HEADS_PATH")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global _model, _tokenizer, _config

    print("Loading TRM-Chat model...")

    try:
        from ..model import TRMChatConfig
        from ..model.trm_wrapper import load_trm_chat

        model_path = get_model_path()
        trm_heads_path = get_trm_heads_path()

        # Load config
        _config = TRMChatConfig(base_model_path=model_path)

        # Load model
        _model = load_trm_chat(
            base_model_path=model_path,
            trm_heads_path=trm_heads_path,
            config=_config,
            use_adaptive_halt=True
        )
        _tokenizer = _model.tokenizer

        print(f"Model loaded from {model_path}")
        if trm_heads_path:
            print(f"TRM heads loaded from {trm_heads_path}")

    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Server will start but /api/chat will return errors")

    yield

    # Cleanup
    _model = None
    _tokenizer = None
    print("Model unloaded")


app = FastAPI(
    title="TRM-Chat Server",
    description="Ollama-compatible API for TRM-Chat model",
    version="0.1.0",
    lifespan=lifespan
)


def format_chat_prompt(messages: List[Message], tokenizer) -> str:
    """Format messages into a prompt string.

    This follows the chat template of the base model.
    """
    # Try to use the tokenizer's chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        # Convert to dict format
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # Fallback: Simple format
    prompt_parts = []
    for msg in messages:
        if msg.role == "system":
            prompt_parts.append(f"<|system|>\n{msg.content}\n")
        elif msg.role == "user":
            prompt_parts.append(f"<|user|>\n{msg.content}\n")
        elif msg.role == "assistant":
            prompt_parts.append(f"<|assistant|>\n{msg.content}\n")

    # Add generation prompt
    prompt_parts.append("<|assistant|>\n")

    return "".join(prompt_parts)


def extract_profile_from_messages(messages: List[Message]) -> str:
    """Extract profile from system message if present."""
    for msg in messages:
        if msg.role == "system":
            content = msg.content.lower()
            if "developer" in content or "code" in content or "technical" in content:
                return "developer"
            elif "business" in content or "product" in content:
                return "business"
    return "life"  # Default


async def generate_response(
    messages: List[Message],
    options: Dict[str, Any],
    stream: bool = False
) -> AsyncGenerator[str, None]:
    """Generate response from model.

    Args:
        messages: Chat messages
        options: Generation options
        stream: Whether to stream tokens

    Yields:
        Generated tokens (if streaming) or full response
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Format prompt
    prompt = format_chat_prompt(messages, _tokenizer)

    # Get options
    max_tokens = options.get("num_ctx", 512)
    temperature = options.get("temperature", 0.7)

    # Get profile
    profile = extract_profile_from_messages(messages)

    # Tokenize
    tokens = _tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    profile_id = _config.profile_names.index(profile) if profile in _config.profile_names else 2
    profile_ids = mx.array([profile_id])

    # Generate
    generated_tokens = []
    generated_text = ""

    for i in range(max_tokens):
        # Forward pass
        outputs = _model.forward(input_ids, profile_ids=profile_ids)
        logits = outputs["logits"]

        # Sample next token
        next_token_logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = mx.softmax(next_token_logits, axis=-1)

        # Top-p sampling
        top_p = options.get("top_p", 0.9)
        if top_p < 1.0:
            sorted_probs = mx.sort(probs, axis=-1)[:, ::-1]
            cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
            mask = cumulative_probs <= top_p
            # Keep at least one token
            mask = mx.concatenate([mx.ones((1, 1)), mask[:, :-1]], axis=-1)
            probs = mx.where(mask, probs, mx.zeros_like(probs))
            probs = probs / mx.sum(probs, axis=-1, keepdims=True)

        # Sample
        if temperature > 0:
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
        else:
            next_token = mx.argmax(probs, axis=-1)

        next_token = next_token.item()

        # Check for EOS
        eos_id = getattr(_tokenizer, 'eos_token_id', None)
        if eos_id is not None and next_token == eos_id:
            break

        generated_tokens.append(next_token)

        # Decode token
        token_text = _tokenizer.decode([next_token])
        generated_text += token_text

        if stream:
            yield token_text

        # Update input for next iteration
        input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

        # Evaluate to prevent memory buildup
        mx.eval(input_ids)

    if not stream:
        yield generated_text


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Ollama-compatible chat endpoint."""
    start_time = time.time()

    if request.stream:
        # Streaming response
        async def stream_generator():
            async for token in generate_response(
                request.messages,
                request.options or {},
                stream=True
            ):
                chunk = StreamChunk(
                    model=request.model,
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    message=Message(role="assistant", content=token),
                    done=False
                )
                yield json.dumps(chunk.model_dump()) + "\n"

            # Final chunk
            final = StreamChunk(
                model=request.model,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                message=Message(role="assistant", content=""),
                done=True
            )
            yield json.dumps(final.model_dump()) + "\n"

        return StreamingResponse(
            stream_generator(),
            media_type="application/x-ndjson"
        )

    else:
        # Non-streaming response
        full_response = ""
        async for text in generate_response(
            request.messages,
            request.options or {},
            stream=False
        ):
            full_response = text  # Last yield is full response

        end_time = time.time()
        duration_ns = int((end_time - start_time) * 1e9)

        return ChatResponse(
            model=request.model,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            message=Message(role="assistant", content=full_response),
            done=True,
            total_duration=duration_ns,
            eval_duration=duration_ns
        )


class EmbeddingRequest(BaseModel):
    """Ollama-compatible embedding request."""
    model: str
    prompt: str


class EmbeddingResponse(BaseModel):
    """Ollama-compatible embedding response."""
    embedding: List[float]


@app.post("/api/embeddings")
async def embeddings(request: EmbeddingRequest):
    """Generate embeddings using base model hidden states."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Tokenize
    tokens = _tokenizer.encode(request.prompt)
    input_ids = mx.array([tokens])

    # Get hidden states from base model (without TRM heads)
    inner_model = _model.base_model.model
    hidden_states = inner_model(input_ids)

    # Mean pooling over sequence length
    embedding = mx.mean(hidden_states[0], axis=0)

    # Normalize
    embedding = embedding / mx.sqrt(mx.sum(embedding ** 2) + 1e-8)

    # Convert to list
    mx.eval(embedding)
    embedding_list = embedding.tolist()

    return EmbeddingResponse(embedding=embedding_list)


@app.get("/api/tags")
async def list_models():
    """List available models (Ollama compatibility)."""
    return {
        "models": [
            {
                "name": "trm-chat:latest",
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "size": 0,
                "digest": "trm-chat",
                "details": {
                    "format": "mlx",
                    "family": "trm-chat",
                    "parameter_size": "3.8B"
                }
            },
            {
                "name": "nomic-embed-text:latest",
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "size": 0,
                "digest": "nomic-embed-text",
                "details": {
                    "format": "mlx",
                    "family": "nomic",
                    "parameter_size": "137M"
                }
            }
        ]
    }


@app.get("/api/version")
async def version():
    """Get server version."""
    return {"version": "0.1.0"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": _model is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
