from __future__ import annotations
import requests


def get_embedding(text: str, base_url: str, model: str, timeout_sec: float = 15.0) -> list[float] | None:
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json()
        vec = data.get("embedding")
        if isinstance(vec, list):
            return [float(x) for x in vec]
    except Exception:
        return None
    return None
