"""Probe local model endpoints for the evaluation runner."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Callable

import requests


@dataclass(frozen=True)
class ProviderStatus:
    provider: str | None
    available: bool
    reason: str
    model: str
    base_url: str


def probe_provider(
    base_url: str,
    model: str,
    *,
    get: Callable = requests.get,
) -> ProviderStatus:
    """Check model availability without starting or downloading a model."""
    base = base_url.rstrip("/")
    ollama_base = base[:-3] if base.endswith("/v1") else base
    openai_base = base if base.endswith("/v1") else f"{base}/v1"
    probes = (
        ("ollama", f"{ollama_base}/api/tags", "models", "name"),
        ("openai_compatible", f"{openai_base}/models", "data", "id"),
    )
    detected: str | None = None
    for provider, url, collection, key in probes:
        try:
            response = get(url, timeout=2)
            if response.status_code != 200:
                continue
            data = response.json()
            detected = provider
            names = [entry.get(key) for entry in data.get(collection, [])]
            if model in names:
                return ProviderStatus(provider, True, "available", model, base_url)
        except (requests.RequestException, ConnectionError, TimeoutError, ValueError):
            continue
    return ProviderStatus(
        detected,
        False,
        "model_not_found" if detected else "endpoint_unreachable",
        model,
        base_url,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe a local eval model")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    result = probe_provider(args.base_url, args.model)
    print(json.dumps(asdict(result)))
    return 0 if result.available else 2


if __name__ == "__main__":
    raise SystemExit(main())
