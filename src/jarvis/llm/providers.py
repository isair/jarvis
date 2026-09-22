"""Provider catalogue for self-hosted LLM backends.

The catalogue is the construction boundary between configuration and the
provider-agnostic :class:`LLMBackend` interface. Built-in adapters register
their provider name here, while the factory remains independent of concrete
backend classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional

from .backend import LLMBackend


class Provider(str, Enum):
    """Configuration names for supported self-hosted providers."""

    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"


BackendConstructor = Callable[[str, Optional[str]], LLMBackend]


@dataclass(frozen=True)
class ProviderAdapter:
    """Metadata and constructor for one provider adapter."""

    name: str
    constructor: BackendConstructor


_ADAPTERS: Dict[str, ProviderAdapter] = {}


def register_provider(
    name: str,
    constructor: BackendConstructor,
    *,
    replace: bool = False,
) -> None:
    """Register a provider constructor for factory dispatch.

    Registration is intentionally explicit and process-local. It supports
    self-hosted adapters supplied by optional packages without making the core
    package depend on them.
    """

    provider_name = name.strip().lower()
    if not provider_name:
        raise ValueError("provider name must not be empty")
    if not callable(constructor):
        raise TypeError("provider constructor must be callable")
    if provider_name in _ADAPTERS and not replace:
        raise ValueError(f"provider already registered: {provider_name}")
    _ADAPTERS[provider_name] = ProviderAdapter(provider_name, constructor)


def get_provider(name: str) -> Optional[ProviderAdapter]:
    """Return registered provider metadata, or ``None`` when unknown."""

    return _ADAPTERS.get(name.strip().lower())


def available_providers() -> tuple[str, ...]:
    """Return registered provider names in registration order."""

    return tuple(_ADAPTERS)
