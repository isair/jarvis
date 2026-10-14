"""Ollama catalogue comparisons without changing configured or request names."""

from collections.abc import Iterable


def missing_ollama_models(required: Iterable[str], available: Iterable[str]) -> list[str]:
    """Return original required names absent from the available catalogue.

    Only an omitted default tag and a terminal ``:latest`` are equivalent.
    Registry ports, namespaces and explicit model variants remain distinct.
    """
    available_names = {name.removesuffix(":latest") for name in available}
    return [name for name in required if name.removesuffix(":latest") not in available_names]
