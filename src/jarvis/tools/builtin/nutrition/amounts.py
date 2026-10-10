"""Shared validation for recorded and retrieved nutrition estimates."""
import math
from typing import Any, Optional


def normalise_amount(value: Any) -> Optional[float]:
    """Return a finite non-negative amount, preserving missing estimates."""
    if value is None or isinstance(value, bool):
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return amount if math.isfinite(amount) and amount >= 0 else None
