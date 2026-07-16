"""Example plugin tools for Jarvis.

Drop this file (or your own) into ``~/.jarvis/plugins/`` and restart the
daemon. Every ``@tool``-decorated function is registered automatically and
becomes available to the LLM like a built-in tool.

Example: ``"roll a die"`` or ``"roll a 20-sided die"``.
"""

from jarvis.tools.plugin import tool


@tool()
def roll_dice(sides: int = 6) -> str:
    """Roll a die with the given number of sides and return the result.

    Use when the user wants a random number, a dice roll, or a simple
    chance outcome (e.g. "flip a coin", "roll a d20").
    """
    import random

    if sides < 2:
        sides = 2
    return f"You rolled a {random.randint(1, sides)} (out of {sides})."


@tool()
def flip_coin() -> str:
    """Flip a coin and return heads or tails."""
    import random

    return "Heads." if random.random() < 0.5 else "Tails."
