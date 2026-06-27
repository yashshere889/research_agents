from __future__ import annotations


def count_tokens(text: str) -> int:
    """Approximate token count for budget tracking when provider metadata is absent."""
    return max(1, len(text) // 4) if text else 0

