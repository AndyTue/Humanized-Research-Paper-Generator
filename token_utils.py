"""
Centralized token counting and truncation utilities.
Uses tiktoken with cl100k_base encoding for accurate LLM token measurement.
"""

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Returns the exact token count of *text* using cl100k_base."""
    if not text:
        return 0
    return len(_ENCODING.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncates *text* to at most *max_tokens* tokens.
    Tries to cut at the last sentence boundary ('. ') within the
    truncated span so the output stays coherent.
    """
    tokens = _ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return text

    truncated = _ENCODING.decode(tokens[:max_tokens])

    # Try to cut at the last sentence boundary for coherence
    last_period = truncated.rfind(". ")
    if last_period > len(truncated) // 2:
        truncated = truncated[: last_period + 1]

    return truncated
