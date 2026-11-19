"""Token counting.

Uses tiktoken when its encoding can be loaded. If it can't (no network on first use,
for example), falls back to an estimate of ~4 characters per token and says so via
`is_exact()`, so reports never pass an estimate off as an exact count.
"""

from __future__ import annotations

from functools import lru_cache

DEFAULT_ENCODING = "o200k_base"


@lru_cache(maxsize=4)
def _encoder(name: str):
    try:
        import tiktoken

        return tiktoken.get_encoding(name)
    except Exception:  # noqa: BLE001 - any load failure (offline, missing file) means fall back
        return None


def is_exact(encoding: str = DEFAULT_ENCODING) -> bool:
    return _encoder(encoding) is not None


def count_tokens(text: str, encoding: str = DEFAULT_ENCODING) -> int:
    if not text:
        return 0
    enc = _encoder(encoding)
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    return max(1, (len(text) + 3) // 4)
