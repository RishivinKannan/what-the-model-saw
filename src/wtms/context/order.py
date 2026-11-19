"""Strategy 4: order the prompt so the stable parts form a shared prefix.

Providers that cache prompts (OpenAI does this automatically for long prompts) reuse
work only for an identical *prefix*. So the parts that rarely change go first and the
parts that change every call go last:

    system prompt → documents (sorted by id) → history → tool outputs → question

Sorting documents by id rather than by score means the same set of documents always
produces the same bytes, even if their scores shift. The cost: the most relevant chunk
may no longer sit next to the question. The eval measures whether that hurts accuracy.

With ordering off, the default is relevance order with the best chunk placed last,
right before the question, where models tend to use it most reliably.
"""

from __future__ import annotations

from collections.abc import Sequence

from wtms.types import Chunk


def cache_friendly(chunks: Sequence[Chunk]) -> list[Chunk]:
    return sorted(chunks, key=lambda c: c.id)


def best_last(ranked: Sequence[Chunk]) -> list[Chunk]:
    """`ranked` is highest-first; return it lowest-first so the best chunk ends nearest the question."""
    return list(reversed(ranked))
