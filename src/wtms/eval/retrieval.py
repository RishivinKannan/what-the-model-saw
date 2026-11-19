"""Retrieval metrics: did the chunks that answer the question come back, and how high?"""

from __future__ import annotations

from collections.abc import Collection, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Share of relevant chunk ids found in the top k. 1.0 when nothing is relevant."""
    if not relevant:
        return 1.0
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant))


def reciprocal_rank(retrieved: Sequence[str], relevant: Collection[str]) -> float:
    """1/rank of the first relevant id; 0.0 if none was retrieved."""
    rel = set(relevant)
    for i, cid in enumerate(retrieved, start=1):
        if cid in rel:
            return 1.0 / i
    return 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
