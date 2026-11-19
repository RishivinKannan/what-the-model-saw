"""Strategy 1: drop duplicate chunks and rank the rest by relevance to the question.

Relevance here is lexical (BM25), so it runs offline with no API calls. An embedding
scorer can be plugged in later through `Scorer` and compared against this baseline.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence

from wtms.types import Chunk

Scorer = Callable[[str, Sequence[Chunk]], list[float]]

_WORD = re.compile(r"[a-z0-9_]+")
_STOP = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what", "when", "where", "which", "who", "why", "with", "you", "does", "do", "can"]
)


def _words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


def _shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def dedupe(chunks: Sequence[Chunk], threshold: float = 0.8) -> tuple[list[Chunk], list[Chunk]]:
    """Keep the first of any group of near-identical chunks.

    Two chunks are duplicates when the Jaccard similarity of their word 3-grams is at
    least `threshold`. Returns (kept, dropped).
    """
    kept: list[Chunk] = []
    kept_shingles: list[set[tuple[str, ...]]] = []
    dropped: list[Chunk] = []
    for chunk in chunks:
        sh = _shingles(chunk.text)
        is_dup = False
        for other in kept_shingles:
            union = len(sh | other)
            if union and len(sh & other) / union >= threshold:
                is_dup = True
                break
        if is_dup:
            dropped.append(chunk)
        else:
            kept.append(chunk)
            kept_shingles.append(sh)
    return kept, dropped


def bm25_scores(question: str, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """BM25 score of each chunk for the question, using the chunks themselves as the corpus."""
    docs = [_words(c.text) for c in chunks]
    if not docs:
        return []
    n = len(docs)
    avg_len = sum(len(d) for d in docs) / n or 1.0
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    query = _words(question)
    scores: list[float] = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for term in query:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            freq = tf[term]
            s += idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * len(d) / avg_len))
        scores.append(s)
    return scores


def rank(question: str, chunks: Sequence[Chunk], scorer: Scorer = bm25_scores) -> list[Chunk]:
    """Chunks sorted by score, highest first. Ties keep the retriever's original order."""
    scores = scorer(question, chunks)
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])
    return [chunks[i] for i in order]
