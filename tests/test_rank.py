from wtms.context.rank import bm25_scores, dedupe, rank
from wtms.types import Chunk


def test_dedupe_drops_near_duplicates_keeps_first():
    a = Chunk("a", "VACUUM reclaims storage occupied by dead tuples in a table.")
    b = Chunk("b", "VACUUM reclaims storage occupied by dead tuples in a table!")
    c = Chunk("c", "CREATE INDEX builds an index on the specified column of a table.")
    kept, dropped = dedupe([a, b, c])
    assert [x.id for x in kept] == ["a", "c"]
    assert [x.id for x in dropped] == ["b"]


def test_dedupe_keeps_distinct_chunks():
    chunks = [Chunk(str(i), f"unique sentence number {i} about topic {i}") for i in range(5)]
    kept, dropped = dedupe(chunks)
    assert len(kept) == 5 and not dropped


def test_rank_puts_matching_chunk_first():
    chunks = [
        Chunk("idx", "CREATE INDEX builds an index on a table column."),
        Chunk("vac", "VACUUM reclaims storage occupied by dead tuples."),
        Chunk("wal", "The write-ahead log records changes before they reach data files."),
    ]
    ranked = rank("How does VACUUM reclaim dead tuples?", chunks)
    assert ranked[0].id == "vac"


def test_bm25_zero_for_no_overlap():
    assert bm25_scores("zebra", [Chunk("x", "postgres table")]) == [0.0]


def test_rank_empty():
    assert rank("anything", []) == []
