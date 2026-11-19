import pytest

from wtms import Chunk, Message, Strategies, ToolOutput, assemble
from wtms.eval.retrieval import mean, recall_at_k, reciprocal_rank

QUESTION = "How does VACUUM reclaim dead tuples?"


def _chunks():
    filler = [Chunk(f"f{i}", f"Unrelated paragraph {i} about replication slots and standby servers. " * 6)
              for i in range(10)]
    answer = Chunk("vac", "VACUUM reclaims storage occupied by dead tuples so space can be reused.")
    dup = Chunk("vac-copy", "VACUUM reclaims storage occupied by dead tuples so space can be reused.")
    return [*filler, answer, dup]  # the answer arrives last from the retriever


def test_budget_respected_and_question_last():
    ctx = assemble(QUESTION, budget=300, system="Answer from the documents.", chunks=_chunks())
    assert ctx.report.tokens_out <= 300
    assert ctx.messages[-1].content == QUESTION
    assert ctx.messages[0].role == "system"


def test_ranking_saves_the_answer_under_tight_budget():
    baseline = assemble(QUESTION, budget=300, chunks=_chunks(), strategies=Strategies.none())
    tuned = assemble(QUESTION, budget=300, chunks=_chunks())
    assert baseline.report.chunk_fate("vac") == "over_budget"
    assert tuned.report.chunk_fate("vac") == "sent"
    assert tuned.report.chunk_fate("vac-copy") == "duplicate"
    assert tuned.report.chunk_fate("never-seen") == "not_retrieved"


def test_best_chunk_sits_next_to_question():
    ctx = assemble(QUESTION, budget=2000, chunks=_chunks())
    docs = next(m.content for m in ctx.messages if m.content.startswith("Reference documents"))
    assert docs.rstrip().endswith("so space can be reused.")


def test_cache_order_is_stable_across_rankings():
    a = assemble("VACUUM dead tuples", budget=5000, chunks=_chunks(),
                 strategies=Strategies(cache_order=True))
    b = assemble("replication standby", budget=5000, chunks=_chunks(),
                 strategies=Strategies(cache_order=True))
    assert a.messages[0].content == b.messages[0].content  # same documents, same bytes


def test_history_dropped_oldest_first_when_over_budget():
    hist = [Message("user", f"turn {i} " + "words " * 40) for i in range(12)]
    ctx = assemble(QUESTION, budget=200, history=hist,
                   strategies=Strategies(compress_history=False))
    assert ctx.report.turns_dropped > 0
    sent = [m.content for m in ctx.messages if m.content.startswith("turn ")]
    assert sent == [m.content for m in hist[-len(sent):]]


def test_huge_tool_output_does_not_wipe_history():
    hist = [Message("user", f"turn {i}") for i in range(4)]
    huge = ToolOutput("logs", "x " * 5000)
    ctx = assemble(QUESTION, budget=300, history=hist, tool_outputs=[huge],
                   strategies=Strategies.none())
    assert ctx.report.tools_dropped == ["logs"]
    assert ctx.report.turns_dropped == 0
    assert [m.content for m in ctx.messages if m.content.startswith("turn ")] == [
        m.content for m in hist
    ]


def test_tool_output_trimmed():
    big = ToolOutput("logs", "\n".join(f"INFO line {i}" for i in range(400)))
    ctx = assemble(QUESTION, budget=4000, tool_outputs=[big])
    assert ctx.report.tools_trimmed == ["logs"]
    assert ctx.report.sections["tools"] < ctx.report.tokens_in


def test_too_small_budget_raises():
    with pytest.raises(ValueError):
        assemble("q " * 500, budget=50)


def test_retrieval_metrics():
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, k=2) == 0.5
    assert recall_at_k(["a"], set(), k=1) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a"], {"z"}) == 0.0
    assert mean([]) == 0.0
