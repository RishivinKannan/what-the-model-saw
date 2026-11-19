"""Fit one model call's context into a token budget, and record every decision.

The report says, for each retrieved chunk, whether it reached the model and if not why
("duplicate" or "over_budget"). That is what lets the eval tell "never retrieved" apart
from "retrieved but cut", which is the point of this project.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from wtms.context.compress import Summariser, compress_history, extractive_summary
from wtms.context.order import best_last, cache_friendly
from wtms.context.rank import Scorer, bm25_scores, dedupe, rank
from wtms.context.trim import trim_tool_output
from wtms.tokens import count_tokens, is_exact
from wtms.types import Chunk, Message, ToolOutput

# Rough per-message overhead in chat formats (role markers etc.).
MESSAGE_OVERHEAD = 4


@dataclass(frozen=True)
class Strategies:
    """Which strategies are on. The eval runs every combination it needs."""

    dedupe: bool = True
    rank: bool = True
    trim_tools: bool = True
    compress_history: bool = True
    cache_order: bool = False
    # When a chunk doesn't fit: True skips it and tries the next (smaller) ones;
    # False stops there, like an app that simply cuts the list off at the budget.
    skip_oversized: bool = True
    keep_recent_turns: int = 4
    scorer: Scorer = bm25_scores
    summariser: Summariser = extractive_summary

    @classmethod
    def none(cls) -> Strategies:
        """Baseline: only the budget is enforced, in retriever order."""
        return cls(
            dedupe=False, rank=False, trim_tools=False, compress_history=False, skip_oversized=False
        )


DEFAULT_STRATEGIES = Strategies()


@dataclass
class Report:
    budget: int
    tokens_in: int = 0  # everything offered, before any strategy
    tokens_out: int = 0  # what was sent
    sections: dict[str, int] = field(default_factory=dict)
    chunks_kept: list[str] = field(default_factory=list)
    chunks_dropped: dict[str, str] = field(default_factory=dict)  # id -> reason
    turns_summarised: int = 0
    turns_dropped: int = 0
    tools_trimmed: list[str] = field(default_factory=list)
    tools_dropped: list[str] = field(default_factory=list)
    exact_token_count: bool = True

    def chunk_fate(self, chunk_id: str) -> str:
        if chunk_id in self.chunks_kept:
            return "sent"
        return self.chunks_dropped.get(chunk_id, "not_retrieved")


@dataclass
class AssembledContext:
    messages: list[Message]
    report: Report


def _tokens(messages: Sequence[Message]) -> int:
    return sum(count_tokens(m.content) + MESSAGE_OVERHEAD for m in messages)


def _doc_message(chunks: Sequence[Chunk]) -> Message:
    body = "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)
    return Message("user", "Reference documents:\n\n" + body)


def _tool_message(t: ToolOutput) -> Message:
    return Message("user", f"Tool result ({t.name}):\n{t.content}")


def assemble(
    question: str,
    *,
    budget: int,
    system: str = "",
    history: Sequence[Message] = (),
    chunks: Sequence[Chunk] = (),
    tool_outputs: Sequence[ToolOutput] = (),
    strategies: Strategies = DEFAULT_STRATEGIES,
) -> AssembledContext:
    """Build the messages for one call within `budget` tokens.

    Priority when space runs out: system prompt and question always go in (a
    ValueError is raised if they alone exceed the budget), then recent history, then
    tool outputs, then document chunks in ranked order. Older history turns are
    dropped before tool outputs, and tool outputs before the question.
    """
    s = strategies
    sys_msgs = [Message("system", system)] if system else []
    q_msg = Message("user", question)
    report = Report(budget=budget, exact_token_count=is_exact())

    report.tokens_in = _tokens(
        [*sys_msgs, *history, *(_tool_message(t) for t in tool_outputs), q_msg]
    ) + (_tokens([_doc_message(chunks)]) if chunks else 0)

    fixed = _tokens([*sys_msgs, q_msg])
    if fixed > budget:
        raise ValueError(f"system prompt + question need {fixed} tokens; budget is {budget}")

    # Chunks: dedupe, then rank.
    pool = list(chunks)
    if s.dedupe:
        pool, dups = dedupe(pool)
        report.chunks_dropped.update({c.id: "duplicate" for c in dups})
    if s.rank:
        pool = rank(question, pool, s.scorer)

    # Tool outputs: trim.
    tools = list(tool_outputs)
    if s.trim_tools:
        trimmed = [trim_tool_output(t, question) for t in tools]
        report.tools_trimmed = [t.name for t, new in zip(tools, trimmed) if new is not t]
        tools = trimmed

    # History: compress old turns.
    hist = list(history)
    if s.compress_history:
        hist, report.turns_summarised = compress_history(hist, s.keep_recent_turns, s.summariser)

    # Fit history + tools into what's left, dropping oldest first.
    remaining = budget - fixed
    tool_msgs = [_tool_message(t) for t in tools]
    dropped_turns: list[Message] = []  # oldest first
    while hist and _tokens(hist) + _tokens(tool_msgs) > remaining:
        dropped_turns.append(hist.pop(0))
    while tool_msgs and _tokens(hist) + _tokens(tool_msgs) > remaining:
        tool_msgs.pop(0)
        report.tools_dropped.append(tools.pop(0).name)
    # Dropping a tool may have freed room: bring turns back, newest first.
    while dropped_turns and _tokens([dropped_turns[-1], *hist]) + _tokens(tool_msgs) <= remaining:
        hist.insert(0, dropped_turns.pop())
    report.turns_dropped = len(dropped_turns)
    remaining -= _tokens(hist) + _tokens(tool_msgs)

    # Chunks fill the rest, best first.
    chosen: list[Chunk] = []
    full = False
    for c in pool:
        if not full and _tokens([_doc_message([*chosen, c])]) <= remaining:
            chosen.append(c)
        else:
            report.chunks_dropped[c.id] = "over_budget"
            full = not s.skip_oversized
    report.chunks_kept = [c.id for c in chosen]

    if s.cache_order:
        ordered = cache_friendly(chosen)
    elif s.rank:
        ordered = best_last(chosen)
    else:
        ordered = chosen

    doc_msgs = [_doc_message(ordered)] if ordered else []
    messages = [*sys_msgs, *doc_msgs, *hist, *tool_msgs, q_msg]

    report.sections = {
        "system": _tokens(sys_msgs),
        "documents": _tokens(doc_msgs),
        "history": _tokens(hist),
        "tools": _tokens(tool_msgs),
        "question": _tokens([q_msg]),
    }
    report.tokens_out = sum(report.sections.values())
    return AssembledContext(messages=messages, report=report)
