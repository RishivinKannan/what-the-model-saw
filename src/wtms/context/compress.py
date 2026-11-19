"""Strategy 3: keep recent turns verbatim, replace older turns with a summary.

The summariser is pluggable. `extractive_summary` needs no model: it keeps the first
sentence of each old turn, which makes a cheap, deterministic baseline. `llm_summary`
asks a model for a summary and is what most real agents do; the eval compares both.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from wtms.types import Message

Summariser = Callable[[Sequence[Message]], str]

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


def extractive_summary(turns: Sequence[Message], max_chars: int = 160) -> str:
    lines = []
    for m in turns:
        first = _SENTENCE_END.split(m.content.strip(), maxsplit=1)[0]
        if len(first) > max_chars:
            first = first[:max_chars] + "…"
        lines.append(f"- {m.role}: {first}")
    return "\n".join(lines)


def llm_summary(model: str = "gpt-5-mini") -> Summariser:
    """Summariser backed by the OpenAI API. Needs `pip install wtms[llm]` and OPENAI_API_KEY."""

    def summarise(turns: Sequence[Message]) -> str:
        from openai import OpenAI

        transcript = "\n".join(f"{m.role}: {m.content}" for m in turns)
        resp = OpenAI().chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Summarise this conversation for an assistant that will continue it. "
                    "Keep facts, decisions, names, numbers and open questions. Drop pleasantries. "
                    "Use short bullet points.",
                },
                {"role": "user", "content": transcript},
            ],
        )
        return resp.choices[0].message.content or ""

    return summarise


def compress_history(
    history: Sequence[Message],
    keep_recent: int = 4,
    summariser: Summariser = extractive_summary,
) -> tuple[list[Message], int]:
    """Return (new_history, number_of_turns_summarised).

    System messages are never summarised. If there are no more than `keep_recent` other
    turns, history is returned unchanged.
    """
    system = [m for m in history if m.role == "system"]
    turns = [m for m in history if m.role != "system"]
    if len(turns) <= keep_recent:
        return list(history), 0
    old, recent = turns[:-keep_recent], turns[-keep_recent:]
    summary = Message("user", "Summary of the earlier conversation:\n" + summariser(old))
    return [*system, summary, *recent], len(old)
