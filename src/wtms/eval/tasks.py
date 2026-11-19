"""The task set: questions over the PostgreSQL docs with known answers.

One JSON object per line in data/tasks/*.jsonl:

    {"id": "pg-001",
     "question": "...",
     "answer": "short reference answer",
     "relevant_chunks": ["chunk-id", ...],   # ids that contain the answer
     "kind": "lookup" | "multi_hop" | "tool" | "long_history",
     "source_url": "https://www.postgresql.org/docs/..."}

Write answers from the docs page itself and keep `source_url`, so anyone can check them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

KINDS = {"lookup", "multi_hop", "tool", "long_history"}


@dataclass(frozen=True)
class Task:
    id: str
    question: str
    answer: str
    relevant_chunks: tuple[str, ...] = ()
    kind: str = "lookup"
    source_url: str = ""
    extra: dict = field(default_factory=dict)


def load_tasks(path: str | Path) -> list[Task]:
    tasks: list[Task] = []
    seen: set[str] = set()
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("id", "question", "answer"):
            if not row.get(key):
                raise ValueError(f"{path}:{lineno}: missing '{key}'")
        if row["id"] in seen:
            raise ValueError(f"{path}:{lineno}: duplicate id {row['id']}")
        kind = row.get("kind", "lookup")
        if kind not in KINDS:
            raise ValueError(f"{path}:{lineno}: unknown kind '{kind}'")
        seen.add(row["id"])
        known = {"id", "question", "answer", "relevant_chunks", "kind", "source_url"}
        tasks.append(
            Task(
                id=row["id"],
                question=row["question"],
                answer=row["answer"],
                relevant_chunks=tuple(row.get("relevant_chunks", [])),
                kind=kind,
                source_url=row.get("source_url", ""),
                extra={k: v for k, v in row.items() if k not in known},
            )
        )
    return tasks
