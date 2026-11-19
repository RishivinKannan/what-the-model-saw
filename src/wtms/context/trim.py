"""Strategy 2: shrink large tool outputs to the parts that matter.

- JSON: keep the structure, cut long arrays to their first few items (noting how many
  were cut), truncate long strings, and drop keys listed as noise.
- Logs / plain text: keep lines that look like errors or mention the question's words,
  plus the last few lines; collapse the rest into a "[N lines omitted]" marker.

Every cut is marked in the text, so the model knows something was removed.
"""

from __future__ import annotations

import json
import re
from typing import Any

from wtms.tokens import count_tokens
from wtms.types import ToolOutput

_IMPORTANT = re.compile(r"\b(error|fatal|panic|exception|fail(ed|ure)?|denied|timeout|traceback)\b", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9_]{4,}")

DEFAULT_NOISE_KEYS = frozenset({"_links", "etag", "request_id", "trace_id", "headers", "metadata"})


def _trim_json(value: Any, max_items: int, max_str: int, noise: frozenset[str]) -> Any:
    if isinstance(value, dict):
        return {k: _trim_json(v, max_items, max_str, noise) for k, v in value.items() if k not in noise}
    if isinstance(value, list):
        head = [_trim_json(v, max_items, max_str, noise) for v in value[:max_items]]
        if len(value) > max_items:
            head.append(f"[{len(value) - max_items} more items omitted]")
        return head
    if isinstance(value, str) and len(value) > max_str:
        return value[:max_str] + f"…[{len(value) - max_str} chars omitted]"
    return value


def _trim_text(text: str, question: str, tail: int) -> str:
    lines = text.splitlines()
    if len(lines) <= tail * 2:
        return text
    terms = set(_WORD.findall(question.lower()))
    keep = set(range(max(0, len(lines) - tail), len(lines)))
    for i, line in enumerate(lines):
        low = line.lower()
        if _IMPORTANT.search(line) or any(t in low for t in terms):
            keep.add(i)
    out: list[str] = []
    skipped = 0
    for i, line in enumerate(lines):
        if i in keep:
            if skipped:
                out.append(f"[{skipped} lines omitted]")
                skipped = 0
            out.append(line)
        else:
            skipped += 1
    if skipped:
        out.append(f"[{skipped} lines omitted]")
    return "\n".join(out)


def trim_tool_output(
    output: ToolOutput,
    question: str = "",
    *,
    min_tokens: int = 300,
    max_items: int = 5,
    max_str: int = 400,
    tail_lines: int = 20,
    noise_keys: frozenset[str] = DEFAULT_NOISE_KEYS,
) -> ToolOutput:
    """Return a trimmed copy. Outputs under `min_tokens` are returned unchanged."""
    if count_tokens(output.content) < min_tokens:
        return output
    try:
        parsed = json.loads(output.content)
    except (ValueError, TypeError):
        trimmed = _trim_text(output.content, question, tail_lines)
    else:
        trimmed = json.dumps(_trim_json(parsed, max_items, max_str, noise_keys), indent=1)
    return ToolOutput(name=output.name, content=trimmed, meta={**output.meta, "trimmed": "true"})
