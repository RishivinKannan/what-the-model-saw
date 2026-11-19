"""Inputs that compete for space in one model call."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class Chunk:
    """A retrieved piece of a document."""

    id: str
    text: str
    source: str = ""
    score: float | None = None  # retriever's own score, if any


@dataclass(frozen=True)
class ToolOutput:
    """The raw result of a tool call, before trimming."""

    name: str
    content: str
    meta: dict[str, str] = field(default_factory=dict)
