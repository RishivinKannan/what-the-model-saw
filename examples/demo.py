"""Offline demo: the same inputs, with and without strategies, under one budget.

Run:  python examples/demo.py
The chunks below are made-up placeholders, not real PostgreSQL docs text. They only
show the mechanics; real results come from the eval over the real docs.
"""

from wtms import Chunk, Message, Strategies, ToolOutput, assemble

question = "Why is my table still large after deleting rows?"

chunks = [
    *(Chunk(f"filler-{i}", f"Placeholder paragraph {i} about an unrelated topic. " * 8) for i in range(6)),
    Chunk("answer", "Placeholder: deleted rows leave dead tuples; space is reclaimed by vacuuming."),
    Chunk("answer-dup", "Placeholder: deleted rows leave dead tuples; space is reclaimed by vacuuming."),
]
history = [Message("user" if i % 2 == 0 else "assistant", f"Earlier turn {i}. Details follow.")
           for i in range(8)]
logs = ToolOutput("server_logs", "\n".join(
    "ERROR: could not extend file" if i == 42 else f"LOG: checkpoint {i} complete" for i in range(300)
))

for label, strategies in [("baseline", Strategies.none()), ("all strategies", Strategies())]:
    ctx = assemble(question, budget=600, system="Answer using the documents.",
                   history=history, chunks=chunks, tool_outputs=[logs], strategies=strategies)
    r = ctx.report
    print(f"\n== {label} ==")
    print(f"tokens offered {r.tokens_in}, sent {r.tokens_out} / budget {r.budget}"
          + ("" if r.exact_token_count else "  (estimated: tiktoken encoding not available)"))
    print("sections:", r.sections)
    print("answer chunk:", r.chunk_fate("answer"), "| duplicate:", r.chunk_fate("answer-dup"))
    print(f"turns summarised {r.turns_summarised}, dropped {r.turns_dropped}; "
          f"tools trimmed {r.tools_trimmed}, dropped {r.tools_dropped}")
