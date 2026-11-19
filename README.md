# what-the-model-saw (`wtms`)

**Does the model see the right information when it answers?**

An LLM agent fills every call with a system prompt, chat history, retrieved documents and
tool outputs. When that exceeds the budget, something gets cut, and answers go wrong in
ways that are hard to trace. This project:

1. **Fits context into a token budget** with four strategies you can switch on and off:
   - **Dedupe + rank** retrieved chunks by relevance to the question
   - **Trim tool outputs** (big JSON and logs → the parts that matter, with cuts marked)
   - **Compress history** (old turns summarised, recent turns kept verbatim)
   - **Cache-friendly ordering** (stable parts first, so provider prompt caching can reuse them)
2. **Records the fate of every chunk**: `sent`, `duplicate`, `over_budget` or `not_retrieved`.
   That separates *"the answer was never retrieved"* from *"it was retrieved, then cut"*.
3. **Measures each strategy** on a fixed question set over the PostgreSQL documentation:
   tokens, cost, latency, retrieval recall and answer accuracy.

Long-term **memory quality** (what an agent saves, recalls and forgets across sessions) is
the planned second release.

## Status

Early. The context-budget library and retrieval metrics work and are tested. The task set,
answer runner and judge are in progress. See [docs/PLAN.md](docs/PLAN.md).

## Results

Not run yet. This table will be filled only from real runs, including strategies that
didn't help.

| Strategy | Tokens sent | Cost / 100 questions | Accuracy | Answer chunk sent |
|---|---|---|---|---|
| Baseline (cut at budget) | – | – | – | – |
| + dedupe & rank | – | – | – | – |
| + trim tool outputs | – | – | – | – |
| + compress history | – | – | – | – |
| + cache ordering | – | – | – | – |

## Quick start

```bash
pip install -e ".[dev]"
pytest
python examples/demo.py
```

```python
from wtms import Chunk, Message, Strategies, ToolOutput, assemble

ctx = assemble(
    "Why is my table still large after deleting rows?",
    budget=4000,
    system="Answer using the documents.",
    history=[Message("user", "..."), Message("assistant", "...")],
    chunks=[Chunk("docs/routine-vacuuming#1", "..."), ...],
    tool_outputs=[ToolOutput("server_logs", "...")],
    strategies=Strategies(cache_order=True),
)
ctx.messages                      # ready for a chat completions call
ctx.report.chunk_fate("docs/routine-vacuuming#1")   # "sent" / "duplicate" / "over_budget" / ...
ctx.report.sections               # tokens per section
```

## How the budget is spent

The system prompt and question always go in. Then recent history, then tool outputs, then
document chunks in ranked order. When space runs out, the oldest history turns go first,
then the oldest tool outputs. Turns come back if dropping a tool output freed enough room.

Token counts use `tiktoken` (`o200k_base`). If its encoding can't be loaded, counts fall
back to an estimate, and the report says so (`report.exact_token_count`).

## Layout

```
src/wtms/
  tokens.py            token counting
  types.py             Chunk, Message, ToolOutput
  context/
    rank.py            dedupe (word 3-gram Jaccard) + BM25 ranking
    trim.py            tool-output trimming
    compress.py        history compression (extractive or LLM summary)
    order.py           cache-friendly vs best-last ordering
    assemble.py        puts it together under a budget, with a report
  eval/
    retrieval.py       recall@k, MRR
    tasks.py           task-set format and loader
```

## Data

Questions are written against the [PostgreSQL documentation](https://www.postgresql.org/docs/current/),
used under the PostgreSQL License. Every task links the docs page its answer comes from.

## License

MIT
