# Plan

About 5 hours a week. Python, OpenAI API (small models), PostgreSQL docs.

An honest note on time: the first brief said 4–6 weekends for the context half. At 5 hours a
week that is closer to **9 weeks to a public v1**, because writing and checking the task set
is slow work that can't be skipped. Memory quality is a second release after that.

Each week ends with something committed and working.

| Week | Build | Done when |
|---|---|---|
| 1 | Context-budget core: dedupe/rank, trim, compress, ordering, report, tests | `pytest` passes; demo runs (**done**) |
| 2 | Corpus: download the current PostgreSQL docs, split by section into chunks with stable ids (`page#section`); BM25 search over the whole corpus | `search("autovacuum threshold")` returns sensible chunks |
| 3 | Task set, part 1: 30 questions (lookup + multi-hop), each with answer, source URL and relevant chunk ids | `load_tasks` validates the file; every URL opens |
| 4 | Task set, part 2: 30 more (long-history and tool-output kinds); first retrieval numbers (recall@k, MRR) for BM25 vs OpenAI embeddings | Retrieval table for both retrievers |
| 5 | Answer runner: one OpenAI call per task × strategy × budget; log tokens, cost and latency from the API's usage field; cache every response to JSONL so reruns cost nothing | 60 tasks × baseline run end to end |
| 6 | Judge: LLM-as-judge scores correctness and faithfulness; hand-grade 30 answers yourself and report how often the judge agrees | Agreement number written down |
| 7 | Experiments: each strategy on/off at 2–3 budgets; a script turns results into the README table and one chart | Table generated, not typed |
| 8 | Tool tasks through MCP: a small MCP server with `run_sql` and `get_logs` over a sample Postgres in Docker, so tool outputs are real | Tool-kind tasks run through the MCP server |
| 9 | Publish: README results and "what didn't help", write-up, make repo public, LinkedIn post + Featured + resume line | Repo public |

## Release 2: memory quality (about 6 weeks)

Scripted multi-session conversations with known correct answers, testing:
write quality (saves what matters), recall (right memory when needed), freshness
(newer fact beats older, e.g. "moved to Chennai" over "lives in Vellore"), leakage
(no cross-user memory) and bloat (recall as memory grows). Failures are traced per
stage: never saved / not recalled / recalled but cut from the context.

## Rules for results

- Every number comes from a real run, and the run's cached responses are kept in the repo.
- Report strategies that didn't help, and say by how much.
- Budget: keep a full eval run to a few dollars; cache responses and rerun only what changed.
- Nothing from Atatus: no internal prompts, data, methods, code or names.
