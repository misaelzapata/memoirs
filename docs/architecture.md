# Architecture

memoirs is a 6-layer pipeline that turns raw chat transcripts into a queryable, ranked, conflict-resolved memory store. Every layer is independently testable and disable-able.

## Data flow

```
sources (jsonl, vscdb, zip, md, claude_export)
   │
   │ ingest layer
   ▼
sources / conversations / messages              ← Layer 1: raw immutable history
   │
   │ extract layer (Qwen / Phi / Gemma + spaCy fallback)
   ▼
memory_candidates                               ← Layer 2: typed extraction
   │  type ∈ {preference, fact, project, task, decision, style,
   │          credential_pointer, tool_call}
   │
   │ consolidate layer (curator: Qwen JSON ADD/UPDATE/MERGE/IGNORE/EXPIRE)
   ▼
memories                                        ← Layer 5: durable corpus
   │  + score, importance, confidence, strength, last_accessed_at,
   │    valid_from / valid_to, archived_at, superseded_by, scope
   │
   │ embedding layer (sentence-transformers / fastembed)
   ▼
memory_embeddings  ──→  memories_fts (FTS5)     ← Layer 4: dual index
   │
   │ knowledge graph layer
   ▼
entities / relationships / memory_entities      ← Layer 3
memory_links (Zettelkasten)
summary_nodes / summary_node_members (RAPTOR)
   │
   │ retrieval layer (assemble_context)
   ▼
ranked context  →  Layer 6: MCP / HTTP / UI
```

## Layer detail

### Layer 1 — raw

`sources`, `conversations`, `messages`. Append-only, deduped by `(source_uri, mtime, size)` for files and `(conversation_external_id, ordinal)` for messages. Nothing is overwritten — re-ingesting a file is idempotent.

Ingesters live in `memoirs/ingesters/`:
- `claude_code.py` — Anthropic transcripts at `~/.claude/projects/**/*.jsonl`
- `cursor.py` — Cursor's `state.vscdb` (defensive — can be locked)
- `claude_export.py` — official Claude.ai export zip
- `importers.py` — ChatGPT zip, generic JSONL, Markdown

Each ingester emits a `messages_ingested` event into `event_queue` for the sleep cron to pick up.

### Layer 2 — extraction

`memoirs/engine/gemma.py` (despite the name, hosts Qwen / Phi / Gemma — auto-detected). The prompt asks the local LLM for a JSON array of typed candidates with `importance ∈ [1..5]` and `confidence ∈ [0..1]`.

Robustness:
- Token-budget chunking — long conversations are split using the model's real `tokenize()` (not char/4 heuristics) with 200-token overlap.
- Tolerant JSON parser that salvages bare strings, fenced output, BOMs, mid-string truncation.
- spaCy-only fallback if no model is loaded.
- All candidates pass through `core/redact.py` (PII + secrets) and `core/normalize.py` before persistence.

### Layer 3 — knowledge graph

`entities` (people / tools / projects / concepts), `relationships` (typed edges with confidence), `memory_entities` (memory ↔ entity). Plus:
- `memory_links` — Zettelkasten edges between memories (semantic + shared_entity), 4 selection modes (absolute / topk / adaptive / zscore).
- `summary_nodes` + `summary_node_members` — RAPTOR hierarchical summary tree.

### Layer 4 — embeddings

`memory_embeddings` (BLOB + model name) feeds `vec0` (sqlite-vec). `memories_fts` is an FTS5 virtual table kept in sync via triggers on `memories`. Hybrid retrieval fuses both via Reciprocal Rank Fusion.

LRU cache on query embeddings: ~2,000× speedup on repeats (0.005 ms vs 9.7 ms cold).

### Layer 5 — engine

The brains. Lives in `memoirs/engine/memory_engine.py` plus specialized siblings:
- `memory_engine.py` — scoring, decide_memory_action, apply_decision, assemble_context (+ stream variant)
- `lifecycle.py` — daily maintenance, expire/archive policies
- `lifecycle_decisions.py` — EXPIRE/ARCHIVE generation via `enrich_decision`
- `zettelkasten.py` — link_memory, link_by_shared_entities, get_neighbors (recursive CTE BFS)
- `hybrid_retrieval.py` — BM25 + dense + RRF
- `graph_retrieval.py` — Personalized PageRank multi-hop (HippoRAG-style)
- `raptor.py` — cluster + summarize + tree retrieval
- `reranker.py` — cross-encoder rerank (BGE optional)
- `hyde.py` — query expansion via curator
- `mmr.py` — Maximal Marginal Relevance diversification
- `explain.py` — provenance chain construction
- `sleep_consolidation.py` — async cron (consolidate / dedup / link_rebuild / prune / contradictions / event_queue)
- `event_queue.py` — durable async work queue (8 status states, requeue_failed)
- `acl.py` — visibility tiers + share/unshare
- `conflicts.py` — record + resolve (keep_a / keep_b / keep_both / merge / dismiss)

Score formula (post-Ebbinghaus):

```
score = importance · 0.35
      + confidence · 0.20
      + R(Δt, S) · 0.15        ← R = e^(-Δt_h / (S · 24)), clamp [0.01, 1.0]
      + usage_curve · 0.15
      + user_signal · 0.15
```

Strength `S` starts at 1.0, multiplies by 1.5 on each access (cap 64), and is the consolidation factor — old memories that are still being used decay slowly, unused ones decay fast.

### Layer 6 — surface

Three transports for the same engine:

- **MCP** — `memoirs/mcp/server.py` (stdio JSON-RPC, MCP 2025-06-18). 22 tools.
- **HTTP** — `memoirs/api/server.py` (FastAPI). REST + SSE streaming on `/context/stream`. Inspector UI mounted on `/ui`.
- **CLI** — `memoirs/cli.py`. 35 subcommands. Same library underneath.

## Schema versions

Migrations are versioned under `memoirs/migrations/NNN_name.py` with explicit `up(conn)` / `down(conn)`. Each runs in a transaction; failure rolls back. Current target: **v9**.

| version | name | brings |
|---|---|---|
| 1 | initial | sources / conversations / messages / candidates / memories / embeddings / entities / relationships / memory_entities / event_queue / import_runs |
| 2 | memory_links | Zettelkasten edges + indexes |
| 3 | fts5 | `memories_fts` virtual table + sync triggers |
| 4 | memory_strength | Ebbinghaus columns + index on `last_accessed_at` |
| 5 | tool_call_memory | tool_name, tool_args_json, tool_result_hash, tool_status |
| 6 | raptor_summaries | summary_nodes + summary_node_members |
| 7 | sleep_runs | cron job audit log |
| 8 | scoping | user_id / agent_id / run_id / namespace / visibility + memory_share |
| 9 | conflicts | memory_conflicts triage table |

`memoirs db migrate` applies pending; `memoirs db rollback --steps N` undoes. Schema state is also reproducible from scratch via the migration runner — `db.SCHEMA` is a re-export, not a duplicate.

## Concurrency model

- SQLite WAL mode + `mmap_size=256MB` + `temp_store=MEMORY`.
- One singleton LLM (`_LLM_SINGLETON`) protected by a `threading.Lock`. Lazy-loaded on first use.
- The hot retrieval path (`assemble_context`) **never** loads the LLM unless `MEMOIRS_RETRIEVAL_GEMMA=on` is set, and even then is capped at 2 calls per query.
- Daemon = three workers (watcher, extractor, sleep) each in its own process so the embedding/curator memory isn't duplicated and one crash doesn't kill the others.

## Test layout

```
tests/
├── test_db.py / test_migrations.py / test_pragmas.py    ← schema
├── test_normalize.py / test_redact.py                   ← ingest hygiene
├── test_extract_spacy.py / test_gemma_*.py              ← extraction
├── test_scoring.py / test_lifecycle*.py                 ← engine
├── test_zettelkasten*.py / test_graph_retrieval.py      ← graph
├── test_hybrid_retrieval.py / test_retrieval_pipeline.py ← retrieval
├── test_raptor*.py                                      ← summaries
├── test_*_coverage.py                                   ← coverage targets
└── test_eval_harness.py                                 ← LongMemEval / LoCoMo
```

693 tests, ~90 s end-to-end. Coverage target: 60% (`fail_under` in pyproject).
