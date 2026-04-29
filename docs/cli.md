# CLI reference

memoirs has 35 subcommands. Run `memoirs --help` for the live list. Below is a topic-grouped tour with the flags that matter day-to-day.

```
memoirs init        ingest      watch      status      doctor     setup
        daemon      sleep       logs       trace-id    conversations
        messages    mcp         serve      ui          extract    consolidate
        maintenance cleanup     audit      ask         why        index-entities
        projects-refresh        trace      graph       links      raptor
        review      models      eval       tool-calls  events     conflicts
        db          export      import     scope       share      unshare
```

Global:

```
--db PATH           override MEMOIRS_DB
--help              standard help
```

---

## Setup & lifecycle

### `memoirs setup`

One-shot bootstrap. Creates the venv, installs `[all]`, downloads the curator model, runs `init`, writes the MCP client config snippets. Use on fresh machines.

### `memoirs init`

Creates `.memoirs/memoirs.sqlite` and runs migrations to the latest version.

### `memoirs doctor`

Health check: deps installed, models present, DB writable, MCP clients configured, daemon status, GPU/Vulkan availability.

### `memoirs status [--json]`

Counts (sources, conversations, messages, candidates, memories) + last 5 import runs.

---

## Ingestion

### `memoirs ingest <path> [--kind auto|claude-export]`

Auto-detects `.jsonl`, `.json`, `.zip`, `.md`, `state.vscdb`. For Claude.ai's official export zip use `--kind claude-export`.

```bash
memoirs ingest ~/.claude/projects                     # batch all transcripts
memoirs ingest ~/Downloads/chatgpt-export.zip          # ChatGPT
memoirs ingest --kind claude-export ~/Downloads/claude-export.zip
```

### `memoirs watch <path> [--interval N] [--realtime]`

Polls (default 2 s) or uses `inotify` (`--realtime`, requires `[realtime]`). Idempotent.

### `memoirs daemon {start|stop|status|restart}`

Background = three workers:
- `watcher` — file change → re-ingest
- `extract` — pending conversations → candidates (with `--max-load`, `--min-free-mem-mb` self-throttling)
- `sleep` — async housekeeping cron (consolidate / dedup / link_rebuild / prune / contradictions / event_queue)

Skip the sleep worker with `daemon start --no-sleep`.

---

## Engine commands

| command | layer | what it does |
|---|---|---|
| `extract [--daemon] [--auto-consolidate]` | 2 | Curator pass on conversations without candidates |
| `consolidate [--curator gemma|heuristic|auto]` | 5.1 | Candidates → memories with ADD/UPDATE/MERGE/IGNORE/EXPIRE/ARCHIVE |
| `maintenance [--enrich-decisions]` | 5.2/5.3 | Recompute scores, expire / archive |
| `cleanup [--threshold X]` | 5.5 | Merge near-duplicates, flag contradictions |
| `audit [--limit N] [--apply]` | 5 | Curator-driven QA over the corpus |
| `index-entities` | 3 | NER pass on memories without entity links |
| `projects-refresh` | 3 | Re-derive project entities from conversation cwd metadata |

---

## Retrieval

### `memoirs ask <query> [--top-k 10] [--max-lines 5] [--as-of ISO]`

One-shot context query. The same path as `mcp_get_context`.

### `memoirs why <memory_id> --query "..."`

Provenance trace — print the path from query to memory step by step.

### `memoirs trace <conversation_id>`

Source → messages → candidates → memories chain.

### `memoirs graph {entities|decisions|memory|list} [--html]`

Render an interactive HTML graph (requires `[viz]`).

### `memoirs links {rebuild|prune|stats|show}`

Zettelkasten edges (P1-3). `rebuild --mode topk|absolute|adaptive|zscore`, `prune --max-per-memory N`, `show <id> [--depth 2]`.

### `memoirs raptor {build|stats|show|query}`

RAPTOR hierarchical tree (P1-6). Build clusters every layer; query descends through summaries.

---

## Curation & review

### `memoirs review [--auto-accept|--auto-reject]`

Interactive triage of pending candidates.

### `memoirs conflicts {list|show|resolve}`

Triage detected memory contradictions. `resolve <id> --action keep_a|keep_b|keep_both|merge|dismiss [--notes "..."]`.

### `memoirs tool-calls {list|stats}`

Inspect `tool_call`-typed memories.

### `memoirs sleep {run-once|status|history}`

Manually fire the housekeeping cron.

### `memoirs events {list|stats|process|requeue-failed}`

Drain the durable `event_queue`.

---

## Privacy & multi-tenancy

### `memoirs scope {set|show}`

Set the active user_id / agent_id / namespace. Persists to `~/.memoirs/scope.json`.

### `memoirs share <memory_id> --user <other_user>` / `unshare`

Per-memory ACL.

### `memoirs export [--user-id X] [--out alice.zip] [--include-embeddings] [--redact-pii]`

GDPR-friendly portable bundle (zip + sha256 manifest).

### `memoirs import <bundle.zip> [--mode merge|replace|new_user]`

Re-hydrate a memoirs DB from an export.

---

## Database

### `memoirs db {version|migrate|rollback|list|rebuild-fts|encrypt|decrypt|rekey}`

```bash
memoirs db version             # {current, target, pending}
memoirs db migrate             # apply pending
memoirs db rollback --steps 1
memoirs db list                # all migrations + applied flag
memoirs db rebuild-fts         # rebuild memories_fts virtual table
memoirs db encrypt --key 'passphrase'
memoirs db rekey --old <a> --new <b>
```

---

## Eval

### `memoirs eval [--suite synthetic_basic|<jsonl>] [--top-k 10] [--modes hybrid,bm25,dense] [--save results.json]`

LongMemEval / LoCoMo adapter is built in. Reports `precision@k`, `recall@k`, `MRR`, `Hit@1`, `Hit@5`, `time_to_first_relevant_ms`, `latency_p50/p95`.

```bash
memoirs eval --suite synthetic_basic --modes hybrid,bm25,dense
memoirs eval --suite ~/datasets/longmemeval.jsonl --save ~/results.json
```

---

## Surfaces

### `memoirs mcp`

Run the stdio MCP server. See [`mcp.md`](mcp.md).

### `memoirs serve [--host 127.0.0.1] [--port 8283]`

FastAPI server with REST + SSE streaming.

### `memoirs ui [--port 8284]`

Web inspector (HTMX + Tailwind via CDN, no build step). Browse memories, explore the graph, resolve conflicts.

---

## Models

### `memoirs models {pull|list}`

Reproducible model downloads. Currently supports `gemma-2b`, `qwen2.5-3b`, `phi-3.5-mini` GGUFs.

```bash
memoirs models list
memoirs models pull qwen2.5-3b
```

---

## Logs & introspection

### `memoirs logs [--follow] [--format json|text] [--since 5m] [--grep ERROR] [--trace-id <X>]`

Tail the rotating log file (`.memoirs/memoirs.log`).

### `memoirs trace-id`

Print the active trace_id (useful when correlating logs across requests).

### `memoirs conversations [--json]` / `memoirs messages [--limit N] [--conversation-id ID]`

Raw inspection.
