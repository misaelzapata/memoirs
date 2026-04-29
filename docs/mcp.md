# MCP server reference

memoirs ships a single-binary MCP server (Model Context Protocol 2025-06-18) that wires into any compatible client over stdio. 22 tools, no API key, no network.

## Connect

```bash
python -m memoirs mcp        # stdio JSON-RPC, blocks
```

Use the wiring snippet for your IDE:

### Claude Desktop / Claude Code

```jsonc
// ~/Library/Application Support/Claude/claude_desktop_config.json (macOS)
// or ~/.config/Claude/claude_desktop_config.json (Linux)
{
  "mcpServers": {
    "memoirs": {
      "command": "/path/to/memoirs/.venv/bin/python3",
      "args": ["-m", "memoirs", "--db", "/path/to/.memoirs/memoirs.sqlite", "mcp"]
    }
  }
}
```

### VS Code (Copilot Chat ≥ 1.95)

```jsonc
// .vscode/mcp.json
{
  "servers": {
    "memoirs": {
      "type": "stdio",
      "command": "/path/to/memoirs/.venv/bin/python3",
      "args": ["-m", "memoirs", "mcp"]
    }
  }
}
```

### Cursor / Continue / Cline / Codex

Same stdio shape under `mcpServers` in their respective settings files.

## Tools

### Retrieval

| tool | purpose |
|---|---|
| `mcp_get_context` | **The retrieval workhorse.** Query → ranked, conflict-resolved memories in ~600–1,500 tokens. Supports `top_k`, `max_lines`, `as_of` (time-travel), `user_id`, `agent_id`, `namespace`. |
| `mcp_search_memory` | Cosine similarity over `memories`. |
| `mcp_explain_context` | Like `mcp_get_context` but each result carries `provenance_chain` — the path query → entity → relation → memory. Use for debugging retrieval. |
| `mcp_summarize_thread` | Compress a 50+-message conversation into a single durable memory. |

### Write

| tool | purpose |
|---|---|
| `mcp_add_memory` | Manual insert. `type` ∈ {preference, fact, project, task, decision, style, credential_pointer, tool_call}. |
| `mcp_update_memory` | Versioned append-only update. Old version stays as `superseded_by`. |
| `mcp_record_tool_call` | Persist a tool invocation as a `tool_call`-typed memory with args + result hash. |
| `mcp_score_feedback` | User signal ±0.2 (👍 / 👎). Influences future ranking. |
| `mcp_forget_memory` | Soft-delete (`archived_at = now`). Reversible. |

### Ingest

| tool | purpose |
|---|---|
| `mcp_ingest_event` | Single message (idempotent by `message_id`). |
| `mcp_ingest_conversation` | Batch insert. Returns `(source_id, conversation_id, n_messages)`. |
| `mcp_status` | Counts + recent imports. |

### Engine

| tool | purpose |
|---|---|
| `mcp_extract_pending` | Run the curator over conversations with no candidates yet. |
| `mcp_consolidate_pending` | Promote candidates to memories. |
| `mcp_consolidate_with_gemma` | Force-curator a single candidate (debugging). |
| `mcp_audit_corpus` | Curator pass that flags misclassified / generic / noisy memories. |
| `mcp_run_maintenance` | Recompute scores, expire vencidas, archive low-value. |
| `mcp_list_memories` | Filterable list, ordered by score. |

### Graph

| tool | purpose |
|---|---|
| `mcp_index_entities` | Extract entities for memories not yet linked. |
| `mcp_get_project_context` | Memories + entities scoped to a project name. |
| `mcp_list_projects` | Project-typed entities. |

### Ops

| tool | purpose |
|---|---|
| `mcp_event_stats` | Pending / processing / done / failed counts in `event_queue`. |
| `mcp_export_user_data` | GDPR export → base64-encoded zip + manifest. |
| `mcp_import_user_data` | Re-hydrate a previously exported bundle. |

## Example session

The agent calls `mcp_get_context` at the start of every task:

```
client → server: tools/call mcp_get_context {"query": "what's our auth strategy?"}
server → client: {
  "context": [
    "[decision] auth uses session cookies + CSRF token, set 2026-03-15",
    "[fact] CSRF token is in X-CSRF-Token header, validated server-side",
    "[task] migrate to httpOnly cookies pending"
  ],
  "memories": [
    {"id": "mem_a3f...", "type": "decision", "score": 0.91, "similarity": 0.84,
     "provenance_chain": [
       {"step":0,"kind":"query","value":"auth strategy"},
       {"step":1,"kind":"entity_match","entity":"auth","confidence":0.92},
       {"step":2,"kind":"entity_to_memory","entity":"auth","memory_id":"mem_a3f..."}
     ]},
    ...
  ],
  "token_estimate": 287,
  "conflicts_resolved": 0,
  "as_of": null,
  "live": true
}
```

Cost: ~6 ms p50 on a 4,196-memory corpus with `MEMOIRS_RETRIEVAL_MODE=hybrid_graph`.

## Streaming

The HTTP API exposes `GET /context/stream?q=...` over SSE. MCP itself does not stream tool returns yet (protocol limitation), but the engine is already a generator (`assemble_context_stream`) so wrapping is a 1-liner once MCP supports it.

## Troubleshooting

- **First call is slow (5–8 s).** Curator + sentence-transformers cold-load. Subsequent calls hot.
- **MCP server stalls on parallel calls.** stdio is single-stream; sequential serving is by design. For high-concurrency, run `memoirs serve` (HTTP) and route MCP-style RPCs over it. Sub-agents in Claude Code's `Agent` tool should **not** call MCP directly — they share the parent's context via prompt.
- **`mcp_get_context` returns `[]`.** Check `memoirs status` — likely no memories yet. Run `memoirs extract` + `memoirs consolidate` after the first ingest.
