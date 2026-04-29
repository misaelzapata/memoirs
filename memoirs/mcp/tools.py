"""MCP tool catalog: schemas + dispatcher.

Splitting tool definitions from JSON-RPC plumbing so each lives in one file
and is easy to extend. To add a new tool: register it in `TOOL_SCHEMAS` and
implement a `_handle_<name>` function below; `call_tool()` dispatches by name.
"""
from __future__ import annotations

from typing import Any

from ..core.ids import utc_now
from ..db import MemoirsDB


# ---------------------------------------------------------------------------
# Tool schemas exposed on tools/list
# ---------------------------------------------------------------------------

_INGEST_OUTPUT = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "action": {"type": "string"},
        "source_uri": {"type": "string"},
        "conversation_id": {"type": "string"},
        "message_id": {"type": "string"},
        "ordinal": {"type": "integer"},
    },
    "required": ["ok", "action", "conversation_id", "message_id"],
}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    # Layer 1 — raw ingestion
    {
        "name": "mcp_ingest_event",
        "title": "Ingest Memory Event",
        "description": "Push a chat message or domain event into Memoirs raw storage in real time.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "event": {"type": "object", "description": "Optional event wrapper. If omitted, the arguments object is the event."},
                "type": {"type": "string"},
                "source": {"type": "string"},
                "conversation_id": {"type": "string"},
                "message_id": {"type": "string"},
                "role": {"type": "string"},
                "content": {},
                "created_at": {"type": "string"},
                "title": {"type": "string"},
                "project": {"type": "string"},
                "metadata": {"type": "object"},
            },
            "additionalProperties": True,
        },
        "outputSchema": _INGEST_OUTPUT,
    },
    {
        "name": "mcp_ingest_conversation",
        "title": "Ingest Conversation Batch",
        "description": "Push a whole conversation as a batch of messages.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "conversation_id": {"type": "string"},
                "title": {"type": "string"},
                "messages": {"type": "array", "items": {"type": "object"}},
                "metadata": {"type": "object"},
            },
            "required": ["messages"],
            "additionalProperties": True,
        },
    },
    {
        "name": "mcp_status",
        "title": "Memoirs Status",
        "description": "Return raw ingestion counts and recent import runs.",
        "inputSchema": {"type": "object", "additionalProperties": False},
    },
    # Layer 2 — extraction
    {
        "name": "mcp_extract_pending",
        "title": "Extract Memory Candidates",
        "description": "Run Gemma → spaCy cascade over conversations without candidates yet.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 50},
                "reprocess_with_gemma": {"type": "boolean", "default": False, "description": "Reprocess conversations whose candidates came from spaCy through Gemma"},
            },
            "additionalProperties": False,
        },
    },
    # Layer 5 — memory engine
    {
        "name": "mcp_consolidate_pending",
        "title": "Consolidate Memory Candidates",
        "description": "ADD / UPDATE / MERGE / IGNORE / CONTRADICTION based on dedup + similarity.",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 100}},
            "additionalProperties": False,
        },
    },
    {
        "name": "mcp_consolidate_with_gemma",
        "title": "Inspect Gemma Consolidation Decision",
        "description": "Run gemma_consolidate against a single pending candidate and return the decision (no DB writes). Useful for debugging the curator.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string"},
                "top_k": {"type": "integer", "default": 5, "description": "How many neighbors to feed into the prompt."},
            },
            "required": ["candidate_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "mcp_audit_corpus",
        "title": "Audit Memorias with Gemma",
        "description": "Gemma curates the corpus: flags misclassified, noisy, or generic memorias. dry-run by default; pass apply=true to act.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 30},
                "type": {"type": "string"},
                "apply": {"type": "boolean", "default": False},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "mcp_run_maintenance",
        "title": "Run Daily Maintenance",
        "description": "Recompute scores, expire memories past valid_to, archive low-value.",
        "inputSchema": {"type": "object", "additionalProperties": False},
    },
    {
        "name": "mcp_snapshot_create",
        "title": "Create point-in-time snapshot",
        "description": "Atomic VACUUM INTO copy of the live DB. Optional human label.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "optional label (e.g. 'before-cleanup')"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "mcp_snapshot_list",
        "title": "List snapshots",
        "description": "List point-in-time snapshots newest-first with size + memoria count.",
        "inputSchema": {"type": "object", "additionalProperties": False},
    },
    {
        "name": "mcp_snapshot_diff",
        "title": "Diff snapshots",
        "description": "Compare two snapshots (or a snapshot vs the live DB). Returns counts of added/removed/changed memorias. Pass 'live' for either side to compare against the current DB.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "string", "description": "first snapshot path or 'live'"},
                "b": {"type": "string", "description": "second snapshot path or 'live'", "default": "live"},
            },
            "required": ["a"],
            "additionalProperties": False,
        },
    },
    {
        "name": "mcp_snapshot_restore",
        "title": "Restore from snapshot",
        "description": "Replace the live DB with a snapshot. Always creates a safety snapshot of the current state first. CAUTION: this is destructive — only call after explicit user confirmation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "snapshot_path": {"type": "string"},
            },
            "required": ["snapshot_path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "mcp_get_context",
        "title": "Assemble Memory Context",
        "description": "THE retrieval tool. Returns ranked, conflict-resolved memory in ~600-1500 tokens. Pass as_of for time-travel queries.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 20},
                "max_lines": {"type": "integer", "default": 15},
                "as_of": {"type": "string", "description": "Optional ISO timestamp; queries memory state as of that moment (read-only, no usage increment)."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "mcp_resume_thread",
        "title": "Resume Conversation Thread",
        "description": (
            "Auto-resume: returns the durable thread summary, salient memories, "
            "last decisions and pending actions for a paused conversation. "
            "Pass conversation_id explicitly OR omit it to auto-detect from "
            "the latest JSONL transcript under ~/.claude/projects/ that maps "
            "to the current cwd. Generates a summary on the fly if none exists."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "conversation_id": {"type": "string"},
                "salient_limit": {"type": "integer", "default": 8},
                "generate_if_missing": {"type": "boolean", "default": True},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "mcp_summarize_thread",
        "title": "Summarize Conversation",
        "description": "Run Gemma over a conversation, persist a 200-400 char summary as a fact-memoria with is_summary=True. Use for long threads (50+ messages) to reduce future tokens.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "conversation_id": {"type": "string"},
            },
            "required": ["conversation_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "mcp_search_memory",
        "title": "Search Memories",
        "description": "Similarity search over memories (cosine if embeddings available).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "mcp_add_memory",
        "title": "Add Memory",
        "description": "Manually add a memory (e.g. user said 'remember this').",
        "inputSchema": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["preference", "fact", "project", "task", "decision", "style", "credential_pointer", "procedural"]},
                "content": {"type": "string"},
                "importance": {"type": "integer", "minimum": 1, "maximum": 5},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["type", "content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "mcp_update_memory",
        "title": "Update Memory (versioned)",
        "description": "Append-only update: archives the old memoria and creates a new version with valid_from=now. The new memory inherits importance/confidence from the old. Use when content drifts (e.g. user changes preference).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["memory_id", "content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "mcp_score_feedback",
        "title": "User Feedback on Memory",
        "description": "Bump a memoria's user_signal +0.2 if useful=true, -0.2 if useful=false. Recomputes score immediately. Call after retrieval when the memoria turned out helpful or wrong.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "useful": {"type": "boolean"},
            },
            "required": ["memory_id", "useful"],
            "additionalProperties": False,
        },
    },
    {
        "name": "mcp_explain_context",
        "title": "Explain Context Selection",
        "description": "Like mcp_get_context but returns per-memory explanations: similarity, score, what entities it provides, why it ranked above competitors, and a provenance_chain that traces the path from the query through the entity/memory graph to each result. Use for debugging retrieval quality.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 10},
                "max_hops": {"type": "integer", "default": 3, "description": "Max edges in the provenance chain BFS."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "mcp_forget_memory",
        "title": "Forget Memory",
        "description": "Archive a memory by id (soft delete).",
        "inputSchema": {
            "type": "object",
            "properties": {"memory_id": {"type": "string"}},
            "required": ["memory_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "mcp_list_memories",
        "title": "List Memories",
        "description": "List active memories ordered by score, optionally filtered by type.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "limit": {"type": "integer", "default": 50},
            },
            "additionalProperties": False,
        },
    },
    # Layer 3 — graph
    {
        "name": "mcp_index_entities",
        "title": "Index Memory Entities",
        "description": "Walk memories without entity links and extract+link entities.",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 500}},
            "additionalProperties": False,
        },
    },
    {
        "name": "mcp_get_project_context",
        "title": "Get Project Context",
        "description": "Memories + related entities for a given project name.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["project"],
            "additionalProperties": False,
        },
    },
    {
        "name": "mcp_list_projects",
        "title": "List Projects",
        "description": "List entities classified as projects.",
        "inputSchema": {"type": "object", "additionalProperties": False},
    },
    # Layer 5.7 — event queue (P0-4)
    {
        "name": "mcp_event_stats",
        "title": "Event Queue Stats",
        "description": (
            "Return counts by status (pending|processing|done|failed) plus "
            "the age in seconds of the oldest pending event. Useful for "
            "monitoring whether downstream consumers are keeping up."
        ),
        "inputSchema": {"type": "object", "additionalProperties": False},
    },
    # P3-6 — GDPR export + portable import
    {
        "name": "mcp_export_user_data",
        "title": "Export Memoirs Bundle",
        "description": (
            "GDPR-friendly export of the corpus as a portable zip bundle. "
            "Returns the bundle base64-encoded plus the manifest. The "
            "user_id filter only applies if the schema has the column "
            "(otherwise the whole corpus is exported)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "default": "local"},
                "redact_pii": {"type": "boolean", "default": False},
                "include_embeddings": {"type": "boolean", "default": True},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "mcp_import_user_data",
        "title": "Import Memoirs Bundle",
        "description": (
            "Re-hydrate the configured DB from a base64-encoded bundle "
            "previously emitted by mcp_export_user_data. Modes: merge "
            "(default, INSERT OR IGNORE), replace (wipe first), new_user "
            "(rewrite user_id)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "bundle_b64": {"type": "string"},
                "mode": {"type": "string",
                          "enum": ["merge", "replace", "new_user"],
                          "default": "merge"},
                "new_user_id": {"type": "string"},
                "verify": {"type": "boolean", "default": True},
            },
            "required": ["bundle_b64"],
            "additionalProperties": False,
        },
    },
    # Layer 5.8 — tool-call memory (P1-8)
    {
        "name": "mcp_record_tool_call",
        "title": "Record Tool Call",
        "description": (
            "Persist a tool invocation as a first-class memory of "
            "type='tool_call' (P1-8). Stores tool_name, args, a hash of the "
            "result, and status (success|error|cancelled). The agent should "
            "pass a short result_summary (not the full output)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string"},
                "args": {"type": "object", "additionalProperties": True},
                "result_summary": {
                    "type": "string",
                    "description": "Short human summary of the tool's output. Hashed and stored alongside.",
                },
                "status": {
                    "type": "string",
                    "enum": ["success", "error", "cancelled"],
                    "default": "success",
                },
                "conversation_id": {"type": "string"},
                "importance": {"type": "integer", "minimum": 1, "maximum": 5, "default": 2},
            },
            "required": ["tool_name", "args", "result_summary"],
            "additionalProperties": False,
        },
    },
]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def call_tool(db: MemoirsDB, name: str, arguments: dict) -> dict:
    """Execute a tool by name with the given arguments. Returns the structured payload
    that the caller wraps into MCP tool_result format."""
    handler = _HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"unknown tool: {name}")
    return handler(db, arguments)


# ---- handlers -------------------------------------------------------------


def _h_ingest_event(db: MemoirsDB, args: dict) -> dict:
    event = args.get("event") if isinstance(args.get("event"), dict) else args
    return {"ok": True, **db.ingest_event(event)}


def _h_ingest_conversation(db: MemoirsDB, args: dict) -> dict:
    return {"ok": True, **db.ingest_conversation_event(args)}


def _h_status(db: MemoirsDB, args: dict) -> dict:
    return db.status()


def _h_extract_pending(db: MemoirsDB, args: dict) -> dict:
    from ..engine import curator  # lazy: avoid loading spaCy / curator GGUF during MCP startup
    return curator.extract_pending(
        db,
        limit=int(args.get("limit", 50)),
        reprocess_with_gemma=bool(args.get("reprocess_with_gemma", False)),
    )


def _h_consolidate_pending(db: MemoirsDB, args: dict) -> dict:
    from ..engine import memory_engine
    return memory_engine.consolidate_pending(db, limit=int(args.get("limit", 100)))


def _h_consolidate_with_gemma(db: MemoirsDB, args: dict) -> dict:
    """Run `curator_consolidate` on a single pending candidate without persisting.

    Returns the raw decision (action/target_id/reason/source) plus the
    neighbors block used as context, so the operator can inspect WHY the
    curator LLM decided what it did. The MCP tool name keeps its legacy
    ``gemma`` spelling for backward compat with existing clients.
    """
    from ..engine import embeddings
    from ..engine import curator as _curator

    cid = str(args.get("candidate_id", "")).strip()
    if not cid:
        raise ValueError("candidate_id is required")
    top_k = int(args.get("top_k", 5))

    row = db.conn.execute(
        "SELECT id, type, content, importance, confidence, status FROM memory_candidates WHERE id = ?",
        (cid,),
    ).fetchone()
    if not row:
        raise ValueError(f"candidate {cid} not found")

    candidate = {
        "id": row["id"],
        "type": row["type"],
        "content": row["content"],
        "importance": row["importance"],
        "confidence": row["confidence"],
    }

    try:
        neighbors = embeddings.search_similar_memories(db, row["content"], top_k=top_k)
    except Exception:
        neighbors = []

    decision = _curator.curator_consolidate(candidate, neighbors)
    return {
        "candidate": candidate,
        "neighbors": [
            {
                "id": n.get("id"),
                "type": n.get("type"),
                "content": (n.get("content") or "")[:200],
                "similarity": n.get("similarity"),
            }
            for n in neighbors
        ],
        "decision": decision,
        "candidate_status": row["status"],
    }


def _h_audit_corpus(db: MemoirsDB, args: dict) -> dict:
    from ..engine.audit import audit_corpus
    return audit_corpus(
        db,
        limit=int(args.get("limit", 30)),
        type_filter=args.get("type"),
        apply=bool(args.get("apply", False)),
    )


def _h_run_maintenance(db: MemoirsDB, args: dict) -> dict:
    from ..engine import memory_engine
    return memory_engine.run_daily_maintenance(db)


def _h_get_context(db: MemoirsDB, args: dict) -> dict:
    from ..engine import memory_engine
    query = str(args.get("query", "")).strip()
    if not query:
        raise ValueError("query is required")
    as_of = args.get("as_of") or None
    return memory_engine.assemble_context(
        db,
        query,
        top_k=int(args.get("top_k", 20)),
        max_lines=int(args.get("max_lines", 15)),
        as_of=str(as_of) if as_of else None,
    )


def _h_summarize_thread(db: MemoirsDB, args: dict) -> dict:
    from ..engine import curator
    cid = str(args.get("conversation_id", ""))
    if not cid:
        raise ValueError("conversation_id is required")
    return curator.summarize_conversation(db, cid)


def _h_resume_thread(db: MemoirsDB, args: dict) -> dict:
    """Auto-resume a paused conversation thread.

    If ``conversation_id`` is not supplied we auto-detect by mapping the
    server's cwd to ``~/.claude/projects/-<encoded>/`` and picking the
    most recently-modified ``*.jsonl`` transcript. The session uuid in
    that filename is used to look up the matching conversation row.
    """
    from ..engine import thread_resume

    cid = (args.get("conversation_id") or "").strip() or None
    auto_detected = False
    if not cid:
        cid = thread_resume.find_conversation_id_for_cwd(db)
        auto_detected = bool(cid)
    if not cid:
        return {
            "ok": False,
            "auto_detected": False,
            "reason": "no conversation_id provided and no JSONL found for cwd",
            "conversation_id": None,
        }
    payload = thread_resume.resume_thread(
        db,
        cid,
        salient_limit=int(args.get("salient_limit", 8)),
        generate_if_missing=bool(args.get("generate_if_missing", True)),
    )
    payload["ok"] = True
    payload["auto_detected"] = auto_detected
    return payload


def _h_search_memory(db: MemoirsDB, args: dict) -> dict:
    from ..engine import embeddings
    query = str(args.get("query", "")).strip()
    if not query:
        raise ValueError("query is required")
    limit = int(args.get("limit", 10))
    return {"results": embeddings.search_similar_memories(db, query, top_k=limit)}


def _h_add_memory(db: MemoirsDB, args: dict) -> dict:
    from ..engine import memory_engine
    from ..engine.curator import Candidate, validate_allowed_memory_type

    cand = Candidate(
        type=str(args.get("type", "fact")),
        content=str(args.get("content", "")),
        importance=int(args.get("importance", 4)),
        confidence=float(args.get("confidence", 0.95)),
    )
    if not cand.content:
        raise ValueError("content is required")
    if not validate_allowed_memory_type(cand):
        raise ValueError(f"invalid memory type: {cand.type}")
    decision = memory_engine.decide_memory_action(db, cand)
    return memory_engine.apply_decision(db, cand, decision)


def _h_update_memory(db: MemoirsDB, args: dict) -> dict:
    from ..engine.memory_engine import create_memory_version
    mid = str(args.get("memory_id", ""))
    content = str(args.get("content", "")).strip()
    if not mid or not content:
        raise ValueError("memory_id and content are required")
    new_id = create_memory_version(db, mid, content)
    return {"ok": True, "old_memory_id": mid, "new_memory_id": new_id}


def _h_score_feedback(db: MemoirsDB, args: dict) -> dict:
    from ..engine.memory_engine import calculate_memory_score
    mid = str(args.get("memory_id", ""))
    useful = bool(args.get("useful", True))
    if not mid:
        raise ValueError("memory_id is required")
    delta = 0.2 if useful else -0.2
    db.conn.execute(
        "UPDATE memories SET "
        "  user_signal = MAX(0.0, MIN(1.0, COALESCE(user_signal,0) + ?)), "
        "  updated_at = ? "
        "WHERE id = ? AND archived_at IS NULL",
        (delta, utc_now(), mid),
    )
    row = db.conn.execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone()
    if not row:
        raise ValueError(f"memory {mid} not found")
    new_score = calculate_memory_score(dict(row))
    db.conn.execute(
        "UPDATE memories SET score = ? WHERE id = ?", (new_score, mid),
    )
    db.conn.commit()
    return {"ok": True, "memory_id": mid, "new_user_signal": row["user_signal"], "new_score": new_score}


def _h_explain_context(db: MemoirsDB, args: dict) -> dict:
    """Like get_context but with per-memory rationale + provenance chain.

    Each result now carries a ``provenance_chain`` field (P1-9) — a list
    of typed steps that traces the path from the query through the
    entity / memory_link graph to the candidate. The legacy
    ``rationale``, ``entities``, etc. fields stay for back-compat.
    """
    from ..engine import memory_engine, embeddings
    from ..engine import explain as explain_mod

    query = str(args.get("query", "")).strip()
    if not query:
        raise ValueError("query is required")
    top_k = int(args.get("top_k", 10))
    max_hops = int(args.get("max_hops", 3))
    candidates = embeddings.search_similar_memories(db, query, top_k=top_k)

    # Enrich with provenance — same candidates list, gains a
    # ``provenance_chain`` field per row.
    enriched = explain_mod.explain_memory_selection(
        db, query, None, candidates, max_hops=max_hops,
    )
    by_id = {c.get("id"): c.get("provenance_chain", []) for c in enriched}

    explained = []
    for m in candidates:
        # entity links for this memory
        ents = [
            r["name"] for r in db.conn.execute(
                "SELECT e.name FROM entities e JOIN memory_entities me "
                "ON me.entity_id = e.id WHERE me.memory_id = ? LIMIT 5",
                (m["id"],),
            ).fetchall()
        ]
        explained.append({
            "id": m["id"],
            "type": m["type"],
            "content": m["content"][:200],
            "similarity": m.get("similarity"),
            "score": m.get("score"),
            "entities": ents,
            "rationale": (
                f"sim={m.get('similarity', 0):.3f} score={m.get('score', 0):.3f} "
                f"importance={m['importance']} confidence={m['confidence']:.2f} "
                f"usage={m['usage_count']}"
            ),
            "provenance_chain": by_id.get(m["id"], []),
        })
    return {"query": query, "top_k": top_k, "results": explained}


def _h_forget_memory(db: MemoirsDB, args: dict) -> dict:
    mid = str(args.get("memory_id", ""))
    if not mid:
        raise ValueError("memory_id is required")
    db.conn.execute(
        "UPDATE memories SET archived_at = ?, archive_reason = 'user requested forget' WHERE id = ?",
        (utc_now(), mid),
    )
    db.conn.commit()
    return {"ok": True, "memory_id": mid}


def _h_list_memories(db: MemoirsDB, args: dict) -> dict:
    limit = int(args.get("limit", 50))
    mtype = args.get("type")
    if mtype:
        rows = db.conn.execute(
            "SELECT id, type, content, score, importance, confidence "
            "FROM memories WHERE archived_at IS NULL AND type = ? "
            "ORDER BY score DESC LIMIT ?",
            (mtype, limit),
        ).fetchall()
    else:
        rows = db.conn.execute(
            "SELECT id, type, content, score, importance, confidence "
            "FROM memories WHERE archived_at IS NULL "
            "ORDER BY score DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return {"memories": [dict(r) for r in rows]}


def _h_index_entities(db: MemoirsDB, args: dict) -> dict:
    from ..engine import graph
    return graph.index_memory_entities(db, limit=int(args.get("limit", 500)))


def _h_get_project_context(db: MemoirsDB, args: dict) -> dict:
    from ..engine import graph
    project = str(args.get("project", ""))
    if not project:
        raise ValueError("project is required")
    return graph.get_project_context(db, project, limit=int(args.get("limit", 20)))


def _h_list_projects(db: MemoirsDB, args: dict) -> dict:
    """Return projects, ensuring the entities table reflects what the
    conversations corpus already knows. Auto-refresh is cheap (just SQL)."""
    from ..engine.graph import refresh_projects_from_conversations
    refresh_projects_from_conversations(db)
    rows = db.conn.execute(
        "SELECT e.id, e.name, e.type, COUNT(me.memory_id) AS memories "
        "FROM entities e LEFT JOIN memory_entities me ON me.entity_id = e.id "
        "WHERE e.type = 'project' GROUP BY e.id ORDER BY memories DESC, e.name"
    ).fetchall()
    return {"projects": [dict(r) for r in rows]}


def _h_event_stats(db: MemoirsDB, args: dict) -> dict:
    """Return per-status counts + age of the oldest pending row (P0-4)."""
    from ..engine import event_queue as eq

    return {"ok": True, **eq.get_stats(db)}


def _h_record_tool_call(db: MemoirsDB, args: dict) -> dict:
    """Wrapper around `record_tool_call`. The MCP surface accepts
    ``result_summary`` (a string) rather than the raw result, so giant tool
    outputs never travel over JSON-RPC; the engine hashes and stores the
    summary verbatim.
    """
    from ..engine import memory_engine

    tool_name = str(args.get("tool_name", "")).strip()
    if not tool_name:
        raise ValueError("tool_name is required")
    raw_args = args.get("args")
    if raw_args is None:
        raw_args = {}
    if not isinstance(raw_args, dict):
        raise ValueError("args must be a JSON object")
    result_summary = args.get("result_summary")
    if result_summary is None:
        raise ValueError("result_summary is required")
    status = str(args.get("status", "success"))
    conversation_id = args.get("conversation_id") or None
    importance = int(args.get("importance", 2))

    mid = memory_engine.record_tool_call(
        db,
        tool_name=tool_name,
        args=raw_args,
        result=result_summary,
        status=status,
        conversation_id=str(conversation_id) if conversation_id else None,
        importance=importance,
    )
    row = db.conn.execute(
        "SELECT content FROM memories WHERE id = ?", (mid,),
    ).fetchone()
    return {
        "ok": True,
        "memory_id": mid,
        "content": row["content"] if row else None,
    }


def _h_export_user_data(db: MemoirsDB, args: dict) -> dict:
    """Export the corpus to a zip bundle and return it base64-encoded.

    The bundle is materialised in a temp file (zipfile needs a seekable
    handle), read back, base64-encoded, and the temp file deleted before
    returning.
    """
    import base64
    import tempfile

    from ..export import export_user_data

    user_id = args.get("user_id")
    if user_id is None:
        user_id = "local"
    redact_pii = bool(args.get("redact_pii", False))
    include_embeddings = bool(args.get("include_embeddings", True))

    with tempfile.TemporaryDirectory(prefix="memoirs-export-") as tmp:
        out_path = __import__("pathlib").Path(tmp) / "bundle.zip"
        manifest = export_user_data(
            db,
            user_id=user_id,
            out_path=out_path,
            include_embeddings=include_embeddings,
            redact_pii=redact_pii,
        )
        data = out_path.read_bytes()
    return {
        "ok": True,
        "bundle_b64": base64.b64encode(data).decode("ascii"),
        "size_bytes": len(data),
        "manifest": manifest.to_dict(),
    }


def _h_import_user_data(db: MemoirsDB, args: dict) -> dict:
    """Decode a base64 bundle and run :func:`import_user_data` against it."""
    import base64
    import tempfile

    from ..export import import_user_data

    blob_b64 = args.get("bundle_b64")
    if not isinstance(blob_b64, str) or not blob_b64:
        raise ValueError("bundle_b64 is required")
    try:
        blob = base64.b64decode(blob_b64)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"bundle_b64 is not valid base64: {exc}")

    mode = str(args.get("mode", "merge"))
    new_user_id = args.get("new_user_id")
    verify = bool(args.get("verify", True))

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(blob)
        tmp_path = tmp.name
    try:
        report = import_user_data(
            db,
            in_path=__import__("pathlib").Path(tmp_path),
            mode=mode,
            new_user_id=new_user_id,
            verify=verify,
        )
    finally:
        try:
            __import__("os").unlink(tmp_path)
        except OSError:
            pass
    return {"ok": True, **report.to_dict()}


_HANDLERS = {
    "mcp_ingest_event": _h_ingest_event,
    "mcp_ingest_conversation": _h_ingest_conversation,
    "mcp_status": _h_status,
    "mcp_extract_pending": _h_extract_pending,
    "mcp_consolidate_pending": _h_consolidate_pending,
    "mcp_consolidate_with_gemma": _h_consolidate_with_gemma,
    "mcp_audit_corpus": _h_audit_corpus,
    "mcp_run_maintenance": _h_run_maintenance,
    "mcp_get_context": _h_get_context,
    "mcp_summarize_thread": _h_summarize_thread,
    "mcp_resume_thread": _h_resume_thread,
    "mcp_search_memory": _h_search_memory,
    "mcp_add_memory": _h_add_memory,
    "mcp_update_memory": _h_update_memory,
    "mcp_score_feedback": _h_score_feedback,
    "mcp_explain_context": _h_explain_context,
    "mcp_forget_memory": _h_forget_memory,
    "mcp_list_memories": _h_list_memories,
    "mcp_index_entities": _h_index_entities,
    "mcp_get_project_context": _h_get_project_context,
    "mcp_list_projects": _h_list_projects,
    "mcp_record_tool_call": _h_record_tool_call,
    "mcp_event_stats": _h_event_stats,
    "mcp_export_user_data": _h_export_user_data,
    "mcp_import_user_data": _h_import_user_data,
}

# Snapshots — defined inline above the dict update so the names resolve.
def _h_snapshot_create(db: MemoirsDB, params: dict) -> dict:
    from ..engine import snapshots as _snap
    info = _snap.create(db.path, name=params.get("name"))
    return {
        "ok": True,
        "path": str(info.path),
        "name": info.name,
        "created_at": info.created_at,
        "size_bytes": info.size_bytes,
        "memory_count": info.memory_count,
    }


def _h_snapshot_list(db: MemoirsDB, params: dict) -> dict:
    from ..engine import snapshots as _snap
    snaps = _snap.list_snapshots(db.path)
    return {
        "ok": True,
        "count": len(snaps),
        "snapshots": [
            {
                "path": str(s.path),
                "name": s.name,
                "created_at": s.created_at,
                "size_bytes": s.size_bytes,
                "memory_count": s.memory_count,
            }
            for s in snaps
        ],
    }


def _h_snapshot_diff(db: MemoirsDB, params: dict) -> dict:
    from pathlib import Path as _P
    from ..engine import snapshots as _snap
    a = params.get("a")
    b = params.get("b", "live")
    a_path = db.path if a == "live" else _P(a)
    b_path = db.path if b == "live" else _P(b)
    d = _snap.diff(a_path, b_path)
    return {"ok": True, **d}


def _h_snapshot_restore(db: MemoirsDB, params: dict) -> dict:
    from ..engine import snapshots as _snap
    sp = params["snapshot_path"]
    db.close()
    info = _snap.restore(sp, db.path)
    return {
        "ok": True,
        "restored_from": str(info.path),
        "restored_at_count": info.memory_count,
    }


_HANDLERS.update({
    "mcp_snapshot_create": _h_snapshot_create,
    "mcp_snapshot_list": _h_snapshot_list,
    "mcp_snapshot_diff": _h_snapshot_diff,
    "mcp_snapshot_restore": _h_snapshot_restore,
})
