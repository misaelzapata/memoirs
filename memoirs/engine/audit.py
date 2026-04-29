"""Layer 5 — Gemma curation pass.

Periodically audits the persisted corpus to catch errors that slipped past the
extraction stage:
  - misclassified types (e.g. fact that's actually a preference)
  - generic noise that shouldn't be a memory at all
  - duplicates that auto-merge missed
  - low-quality content (file paths, code, tool output)

Operates in batches; for each batch Gemma returns a JSON verdict per memory:
  {id, action: "keep" | "fix_type" | "archive", new_type?, reason}

Use --dry-run by default; --apply persists changes.
"""
from __future__ import annotations

import json
import logging
from sqlite3 import OperationalError as sqlite3_OperationalError
from typing import Iterable

from ..core.ids import utc_now
from ..db import MemoirsDB


log = logging.getLogger("memoirs.audit")


_AUDIT_PROMPT = """You are auditing memorias from an auto-extracted memory store.
Be RUTHLESS — most candidates that LOOK like preferences are extraction noise.

Output ONLY a JSON array. One object per input memory in the SAME ORDER.
Each item: {"id": "...", "action": "keep"|"fix_type"|"archive", "new_type": "..."|null, "reason": "..."}

ARCHIVE AGGRESSIVELY when:
- The content is a regex pattern, vocabulary list, or code snippet
  (e.g. "prefer|prefers|prefiero", `["prefer", "preferir"]`, anything with `|` or backticks)
- Content has stray punctuation, brackets, parens that suggest code/JSON fragment
  (e.g. "Prefer fastfn", "prefer official examples)", "prefer, like, hate, love")
- Content references the memoirs codebase itself (functions, modules, validators)
  (e.g. "prefers runtime candidates over `", "Use type=fact ONLY")
- Content is a generic world fact (no user-specific subject)
- Content is a tool path, file path, version number, or commit hash
- Content is tool output, log line, build error, or shell error
- Content is a question or imperative without subject
- Content is < 15 chars or > 200 chars of code-looking gibberish
- Content includes regex metacharacters: `\\b`, `(?:`, `[^`, `{4,200}`

FIX_TYPE when content is durable but mis-classified:
- "user prefers X" but typed as fact → preference
- "we decided to use X" but typed as fact → decision
- "<project_name>" but typed as fact → project
- "fix the X" but typed as fact → task

KEEP only when:
- Statement is about THE USER (prefiere, decide, hace, evita)
- Content is concrete, contextual, > 15 chars of natural language
- The type matches the content (preference is an actual preference)

Allowed types: preference, fact, project, task, decision, style, credential_pointer.

Memorias to audit:

{memorias_block}

Return the JSON array now (no fences, no preamble):"""


def audit_batch(db: MemoirsDB, memorias: list[dict]) -> list[dict]:
    """Run one curator-LLM audit pass over a batch (5-10 memorias). Returns verdicts."""
    from . import curator

    if not curator._have_curator():
        return [{"id": m["id"], "action": "keep", "new_type": None,
                 "reason": "curator unavailable, skipping"} for m in memorias]

    block = "\n".join(
        f"- id: {m['id'][:24]}\n  type: {m['type']}\n  content: {(m['content'] or '')[:160]}"
        for m in memorias
    )
    prompt = (
        "<start_of_turn>user\n"
        + _AUDIT_PROMPT.replace("{memorias_block}", block)
        + "<end_of_turn>\n<start_of_turn>model\n"
    )
    llm = curator._get_llm()
    out = llm.create_completion(prompt=prompt, max_tokens=600, temperature=0.2,
                                 stop=["<end_of_turn>", "\n\n\n"])
    text = out["choices"][0]["text"]
    try:
        verdicts = curator.validate_json_output(text)
    except Exception as e:
        log.warning("audit batch parse failed: %s; head=%s", e, text[:120])
        return [{"id": m["id"], "action": "keep", "new_type": None,
                 "reason": f"parse error: {e}"} for m in memorias]

    if not isinstance(verdicts, list):
        return []
    # Normalize verdicts and align by short-id prefix to actual id
    id_lookup = {m["id"][:24]: m["id"] for m in memorias}
    out_verdicts: list[dict] = []
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        short = str(v.get("id", "")).strip()
        full_id = id_lookup.get(short) or short
        out_verdicts.append({
            "id": full_id,
            "action": str(v.get("action", "keep")).strip().lower(),
            "new_type": v.get("new_type"),
            "reason": str(v.get("reason", ""))[:200],
        })
    return out_verdicts


def audit_corpus(
    db: MemoirsDB,
    *,
    limit: int = 50,
    batch_size: int = 5,
    apply: bool = False,
    type_filter: str | None = None,
) -> dict:
    """Audit a sample of memorias and report (or apply) Gemma's verdicts.

    Args:
        limit: total memorias to audit.
        batch_size: how many memorias per Gemma call.
        apply: if False (default), only report. If True, archive/fix-type rows.
        type_filter: limit audit to memorias of this type.
    """
    from . import curator

    if not curator._have_curator():
        return {"error": "Curator LLM not available — install [gemma] and pull a curator GGUF"}

    sql = (
        "SELECT id, type, content, importance, confidence, score, created_at "
        "FROM memories WHERE archived_at IS NULL"
    )
    params: list = []
    if type_filter:
        sql += " AND type = ?"
        params.append(type_filter)
    # Audit lowest-score / oldest first — those are the likeliest junk
    sql += " ORDER BY score ASC, created_at ASC LIMIT ?"
    params.append(limit)
    rows = [dict(r) for r in db.conn.execute(sql, params).fetchall()]

    if not rows:
        return {"audited": 0, "kept": 0, "reclassified": 0, "archived": 0, "verdicts": []}

    log.info("audit: starting on %d memorias (apply=%s)", len(rows), apply)
    # Eagerly load sqlite-vec so we can clean vec_memories rows when archiving
    if apply:
        try:
            from . import embeddings
            embeddings._require_vec(db)
        except Exception as e:
            log.warning("audit: vec extension not loadable (%s); will skip vec cleanup", e)

    all_verdicts: list[dict] = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i: i + batch_size]
        verdicts = audit_batch(db, batch)
        all_verdicts.extend(verdicts)
        for v in verdicts:
            mem = next((m for m in batch if m["id"] == v["id"]), None)
            if mem:
                snippet = (mem.get("content") or "")[:60].replace("\n", " ")
                log.info("audit %s: %s [%s] %s — %s",
                         v["id"][:16], v["action"], mem["type"], snippet, v["reason"][:60])

    counts = {"audited": len(rows), "kept": 0, "reclassified": 0, "archived": 0,
              "errors": 0}
    now = utc_now()
    for v in all_verdicts:
        action = v["action"]
        if action == "keep":
            counts["kept"] += 1
        elif action == "fix_type" and v.get("new_type"):
            counts["reclassified"] += 1
            if apply:
                db.conn.execute(
                    "UPDATE memories SET type = ?, updated_at = ? WHERE id = ?",
                    (v["new_type"], now, v["id"]),
                )
        elif action == "archive":
            counts["archived"] += 1
            if apply:
                db.conn.execute(
                    "UPDATE memories SET archived_at = ?, "
                    "  archive_reason = ? WHERE id = ?",
                    (now, f"audit: {v['reason'][:100]}", v["id"]),
                )
                # Also clear from vec_memories + memory_embeddings so retrieval
                # never serves the archived row again. Skip silently if vec0
                # isn't loaded — the rows will get cleaned next sync.
                try:
                    db.conn.execute("DELETE FROM vec_memories WHERE memory_id = ?", (v["id"],))
                except sqlite3_OperationalError:
                    pass
                db.conn.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (v["id"],))
        else:
            counts["errors"] += 1
    if apply:
        db.conn.commit()
    return {**counts, "applied": apply, "verdicts": all_verdicts[:50]}
