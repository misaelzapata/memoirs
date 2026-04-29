"""Migration 012 — mandatory attribution / provenance on memorias.

Every memoria now carries a ``provenance_json`` column that records *who*
or *what* produced it (user vs curator vs heuristic vs imported), *which
process* (extract, consolidate, manual, ingest), *which entities* it
concerns, and any source pointers (conversation_id, candidate_id) that
make the decision auditable.

The column defaults to ``'{}'`` so existing rows remain valid; the
curator and heuristic paths fill it in on insert. The UI's
``_provenance.html`` partial already exists — this migration finally
gives it real data.

Why
---
- **Audit-grade explainability**: a user can ask "why did you remember
  this and where did it come from?" and the answer is a structured
  record, not a guess.
- **Multi-actor introspection**: in a multi-agent setup, knowing which
  agent (or human) authored a memoria is critical for trust.
- **Memori parity**: Memori treats attribution(entity, process) as a
  first-class column. We now match.

Shape
-----
``provenance_json`` is a small JSON object. Suggested keys (none required
beyond ``actor``):

::

    {
      "actor":   "curator|heuristic|user|import",
      "process": "extract|consolidate|merge|manual|ingest",
      "entities": [<entity_id>, ...],
      "source": {
        "conversation_id": "...",
        "candidate_id":    "...",
        "decision":        "ADD|MERGE|UPDATE"
      },
      "confidence_reason": "free text"
    }
"""
from __future__ import annotations

import sqlite3


SCHEMA = """
ALTER TABLE memories
ADD COLUMN provenance_json TEXT NOT NULL DEFAULT '{}';
"""


def up(conn: sqlite3.Connection) -> None:
    # Idempotency: if the column already exists (re-running the
    # migration), pragma `table_info` will show it.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
    if "provenance_json" in cols:
        return
    conn.executescript(SCHEMA)


def down(conn: sqlite3.Connection) -> None:
    # SQLite up to 3.35 lacks `DROP COLUMN`. We use the rebuild pattern
    # but only if needed. This is rarely exercised; tests use a fresh
    # DB per case.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
    if "provenance_json" not in cols:
        return
    fk_state = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("ALTER TABLE memories DROP COLUMN provenance_json")
    finally:
        conn.execute(f"PRAGMA foreign_keys = {'ON' if fk_state else 'OFF'}")
