"""Migration 002: memory_links (A-MEM Zettelkasten linking).

Introduces a `memory_links` table that captures memory↔memory relationships
discovered post-insert: semantic neighbors (top-k by cosine), shared-entity
co-occurrences, and (future) shared-tag / temporal links. The result is a
Zettelkasten-style graph that evolves as new memories arrive.

Schema design:
- `(source_memory_id, target_memory_id, reason)` is UNIQUE so re-running the
  linker is idempotent. Different reasons can produce parallel edges between
  the same two memories (semantic + shared_entity, etc.) — that's intentional.
- Two indexes on (source_memory_id) and (target_memory_id) keep neighbor
  lookups O(deg) regardless of corpus size.
- Bidirectional edges are stored explicitly (A→B and B→A as two rows) so
  traversal SQL doesn't need UNION.

P1-3 in GAP.md (A-MEM, NeurIPS 2025).
"""
from __future__ import annotations

import sqlite3


SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_memory_id TEXT NOT NULL,
    target_memory_id TEXT NOT NULL,
    similarity REAL NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source_memory_id, target_memory_id, reason)
);

CREATE INDEX IF NOT EXISTS idx_memory_links_source
    ON memory_links(source_memory_id);

CREATE INDEX IF NOT EXISTS idx_memory_links_target
    ON memory_links(target_memory_id);
"""


def up(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def down(conn: sqlite3.Connection) -> None:
    fk_state = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("DROP INDEX IF EXISTS idx_memory_links_target")
        conn.execute("DROP INDEX IF EXISTS idx_memory_links_source")
        conn.execute("DROP TABLE IF EXISTS memory_links")
    finally:
        conn.execute(f"PRAGMA foreign_keys = {'ON' if fk_state else 'OFF'}")
