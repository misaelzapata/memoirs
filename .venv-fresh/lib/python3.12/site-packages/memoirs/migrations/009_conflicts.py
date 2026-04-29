"""Migration 009: conflict resolution UI (P5-2).

Persists contradictions detected by the sleep-time `contradictions` job (or
any heuristic detector) so the user can triage them through the inspector
or the CLI. Without this table contradictions were detected each cycle and
silently logged.

Schema design
-------------
- ``id`` — autoincrementing surrogate key (most recent = highest id).
- ``memory_a_id`` / ``memory_b_id`` — the two memorias under suspicion. The
  pair is treated as unordered: callers should normalise the pair (e.g.
  ``(min, max)``) before insertion to make the UNIQUE index reliable.
- ``similarity`` — cosine at detection time, NULL when the detector did not
  compute one.
- ``detected_at`` — UTC ISO-8601 timestamp.
- ``detector`` — provenance string (``gemma``, ``heuristic``, ...).
- ``reason`` — short rationale from the detector (free text).
- ``status`` — pending | resolved_keep_a | resolved_keep_b |
  resolved_keep_both | resolved_merge | dismissed.
- ``resolution_notes`` — optional human note attached on resolve.
- ``resolved_at`` — UTC ISO-8601 timestamp set on resolve.

The ``UNIQUE(memory_a_id, memory_b_id)`` constraint lets us call
``ON CONFLICT DO UPDATE`` to refresh stale entries on subsequent sleep
cycles instead of growing an unbounded queue.
"""
from __future__ import annotations

import sqlite3


SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_a_id TEXT NOT NULL,
    memory_b_id TEXT NOT NULL,
    similarity REAL,
    detected_at TEXT NOT NULL DEFAULT (datetime('now')),
    detector TEXT,
    reason TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    resolution_notes TEXT,
    resolved_at TEXT,
    UNIQUE(memory_a_id, memory_b_id)
);

CREATE INDEX IF NOT EXISTS idx_conflicts_status
    ON memory_conflicts(status);
"""


def up(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def down(conn: sqlite3.Connection) -> None:
    fk_state = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("DROP INDEX IF EXISTS idx_conflicts_status")
        conn.execute("DROP TABLE IF EXISTS memory_conflicts")
    finally:
        conn.execute(f"PRAGMA foreign_keys = {'ON' if fk_state else 'OFF'}")
