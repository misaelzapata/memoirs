"""Migration 010 — candidate rejection index (Fix #1.B of GAP audit Fase 5C).

The ``rejection_reason`` column on ``memory_candidates`` already exists
(introduced by ``001_initial.py`` and preserved through ``008_scoping``),
so this migration is a no-op for the column itself — it stays
backwards-compatible with the spec's "ALTER TABLE … ADD COLUMN" by guarding
the ALTER behind a ``PRAGMA table_info`` check.

What this migration adds:

  * a partial index ``idx_memory_candidates_status_reject`` covering
    ``status = 'rejected'`` rows so that the new auditing queries
    (``SELECT … WHERE status='rejected' AND rejection_reason LIKE 'noise:%'``)
    stay cheap as the rejected pool grows.

The index is partial (``WHERE status = 'rejected'``) so it costs nothing on
non-rejected candidates and stays tight even when the corpus has millions
of accepted/merged rows.
"""
from __future__ import annotations

import sqlite3


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)


def up(conn: sqlite3.Connection) -> None:
    # Defensive ADD COLUMN — the spec calls it out, but every prior
    # migration has carried `rejection_reason` since 001. We guard so the
    # migration is safe on any DB that somehow lost it.
    if not _has_column(conn, "memory_candidates", "rejection_reason"):
        conn.execute("ALTER TABLE memory_candidates ADD COLUMN rejection_reason TEXT")

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_candidates_status_reject "
        "ON memory_candidates(status) WHERE status = 'rejected'"
    )


def down(conn: sqlite3.Connection) -> None:
    # We do NOT drop the column — it predates this migration. Just remove
    # the partial index we created.
    conn.execute("DROP INDEX IF EXISTS idx_memory_candidates_status_reject")
