"""Migration 007: sleep_runs (sleep-time async consolidation telemetry).

Records every cycle of the :mod:`memoirs.engine.sleep_consolidation`
scheduler — a Letta-inspired idle-time housekeeping loop that runs jobs
like consolidation, dedup, link rebuild, and low-value pruning while the
daemon is otherwise quiet.

Schema design:
- ``id``           — autoincrementing primary key (most recent = highest id).
- ``started_at``   — UTC ISO-8601 timestamp captured before the first job.
- ``finished_at``  — UTC ISO-8601 timestamp captured after the last job.
- ``jobs_json``    — JSON array (or object) describing each job's outcome,
                     including timing, return value, and any per-job error.
                     Stored as opaque text so we can evolve the schema-of-
                     schema without further migrations.
- ``error``        — top-level error if the entire cycle aborted (e.g. a
                     pre-condition skip). Per-job failures live in
                     ``jobs_json``.

P1-4 in GAP.md (sleep-time async consolidation, Letta inspiration).
"""
from __future__ import annotations

import sqlite3


SCHEMA = """
CREATE TABLE IF NOT EXISTS sleep_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    jobs_json TEXT NOT NULL DEFAULT '[]',
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_sleep_runs_started
    ON sleep_runs(started_at DESC);
"""


def up(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def down(conn: sqlite3.Connection) -> None:
    fk_state = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("DROP INDEX IF EXISTS idx_sleep_runs_started")
        conn.execute("DROP TABLE IF EXISTS sleep_runs")
    finally:
        conn.execute(f"PRAGMA foreign_keys = {'ON' if fk_state else 'OFF'}")
