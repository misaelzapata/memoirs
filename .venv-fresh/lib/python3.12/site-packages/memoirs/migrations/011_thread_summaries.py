"""Migration 011 — thread summaries (auto-resume thread).

Persists a durable per-conversation summary so an agent that returns to a
chat after a long pause can be re-oriented without re-reading every message.

Schema design
-------------
- ``id`` — autoincrementing surrogate key.
- ``conversation_id`` — the conversation the summary is about. Indexed for
  the common "give me the latest summary for X" lookup.
- ``summary`` — 2-3 sentence durable summary text.
- ``generated_at`` — UTC ISO-8601 timestamp at generation time.
- ``message_count_at_summary`` — message count at the moment we ran. Used
  to detect "needs regenerate" after new messages arrived.
- ``last_message_ts`` — timestamp of the most recent message at generation
  time. Indexed for "what convs do I have summaries for" queries.
- ``pending_actions_json`` — JSON array of pending action strings.
- ``salient_entity_ids_json`` — JSON array of entity ids referenced.
- ``user_id`` — multi-tenant column (matches the rest of the corpus).

The ``UNIQUE(conversation_id, generated_at)`` constraint allows multiple
summaries per conversation across time without colliding on regeneration —
each cycle's snapshot is preserved (idempotency in the cron job is enforced
by a "summary newer than the last message" check, NOT by the unique key).
"""
from __future__ import annotations

import sqlite3


SCHEMA = """
CREATE TABLE IF NOT EXISTS thread_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    generated_at TEXT NOT NULL DEFAULT (datetime('now')),
    message_count_at_summary INTEGER NOT NULL,
    last_message_ts TEXT,
    pending_actions_json TEXT,
    salient_entity_ids_json TEXT,
    user_id TEXT NOT NULL DEFAULT 'local',
    UNIQUE(conversation_id, generated_at)
);

CREATE INDEX IF NOT EXISTS idx_thread_summaries_conv
    ON thread_summaries(conversation_id);

CREATE INDEX IF NOT EXISTS idx_thread_summaries_recent
    ON thread_summaries(last_message_ts DESC);
"""


def up(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def down(conn: sqlite3.Connection) -> None:
    fk_state = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("DROP INDEX IF EXISTS idx_thread_summaries_recent")
        conn.execute("DROP INDEX IF EXISTS idx_thread_summaries_conv")
        conn.execute("DROP TABLE IF EXISTS thread_summaries")
    finally:
        conn.execute(f"PRAGMA foreign_keys = {'ON' if fk_state else 'OFF'}")
