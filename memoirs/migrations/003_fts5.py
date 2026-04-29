"""Migration 003: FTS5 full-text index over memories.content.

Adds the `memories_fts` virtual table plus three triggers that keep it in
sync with the canonical `memories` table. Backfills any rows that already
exist so hybrid retrieval works immediately on pre-existing DBs.

The same schema is also applied lazily by
``memoirs.engine.hybrid_retrieval.ensure_fts_schema`` for environments that
don't run migrations on init — this migration is the canonical version and
``up()`` is purposely a strict superset of what the runtime fallback does.
"""
from __future__ import annotations

import sqlite3


CREATE = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory_id UNINDEXED,
    content,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories
WHEN NEW.archived_at IS NULL
BEGIN
    INSERT INTO memories_fts(memory_id, content) VALUES (NEW.id, NEW.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories
BEGIN
    DELETE FROM memories_fts WHERE memory_id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE OF content, archived_at ON memories
BEGIN
    DELETE FROM memories_fts WHERE memory_id = OLD.id;
    INSERT INTO memories_fts(memory_id, content)
        SELECT NEW.id, NEW.content WHERE NEW.archived_at IS NULL;
END;
"""


DROP = """
DROP TRIGGER IF EXISTS memories_fts_ai;
DROP TRIGGER IF EXISTS memories_fts_ad;
DROP TRIGGER IF EXISTS memories_fts_au;
DROP TABLE IF EXISTS memories_fts;
"""


def up(conn: sqlite3.Connection) -> None:
    conn.executescript(CREATE)
    # Backfill: copy non-archived memories into the new FTS index.
    conn.execute(
        "INSERT INTO memories_fts(memory_id, content) "
        "SELECT id, content FROM memories WHERE archived_at IS NULL"
    )


def down(conn: sqlite3.Connection) -> None:
    conn.executescript(DROP)
