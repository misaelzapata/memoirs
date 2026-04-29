"""Migration 004: Ebbinghaus forgetting-curve columns on `memories`.

Adds two columns plus an index that the new recency scorer uses:

- ``strength``           — REAL, default 1.0. Consolidation factor S in
                            R(t) = exp(-Δt / (S * 24 hours)). Each access
                            multiplies it by 1.5 (capped at 64.0 by the
                            engine).
- ``last_accessed_at``   — TEXT (ISO-8601). Updated on every retrieval.
                            Distinct from ``last_used_at`` (which the
                            consolidation path bumps for UPDATE/MERGE) so we
                            can keep the existing usage signal intact while
                            tracking pure read access.

Backfill: every existing memory gets ``last_accessed_at`` initialized to
``last_used_at`` if present, else ``updated_at``, else ``valid_from``, else
``created_at``. ``strength`` defaults to 1.0 (the column-default applies to
every existing row).

Rollback: SQLite < 3.35 cannot drop columns directly. We rebuild the table
from a SELECT of the pre-004 columns — keeping data intact — drop the index,
and rename it back. Tables that reference ``memories`` via FK keep working
because the rebuild preserves the same primary key column.

P1-7 in GAP.md (MemoryBank, AAAI 2024).
"""
from __future__ import annotations

import sqlite3


# Columns of `memories` BEFORE this migration (mirrors 001_initial.SCHEMA).
# The rebuild in down() relies on this list to copy data back into the new
# table without mentioning the columns we're trying to remove.
_PRE_004_COLUMNS = (
    "id",
    "type",
    "content",
    "content_hash",
    "importance",
    "confidence",
    "score",
    "usage_count",
    "last_used_at",
    "user_signal",
    "valid_from",
    "valid_to",
    "superseded_by",
    "archived_at",
    "archive_reason",
    "metadata_json",
    "created_at",
    "updated_at",
)

# Verbatim CREATE TABLE for the pre-004 shape — used by down() to reconstruct
# the table during rollback. Mirrors 001_initial.SCHEMA exactly.
_PRE_004_CREATE = """
CREATE TABLE memories_old_004 (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 3,
    confidence REAL NOT NULL DEFAULT 0.5,
    score REAL NOT NULL DEFAULT 0.0,
    usage_count INTEGER NOT NULL DEFAULT 0,
    last_used_at TEXT,
    user_signal REAL NOT NULL DEFAULT 0.0,
    valid_from TEXT,
    valid_to TEXT,
    superseded_by TEXT REFERENCES memories_old_004(id),
    archived_at TEXT,
    archive_reason TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)


def up(conn: sqlite3.Connection) -> None:
    # Columns — guarded so the migration is idempotent against legacy DBs
    # that may have grown the column out-of-band.
    if not _has_column(conn, "memories", "strength"):
        conn.execute(
            "ALTER TABLE memories ADD COLUMN strength REAL NOT NULL DEFAULT 1.0"
        )
    if not _has_column(conn, "memories", "last_accessed_at"):
        conn.execute("ALTER TABLE memories ADD COLUMN last_accessed_at TEXT")

    # Index for queries that sort by recency / staleness.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_last_accessed "
        "ON memories(last_accessed_at)"
    )

    # Backfill — choose the most recent timestamp we have on hand.
    conn.execute(
        """
        UPDATE memories
        SET last_accessed_at = COALESCE(
            last_used_at,
            updated_at,
            valid_from,
            created_at
        )
        WHERE last_accessed_at IS NULL
        """
    )


def down(conn: sqlite3.Connection) -> None:
    """Reverse 004 by rebuilding `memories` without `strength` /
    `last_accessed_at`. SQLite versions older than 3.35 do not support
    ``DROP COLUMN``; the rebuild path works on every version and is the
    technique recommended by the SQLite docs.
    """
    # If the columns don't exist, this is a no-op.
    has_strength = _has_column(conn, "memories", "strength")
    has_last_accessed = _has_column(conn, "memories", "last_accessed_at")
    if not has_strength and not has_last_accessed:
        # Drop the index too in case it was created out of band.
        conn.execute("DROP INDEX IF EXISTS idx_memories_last_accessed")
        return

    fk_state = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("DROP INDEX IF EXISTS idx_memories_last_accessed")
        conn.execute("DROP TABLE IF EXISTS memories_old_004")
        conn.executescript(_PRE_004_CREATE)
        cols_csv = ", ".join(_PRE_004_COLUMNS)
        conn.execute(
            f"INSERT INTO memories_old_004 ({cols_csv}) "
            f"SELECT {cols_csv} FROM memories"
        )
        # Drop dependent indexes / triggers that name `memories` so the
        # rename below succeeds. They're recreated by re-running the
        # migrations that created them (001 + 003).
        conn.execute("DROP INDEX IF EXISTS idx_memories_type")
        conn.execute("DROP INDEX IF EXISTS idx_memories_score")
        conn.execute("DROP INDEX IF EXISTS idx_memories_valid")
        conn.execute("DROP INDEX IF EXISTS idx_memories_content_hash")
        conn.execute("DROP TRIGGER IF EXISTS memories_fts_ai")
        conn.execute("DROP TRIGGER IF EXISTS memories_fts_ad")
        conn.execute("DROP TRIGGER IF EXISTS memories_fts_au")
        conn.execute("DROP TABLE memories")
        conn.execute("ALTER TABLE memories_old_004 RENAME TO memories")
        # Recreate the indexes from 001_initial.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_type "
            "ON memories(type, archived_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_score "
            "ON memories(score DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_valid "
            "ON memories(valid_from, valid_to)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_content_hash "
            "ON memories(content_hash) WHERE archived_at IS NULL"
        )
        # Recreate the FTS5 triggers from 003 if the FTS table still
        # exists. They were dropped above so the table rename could
        # proceed without locking on the OLD reference.
        fts_exists = conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='memories_fts'"
        ).fetchone()
        if fts_exists:
            conn.executescript(
                """
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
            )
    finally:
        conn.execute(f"PRAGMA foreign_keys = {'ON' if fk_state else 'OFF'}")
