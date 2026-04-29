"""Migration 005: tool-call memory (P1-8 GAP).

Tool-call memorization is a first-class feature in Letta and Claude Memory:
each agent invocation of a tool — what tool, what args, what result — becomes
a retrievable memory. This migration adds four columns to ``memories`` plus
an index, so that ``record_tool_call`` (engine API) can persist these calls
under ``type = 'tool_call'`` without dragging the existing rows.

Columns added (all nullable — they are meaningful only when ``type = 'tool_call'``):

- ``tool_name``        — TEXT. Name of the invoked tool (e.g. ``bash``,
                         ``Read``, ``mcp__memoirs__mcp_get_context``).
- ``tool_args_json``   — TEXT (JSON). Compact JSON of the args dict. Big
                         payloads should be summarized by the caller; the
                         raw result lives only as a hash.
- ``tool_result_hash`` — TEXT. ``sha256(result)[:16]`` so we can detect
                         repeat calls without storing huge outputs.
- ``tool_status``      — TEXT. One of ``success`` / ``error`` /
                         ``cancelled``. Mirrors what the agent observed.

Plus a partial index ``idx_memories_tool_name`` over rows where
``tool_name IS NOT NULL`` so ``GROUP BY tool_name`` for stats is O(distinct
tools) on a large corpus.

Rollback rebuilds ``memories`` from a SELECT that excludes the four new
columns — same technique used by 004_memory_strength. Pre-existing indexes,
triggers (FTS5), and FK references are recreated after the rename.
"""
from __future__ import annotations

import sqlite3


# Columns of `memories` BEFORE this migration (post-004 shape).
_PRE_005_COLUMNS = (
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
    "strength",
    "last_accessed_at",
)

# Verbatim CREATE TABLE for the pre-005 shape — used by down() to reconstruct
# the table during rollback. Mirrors 001_initial + 004 ALTERs.
_PRE_005_CREATE = """
CREATE TABLE memories_old_005 (
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
    superseded_by TEXT REFERENCES memories_old_005(id),
    archived_at TEXT,
    archive_reason TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    strength REAL NOT NULL DEFAULT 1.0,
    last_accessed_at TEXT
)
"""


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)


def up(conn: sqlite3.Connection) -> None:
    # ALTER TABLE ADD COLUMN — guarded for idempotency.
    if not _has_column(conn, "memories", "tool_name"):
        conn.execute("ALTER TABLE memories ADD COLUMN tool_name TEXT")
    if not _has_column(conn, "memories", "tool_args_json"):
        conn.execute("ALTER TABLE memories ADD COLUMN tool_args_json TEXT")
    if not _has_column(conn, "memories", "tool_result_hash"):
        conn.execute("ALTER TABLE memories ADD COLUMN tool_result_hash TEXT")
    if not _has_column(conn, "memories", "tool_status"):
        conn.execute("ALTER TABLE memories ADD COLUMN tool_status TEXT")

    # Partial index — only rows with a tool_name pay storage cost.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_tool_name "
        "ON memories(tool_name) WHERE tool_name IS NOT NULL"
    )


def down(conn: sqlite3.Connection) -> None:
    """Rebuild `memories` without the four tool_* columns + drop the index.

    SQLite < 3.35 cannot DROP COLUMN; the table-rebuild path is safe on every
    supported version. Triggers and indexes that name `memories` are
    recreated after the rename, mirroring 004's down().
    """
    has_any = (
        _has_column(conn, "memories", "tool_name")
        or _has_column(conn, "memories", "tool_args_json")
        or _has_column(conn, "memories", "tool_result_hash")
        or _has_column(conn, "memories", "tool_status")
    )
    if not has_any:
        conn.execute("DROP INDEX IF EXISTS idx_memories_tool_name")
        return

    fk_state = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("DROP INDEX IF EXISTS idx_memories_tool_name")
        conn.execute("DROP TABLE IF EXISTS memories_old_005")
        conn.executescript(_PRE_005_CREATE)
        cols_csv = ", ".join(_PRE_005_COLUMNS)
        conn.execute(
            f"INSERT INTO memories_old_005 ({cols_csv}) "
            f"SELECT {cols_csv} FROM memories"
        )
        # Drop dependent indexes / triggers that name `memories` so the
        # rename below succeeds. They get recreated below from the same
        # definitions used in 001 + 003 + 004.
        conn.execute("DROP INDEX IF EXISTS idx_memories_type")
        conn.execute("DROP INDEX IF EXISTS idx_memories_score")
        conn.execute("DROP INDEX IF EXISTS idx_memories_valid")
        conn.execute("DROP INDEX IF EXISTS idx_memories_content_hash")
        conn.execute("DROP INDEX IF EXISTS idx_memories_last_accessed")
        conn.execute("DROP TRIGGER IF EXISTS memories_fts_ai")
        conn.execute("DROP TRIGGER IF EXISTS memories_fts_ad")
        conn.execute("DROP TRIGGER IF EXISTS memories_fts_au")
        conn.execute("DROP TABLE memories")
        conn.execute("ALTER TABLE memories_old_005 RENAME TO memories")
        # Recreate indexes from 001_initial.
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
        # Recreate the index from 004.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_last_accessed "
            "ON memories(last_accessed_at)"
        )
        # Recreate FTS5 triggers from 003 if the FTS table still exists.
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
