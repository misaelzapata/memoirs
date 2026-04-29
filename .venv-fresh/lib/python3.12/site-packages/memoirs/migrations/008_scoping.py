"""Migration 008: multi-tenant scoping + per-memory ACL (P0-5 / P3-3).

Adds the columns needed to know **who** wrote a memory / candidate / convo
and **who** is allowed to read it back, plus a ``memory_share`` table for
explicit per-memory grants.

Schema additions
----------------
``memories``::
    user_id     TEXT NOT NULL DEFAULT 'local'
    agent_id    TEXT
    run_id      TEXT
    namespace   TEXT
    visibility  TEXT NOT NULL DEFAULT 'private'    -- private|shared|org|public

``conversations``::
    user_id     TEXT NOT NULL DEFAULT 'local'
    agent_id    TEXT

``memory_candidates``::
    user_id     TEXT NOT NULL DEFAULT 'local'

Indexes::
    idx_memories_user_id
    idx_memories_namespace      (partial WHERE namespace IS NOT NULL)
    idx_memories_visibility
    idx_conversations_user_id

New table::
    memory_share (
      memory_id TEXT,
      shared_with_user_id TEXT,
      granted_at TEXT,
      PRIMARY KEY (memory_id, shared_with_user_id)
    )

Default values are chosen so the single-user local-first behaviour continues
unchanged: every existing row is implicitly authored by ``user_id='local'``
with ``visibility='private'``. Multi-tenant deployments opt in by setting
the columns explicitly at write time (see :mod:`memoirs.engine.acl` and the
``Scope`` dataclass in :mod:`memoirs.models`).

Rollback uses the SQLite table-rebuild technique (same approach used by
004 / 005) since SQLite < 3.35 cannot ``DROP COLUMN``.
"""
from __future__ import annotations

import sqlite3


# ---------------------------------------------------------------------------
# Pre-008 column lists for the rebuild path in down()
# ---------------------------------------------------------------------------

# memories — pre-008 = post-005 shape (matches 005's _PRE_005_COLUMNS plus the
# four tool_* columns). Used to copy data back when rolling back.
_PRE_008_MEMORY_COLUMNS = (
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
    "tool_name",
    "tool_args_json",
    "tool_result_hash",
    "tool_status",
)

_PRE_008_MEMORY_CREATE = """
CREATE TABLE memories_old_008 (
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
    superseded_by TEXT REFERENCES memories_old_008(id),
    archived_at TEXT,
    archive_reason TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    strength REAL NOT NULL DEFAULT 1.0,
    last_accessed_at TEXT,
    tool_name TEXT,
    tool_args_json TEXT,
    tool_result_hash TEXT,
    tool_status TEXT
)
"""

_PRE_008_CONVERSATION_COLUMNS = (
    "id",
    "source_id",
    "external_id",
    "title",
    "created_at",
    "updated_at",
    "message_count",
    "metadata_json",
)

_PRE_008_CONVERSATION_CREATE = """
CREATE TABLE conversations_old_008 (
    id TEXT PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    external_id TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT,
    updated_at TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(source_id, external_id)
)
"""

_PRE_008_CANDIDATE_COLUMNS = (
    "id",
    "conversation_id",
    "source_message_ids",
    "type",
    "content",
    "importance",
    "confidence",
    "entities",
    "status",
    "rejection_reason",
    "extractor",
    "raw_json",
    "created_at",
    "updated_at",
    "promoted_memory_id",
)

_PRE_008_CANDIDATE_CREATE = """
CREATE TABLE memory_candidates_old_008 (
    id TEXT PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    source_message_ids TEXT NOT NULL DEFAULT '[]',
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 3,
    confidence REAL NOT NULL DEFAULT 0.5,
    entities TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'pending',
    rejection_reason TEXT,
    extractor TEXT,
    raw_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    promoted_memory_id TEXT
)
"""


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)


# ---------------------------------------------------------------------------
# up()
# ---------------------------------------------------------------------------

def up(conn: sqlite3.Connection) -> None:
    # memories — five new columns. Each guarded for idempotency so a re-run
    # against a partially-applied DB doesn't error.
    if not _has_column(conn, "memories", "user_id"):
        conn.execute(
            "ALTER TABLE memories ADD COLUMN user_id TEXT NOT NULL DEFAULT 'local'"
        )
    if not _has_column(conn, "memories", "agent_id"):
        conn.execute("ALTER TABLE memories ADD COLUMN agent_id TEXT")
    if not _has_column(conn, "memories", "run_id"):
        conn.execute("ALTER TABLE memories ADD COLUMN run_id TEXT")
    if not _has_column(conn, "memories", "namespace"):
        conn.execute("ALTER TABLE memories ADD COLUMN namespace TEXT")
    if not _has_column(conn, "memories", "visibility"):
        conn.execute(
            "ALTER TABLE memories ADD COLUMN visibility TEXT NOT NULL "
            "DEFAULT 'private'"
        )

    # conversations — owner + producing agent.
    if not _has_column(conn, "conversations", "user_id"):
        conn.execute(
            "ALTER TABLE conversations ADD COLUMN user_id TEXT NOT NULL "
            "DEFAULT 'local'"
        )
    if not _has_column(conn, "conversations", "agent_id"):
        conn.execute("ALTER TABLE conversations ADD COLUMN agent_id TEXT")

    # memory_candidates — owner only (visibility is decided at promotion time).
    if not _has_column(conn, "memory_candidates", "user_id"):
        conn.execute(
            "ALTER TABLE memory_candidates ADD COLUMN user_id TEXT NOT NULL "
            "DEFAULT 'local'"
        )

    # Indexes — covers the most common scoped lookups.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_namespace "
        "ON memories(namespace) WHERE namespace IS NOT NULL"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_visibility "
        "ON memories(visibility)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_user_id "
        "ON conversations(user_id)"
    )

    # Per-memory share table. Composite PK ensures a (memory, user) grant
    # is naturally idempotent — re-issuing the same share is a no-op.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_share (
            memory_id TEXT NOT NULL,
            shared_with_user_id TEXT NOT NULL,
            granted_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (memory_id, shared_with_user_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_share_user "
        "ON memory_share(shared_with_user_id)"
    )


# ---------------------------------------------------------------------------
# down()
# ---------------------------------------------------------------------------

def down(conn: sqlite3.Connection) -> None:
    """Rebuild affected tables without the new columns + drop the share table.

    Same SQLite table-rebuild technique used by 004 / 005:
    * disable FKs
    * drop indexes / triggers that reference the table
    * recreate the old shape under a temp name
    * copy data
    * drop the new table, rename old → new
    * recreate indexes / triggers
    """
    # If nothing applied, just clean up the share table + indexes and bail.
    has_anything = (
        _has_column(conn, "memories", "user_id")
        or _has_column(conn, "memories", "visibility")
        or _has_column(conn, "conversations", "user_id")
        or _has_column(conn, "memory_candidates", "user_id")
    )
    if not has_anything:
        for idx in (
            "idx_memories_user_id",
            "idx_memories_namespace",
            "idx_memories_visibility",
            "idx_conversations_user_id",
            "idx_memory_share_user",
        ):
            conn.execute(f"DROP INDEX IF EXISTS {idx}")
        conn.execute("DROP TABLE IF EXISTS memory_share")
        return

    fk_state = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        # ----- memory_share + the 4 008 indexes -----
        for idx in (
            "idx_memories_user_id",
            "idx_memories_namespace",
            "idx_memories_visibility",
            "idx_conversations_user_id",
            "idx_memory_share_user",
        ):
            conn.execute(f"DROP INDEX IF EXISTS {idx}")
        conn.execute("DROP TABLE IF EXISTS memory_share")

        # ----- memories rebuild -----
        if _has_column(conn, "memories", "user_id"):
            conn.execute("DROP TABLE IF EXISTS memories_old_008")
            conn.executescript(_PRE_008_MEMORY_CREATE)
            cols_csv = ", ".join(_PRE_008_MEMORY_COLUMNS)
            conn.execute(
                f"INSERT INTO memories_old_008 ({cols_csv}) "
                f"SELECT {cols_csv} FROM memories"
            )
            # Drop dependent indexes / triggers that name `memories`.
            for idx in (
                "idx_memories_type",
                "idx_memories_score",
                "idx_memories_valid",
                "idx_memories_content_hash",
                "idx_memories_last_accessed",
                "idx_memories_tool_name",
            ):
                conn.execute(f"DROP INDEX IF EXISTS {idx}")
            for trg in ("memories_fts_ai", "memories_fts_ad", "memories_fts_au"):
                conn.execute(f"DROP TRIGGER IF EXISTS {trg}")
            conn.execute("DROP TABLE memories")
            conn.execute("ALTER TABLE memories_old_008 RENAME TO memories")
            # Recreate indexes from 001 + 004 + 005.
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_last_accessed "
                "ON memories(last_accessed_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_tool_name "
                "ON memories(tool_name) WHERE tool_name IS NOT NULL"
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

        # ----- conversations rebuild -----
        if _has_column(conn, "conversations", "user_id"):
            conn.execute("DROP TABLE IF EXISTS conversations_old_008")
            conn.executescript(_PRE_008_CONVERSATION_CREATE)
            cols_csv = ", ".join(_PRE_008_CONVERSATION_COLUMNS)
            conn.execute(
                f"INSERT INTO conversations_old_008 ({cols_csv}) "
                f"SELECT {cols_csv} FROM conversations"
            )
            conn.execute("DROP TABLE conversations")
            conn.execute(
                "ALTER TABLE conversations_old_008 RENAME TO conversations"
            )

        # ----- memory_candidates rebuild -----
        if _has_column(conn, "memory_candidates", "user_id"):
            conn.execute("DROP TABLE IF EXISTS memory_candidates_old_008")
            conn.executescript(_PRE_008_CANDIDATE_CREATE)
            cols_csv = ", ".join(_PRE_008_CANDIDATE_COLUMNS)
            conn.execute(
                f"INSERT INTO memory_candidates_old_008 ({cols_csv}) "
                f"SELECT {cols_csv} FROM memory_candidates"
            )
            for idx in ("idx_candidates_status", "idx_candidates_conv"):
                conn.execute(f"DROP INDEX IF EXISTS {idx}")
            conn.execute("DROP TABLE memory_candidates")
            conn.execute(
                "ALTER TABLE memory_candidates_old_008 RENAME TO memory_candidates"
            )
            # Recreate the indexes from 001_initial.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_candidates_status "
                "ON memory_candidates(status, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_candidates_conv "
                "ON memory_candidates(conversation_id)"
            )
    finally:
        conn.execute(f"PRAGMA foreign_keys = {'ON' if fk_state else 'OFF'}")
