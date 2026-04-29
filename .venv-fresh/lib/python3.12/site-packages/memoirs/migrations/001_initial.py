"""Migration 001: baseline schema.

Captures the full schema as it existed before the versioned migration system
was introduced. The script is intentionally idempotent — every ``CREATE``
uses ``IF NOT EXISTS`` — so running it against a database that already had
the legacy schema (created via the old ``MemoirsDB.init()`` codepath) is a
no-op apart from setting ``PRAGMA user_version = 1``.
"""
from __future__ import annotations

import sqlite3


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uri TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    content_hash TEXT,
    mtime_ns INTEGER,
    size_bytes INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    external_id TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT,
    updated_at TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(source_id, external_id)
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    external_id TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    created_at TEXT,
    content_hash TEXT NOT NULL,
    raw_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    is_active INTEGER NOT NULL DEFAULT 1,
    first_seen_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(conversation_id, ordinal)
);

CREATE TABLE IF NOT EXISTS attachments (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    uri TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS import_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_uri TEXT NOT NULL,
    importer TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    file_mtime_ns INTEGER,
    file_size INTEGER,
    content_hash TEXT,
    conversation_count INTEGER NOT NULL DEFAULT 0,
    message_count INTEGER NOT NULL DEFAULT 0,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, ordinal);

CREATE INDEX IF NOT EXISTS idx_messages_content_hash
    ON messages(content_hash);

CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_external
    ON messages(conversation_id, external_id)
    WHERE external_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_import_runs_source
    ON import_runs(source_uri, started_at);

-- ============================================================
-- Layer 2: extracted memory candidates (Gemma output)
-- ============================================================
CREATE TABLE IF NOT EXISTS memory_candidates (
    id TEXT PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    source_message_ids TEXT NOT NULL DEFAULT '[]',  -- JSON array of message ids
    type TEXT NOT NULL,                              -- preference|fact|project|task|decision|style|credential_pointer
    content TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 3,           -- 1..5
    confidence REAL NOT NULL DEFAULT 0.5,            -- 0..1
    entities TEXT NOT NULL DEFAULT '[]',             -- JSON array of strings
    status TEXT NOT NULL DEFAULT 'pending',          -- pending|accepted|rejected|merged
    rejection_reason TEXT,
    extractor TEXT,                                  -- e.g. 'gemma-2-2b'
    raw_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    promoted_memory_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_candidates_status ON memory_candidates(status, created_at);
CREATE INDEX IF NOT EXISTS idx_candidates_conv ON memory_candidates(conversation_id);

-- ============================================================
-- Layer 5: consolidated memories
-- ============================================================
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 3,
    confidence REAL NOT NULL DEFAULT 0.5,
    score REAL NOT NULL DEFAULT 0.0,
    usage_count INTEGER NOT NULL DEFAULT 0,
    last_used_at TEXT,
    user_signal REAL NOT NULL DEFAULT 0.0,           -- explicit "remember this" boost
    valid_from TEXT,
    valid_to TEXT,                                   -- NULL = current
    superseded_by TEXT REFERENCES memories(id),
    archived_at TEXT,
    archive_reason TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type, archived_at);
CREATE INDEX IF NOT EXISTS idx_memories_score ON memories(score DESC);
CREATE INDEX IF NOT EXISTS idx_memories_valid ON memories(valid_from, valid_to);
CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_content_hash
    ON memories(content_hash) WHERE archived_at IS NULL;

-- Layer 4 stub: embeddings stored as BLOB. Vector search uses sqlite-vec extension if loaded.
CREATE TABLE IF NOT EXISTS memory_embeddings (
    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    dim INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- ============================================================
-- Layer 3: structured knowledge (entities + relationships)
-- ============================================================
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    type TEXT,                                       -- person|tool|project|concept|...
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(normalized_name, type)
);
CREATE INDEX IF NOT EXISTS idx_entities_norm ON entities(normalized_name);

CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    source_entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE(source_entity_id, relation, target_entity_id)
);
CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_entity_id);

CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (memory_id, entity_id)
);

-- ============================================================
-- Layer 5.7: event queue (durable for event-driven ingestion)
-- ============================================================
CREATE TABLE IF NOT EXISTS event_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,                        -- chat_message|conversation_closed|file_imported|task_completed|project_updated|manual_memory_added
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',          -- pending|processing|done|failed
    error TEXT,
    created_at TEXT NOT NULL,
    processed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_eventq_status ON event_queue(status, created_at);
"""


# Tables created by this migration. Used by ``down()`` and by tests that
# need to compare a fresh DB against the snapshot.
TABLES = (
    "sources",
    "conversations",
    "messages",
    "attachments",
    "import_runs",
    "memory_candidates",
    "memories",
    "memory_embeddings",
    "entities",
    "relationships",
    "memory_entities",
    "event_queue",
)


def up(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def down(conn: sqlite3.Connection) -> None:
    """Drop every table created by ``up()``.

    Destructive — only meaningful in tests / dev. Foreign keys are turned off
    for the duration so the order of drops does not matter.
    """
    fk_state = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        for table in TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
    finally:
        conn.execute(f"PRAGMA foreign_keys = {'ON' if fk_state else 'OFF'}")
