"""Migration 006: RAPTOR-style hierarchical summary tree (P1-6).

Introduces two tables that capture a recursive tree of cluster summaries
over the active memory corpus:

- ``summary_nodes``         — one row per node. Level 0 nodes mirror real
                              memories (leaves). Level >= 1 nodes are LLM /
                              heuristic summaries of their member set. The
                              root has ``parent_id IS NULL`` and the highest
                              level reached.
- ``summary_node_members``  — many-to-many edge table linking a parent
                              summary to either real memories (level 0) or
                              child summary nodes. ``similarity`` is the
                              cosine to the cluster centroid at build time.

Scopes: a tree can be built per ``scope_kind`` ∈ {global, project,
conversation} so callers can drill into a single project / conversation
without rebuilding the whole corpus.

Reference: Sarthi et al. "RAPTOR: Recursive Abstractive Processing for
Tree-Organized Retrieval", ICLR 2024.
"""
from __future__ import annotations

import sqlite3


SCHEMA = """
CREATE TABLE IF NOT EXISTS summary_nodes (
    id TEXT PRIMARY KEY,
    level INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB,
    child_count INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    parent_id TEXT REFERENCES summary_nodes(id) ON DELETE SET NULL,
    scope_kind TEXT,
    scope_id TEXT
);

CREATE TABLE IF NOT EXISTS summary_node_members (
    node_id TEXT NOT NULL,
    member_kind TEXT NOT NULL,
    member_id TEXT NOT NULL,
    similarity REAL,
    PRIMARY KEY (node_id, member_kind, member_id)
);

CREATE INDEX IF NOT EXISTS idx_summary_nodes_parent
    ON summary_nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_summary_nodes_level
    ON summary_nodes(level);
CREATE INDEX IF NOT EXISTS idx_summary_nodes_scope
    ON summary_nodes(scope_kind, scope_id);
CREATE INDEX IF NOT EXISTS idx_summary_node_members_member
    ON summary_node_members(member_kind, member_id);
"""


def up(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def down(conn: sqlite3.Connection) -> None:
    fk_state = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("DROP INDEX IF EXISTS idx_summary_node_members_member")
        conn.execute("DROP INDEX IF EXISTS idx_summary_nodes_scope")
        conn.execute("DROP INDEX IF EXISTS idx_summary_nodes_level")
        conn.execute("DROP INDEX IF EXISTS idx_summary_nodes_parent")
        conn.execute("DROP TABLE IF EXISTS summary_node_members")
        conn.execute("DROP TABLE IF EXISTS summary_nodes")
    finally:
        conn.execute(f"PRAGMA foreign_keys = {'ON' if fk_state else 'OFF'}")
