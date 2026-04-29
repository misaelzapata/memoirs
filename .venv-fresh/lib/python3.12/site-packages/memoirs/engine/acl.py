"""Per-memory access-control rules + share helpers.

Companion to migration 008 (scoping). Three concerns live here:

1. **Read / write predicates** — :func:`can_read` / :func:`can_write` consult
   the memory's ``user_id`` + ``visibility`` columns and the ``memory_share``
   table to decide whether a given :class:`~memoirs.models.Scope` is allowed
   to see or mutate the row.

2. **Redaction** — :func:`redact_for_requester` returns a copy of the memory
   with ``content`` replaced by ``"<redacted>"`` when ``can_read`` is False.
   Useful for callers that need to surface "you don't have access" without
   leaking the body.

3. **Share management** — :func:`share_memory` / :func:`unshare_memory` /
   :func:`list_shares` thinly wrap the ``memory_share`` table.

Visibility semantics (mirroring GAP P3-3):

* ``private``  — only the owner (``user_id``) can read.
* ``shared``   — owner OR a user_id present in ``memory_share`` for that
                 memory_id.
* ``org``      — any user_id (placeholder; future iterations may validate
                 an org_id once that column exists).
* ``public``   — any user_id.

Writes are always restricted to the owner regardless of visibility.
"""
from __future__ import annotations

import sqlite3
from typing import Any

from ..core.ids import utc_now
from ..models import DEFAULT_SCOPE, Scope


# Sentinel string surfaced in place of memory.content when the requester
# can't read it. Kept short so callers can pattern-match on it.
REDACTED_PLACEHOLDER = "<redacted>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_owner(memory: dict) -> str:
    """Resolve the owner user_id of a memory dict.

    Falls back to the ``"local"`` default for legacy rows that pre-date
    migration 008 (their PRAGMA-default would be ``"local"`` too, but
    fixtures sometimes hand us bare dicts without the column).
    """
    return memory.get("user_id") or "local"


def _row_visibility(memory: dict) -> str:
    return memory.get("visibility") or "private"


def _shared_with(conn: sqlite3.Connection, memory_id: str, user_id: str) -> bool:
    """Return True iff a (memory_id, user_id) row exists in memory_share."""
    row = conn.execute(
        "SELECT 1 FROM memory_share "
        "WHERE memory_id = ? AND shared_with_user_id = ? LIMIT 1",
        (memory_id, user_id),
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

def can_read(
    memory: dict,
    requester: Scope = DEFAULT_SCOPE,
    *,
    conn: sqlite3.Connection | None = None,
) -> bool:
    """Return True if ``requester`` is allowed to read ``memory``.

    ``memory`` is the row dict (must contain ``id``, ``user_id``,
    ``visibility``). ``conn`` is required only when visibility is
    ``shared`` and the requester is not the owner — that branch needs to
    consult ``memory_share``. Other visibilities are answered without DB
    access.
    """
    owner = _row_owner(memory)
    visibility = _row_visibility(memory)
    if visibility == "public":
        return True
    if visibility == "org":
        # Placeholder: any user_id is ok. Future versions may compare
        # org_ids once that field is plumbed through Scope.
        return True
    if requester.user_id == owner:
        return True
    if visibility == "shared":
        if conn is None:
            # Without a connection we cannot look up the share table; be
            # conservative and refuse (callers that enforce ACLs always
            # pass conn=db.conn).
            return False
        memory_id = memory.get("id")
        if not memory_id:
            return False
        return _shared_with(conn, memory_id, requester.user_id)
    # visibility == 'private' (or unknown) and requester != owner.
    return False


def can_write(
    memory: dict,
    requester: Scope = DEFAULT_SCOPE,
) -> bool:
    """Mutations are always restricted to the owner.

    Sharing grants read-only access by design; if a future spec needs
    "shared write" semantics it should add a dedicated column to
    ``memory_share`` rather than overloading this function.
    """
    return requester.user_id == _row_owner(memory)


def redact_for_requester(
    memory: dict,
    requester: Scope = DEFAULT_SCOPE,
    *,
    conn: sqlite3.Connection | None = None,
) -> dict:
    """Return a shallow-copy of ``memory`` with ``content`` redacted when
    the requester can't read it. The original dict is left untouched so
    callers can keep using it for audit / logging.
    """
    if can_read(memory, requester, conn=conn):
        return memory
    redacted = dict(memory)
    redacted["content"] = REDACTED_PLACEHOLDER
    redacted["redacted"] = True
    return redacted


# ---------------------------------------------------------------------------
# Share management
# ---------------------------------------------------------------------------

def share_memory(
    db: Any,
    memory_id: str,
    target_user_id: str,
) -> dict:
    """Grant ``target_user_id`` read-access to ``memory_id``.

    Idempotent: re-issuing the same share is a no-op (composite PK on
    ``memory_share`` absorbs duplicates). Returns a small payload with
    ``shared`` (True if a new row was inserted, False if it already
    existed) so callers / CLI commands can report meaningfully.
    """
    if not memory_id or not target_user_id:
        raise ValueError("memory_id and target_user_id are required")
    conn = db.conn if hasattr(db, "conn") else db
    existing = _shared_with(conn, memory_id, target_user_id)
    if existing:
        return {
            "ok": True,
            "shared": False,
            "memory_id": memory_id,
            "shared_with_user_id": target_user_id,
        }
    conn.execute(
        "INSERT INTO memory_share (memory_id, shared_with_user_id, granted_at) "
        "VALUES (?, ?, ?)",
        (memory_id, target_user_id, utc_now()),
    )
    conn.commit()
    return {
        "ok": True,
        "shared": True,
        "memory_id": memory_id,
        "shared_with_user_id": target_user_id,
    }


def unshare_memory(
    db: Any,
    memory_id: str,
    target_user_id: str,
) -> dict:
    """Revoke a previously-granted share. No-op if the grant didn't exist."""
    if not memory_id or not target_user_id:
        raise ValueError("memory_id and target_user_id are required")
    conn = db.conn if hasattr(db, "conn") else db
    cur = conn.execute(
        "DELETE FROM memory_share "
        "WHERE memory_id = ? AND shared_with_user_id = ?",
        (memory_id, target_user_id),
    )
    conn.commit()
    return {
        "ok": True,
        "removed": int(cur.rowcount),
        "memory_id": memory_id,
        "shared_with_user_id": target_user_id,
    }


def list_shares(
    db: Any,
    memory_id: str,
) -> list[dict]:
    """Return the share grants for ``memory_id`` ordered by ``granted_at``."""
    conn = db.conn if hasattr(db, "conn") else db
    rows = conn.execute(
        "SELECT memory_id, shared_with_user_id, granted_at "
        "FROM memory_share WHERE memory_id = ? "
        "ORDER BY granted_at ASC",
        (memory_id,),
    ).fetchall()
    return [dict(r) for r in rows]
