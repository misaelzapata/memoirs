"""Conflict resolution surface for memoirs (P5-2).

The sleep-time ``contradictions`` job detects pairs of memorias that look
contradictory but cannot resolve them safely — the right answer is always
human-in-the-loop. This module turns those detections into durable rows in
``memory_conflicts`` and provides the small set of operations the UI / CLI
need to triage them.

Public API
~~~~~~~~~~
- :func:`record_conflict` — idempotent INSERT (UNIQUE pair) that refreshes
  similarity / detector / reason on re-detection while preserving the
  pending status.
- :func:`list_conflicts` — JOIN with ``memories`` so the inspector can
  render contents without N+1 queries.
- :func:`get_conflict` — single row + both memory contents.
- :func:`resolve_conflict` — apply one of the documented actions and bump
  the row's status.

Status values
~~~~~~~~~~~~~
``pending``           — fresh detection, awaiting human triage.
``resolved_keep_a``   — keep memory A, archive memory B.
``resolved_keep_b``   — keep memory B, archive memory A.
``resolved_keep_both``— keep both (false positive); just close the row.
``resolved_merge``    — synthesise a merged summary, archive both, insert
                        the new memoria; the resolution_notes captures the
                        new memory id.
``dismissed``         — close the row without touching memorias.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any, Optional

from ..core.ids import content_hash, stable_id, utc_now
from ..db import MemoirsDB

log = logging.getLogger("memoirs.conflicts")


# Public set of actions accepted by ``resolve_conflict``.
RESOLVE_ACTIONS = (
    "keep_a",
    "keep_b",
    "keep_both",
    "merge",
    "dismiss",
)

_ACTION_TO_STATUS = {
    "keep_a": "resolved_keep_a",
    "keep_b": "resolved_keep_b",
    "keep_both": "resolved_keep_both",
    "merge": "resolved_merge",
    "dismiss": "dismissed",
}


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _normalise_pair(a: str, b: str) -> tuple[str, str]:
    """Return the pair in canonical (sorted) order so the UNIQUE index is
    independent of which side was "A" at detection time."""
    return (a, b) if a <= b else (b, a)


def record_conflict(
    db: MemoirsDB,
    *,
    memory_a_id: str,
    memory_b_id: str,
    similarity: Optional[float] = None,
    detector: Optional[str] = None,
    reason: Optional[str] = None,
) -> int:
    """Persist (or refresh) a contradiction between two memorias.

    Returns the row id. Re-detection of the same pair updates ``similarity``,
    ``detector``, ``reason`` and ``detected_at`` but **preserves** the
    existing ``status`` so the user's resolution does not get overwritten.
    """
    if not memory_a_id or not memory_b_id:
        raise ValueError("memory_a_id and memory_b_id are required")
    if memory_a_id == memory_b_id:
        raise ValueError("a memory cannot conflict with itself")
    a, b = _normalise_pair(memory_a_id, memory_b_id)
    now = utc_now()
    with db.conn:
        db.conn.execute(
            """
            INSERT INTO memory_conflicts (
                memory_a_id, memory_b_id, similarity, detected_at,
                detector, reason, status
            )
            VALUES (?, ?, ?, ?, ?, ?, 'pending')
            ON CONFLICT(memory_a_id, memory_b_id) DO UPDATE SET
                similarity = excluded.similarity,
                detected_at = excluded.detected_at,
                detector = excluded.detector,
                reason = excluded.reason
            """,
            (a, b, similarity, now, detector, reason),
        )
    row = db.conn.execute(
        "SELECT id FROM memory_conflicts WHERE memory_a_id = ? AND memory_b_id = ?",
        (a, b),
    ).fetchone()
    return int(row["id"]) if row else 0


def list_conflicts(
    db: MemoirsDB,
    *,
    status: Optional[str] = "pending",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return the most recent conflicts (optionally filtered by status).

    Joins ``memories`` so callers get both contents in a single query. When
    a memory has been hard-deleted (rare; archived rows still join) the
    fields appear as ``None`` rather than dropping the conflict row.
    """
    limit = max(1, min(500, int(limit)))
    where = []
    params: list[Any] = []
    if status is not None:
        where.append("c.status = ?")
        params.append(status)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    sql = (
        "SELECT c.id, c.memory_a_id, c.memory_b_id, c.similarity, "
        "       c.detected_at, c.detector, c.reason, c.status, "
        "       c.resolution_notes, c.resolved_at, "
        "       ma.type AS a_type, ma.content AS a_content, "
        "       ma.archived_at AS a_archived_at, "
        "       mb.type AS b_type, mb.content AS b_content, "
        "       mb.archived_at AS b_archived_at "
        "  FROM memory_conflicts c "
        "  LEFT JOIN memories ma ON ma.id = c.memory_a_id "
        "  LEFT JOIN memories mb ON mb.id = c.memory_b_id "
        + where_sql +
        " ORDER BY c.detected_at DESC, c.id DESC LIMIT ?"
    )
    params.append(limit)
    rows = db.conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_conflict(db: MemoirsDB, conflict_id: int) -> Optional[dict[str, Any]]:
    """Single conflict row joined with both memorias' content."""
    row = db.conn.execute(
        "SELECT c.*, "
        "       ma.type AS a_type, ma.content AS a_content, "
        "       ma.archived_at AS a_archived_at, "
        "       mb.type AS b_type, mb.content AS b_content, "
        "       mb.archived_at AS b_archived_at "
        "  FROM memory_conflicts c "
        "  LEFT JOIN memories ma ON ma.id = c.memory_a_id "
        "  LEFT JOIN memories mb ON mb.id = c.memory_b_id "
        " WHERE c.id = ?",
        (int(conflict_id),),
    ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Mutation helpers
# ---------------------------------------------------------------------------


def _archive_memory(db: MemoirsDB, memory_id: str, *, reason: str, now: str) -> bool:
    cur = db.conn.execute(
        "UPDATE memories SET archived_at = ?, archive_reason = ?, updated_at = ? "
        "WHERE id = ? AND archived_at IS NULL",
        (now, reason, now, memory_id),
    )
    return bool(cur.rowcount)


def _summarise_merge(a_content: str, b_content: str) -> str:
    """Best-effort merged summary.

    Tries the curator LLM first and falls back to a deterministic
    concatenation when the model is unavailable. The fallback is
    intentionally simple — it keeps tests hermetic and never leaves the
    user with an empty memoria.
    """
    try:
        from . import curator as _curator
        if _curator._have_curator():
            try:
                summary = _curator.curator_summarize([
                    {"role": "user", "content": a_content},
                    {"role": "user", "content": b_content},
                ])
                if summary:
                    return summary[:500]
            except Exception:  # noqa: BLE001
                log.exception("conflict merge: curator_summarize failed")
    except Exception:  # noqa: BLE001 - import guard
        pass
    a = (a_content or "").strip()
    b = (b_content or "").strip()
    if a and b:
        return f"Merged: {a} / {b}"[:500]
    return (a or b or "Merged memory")[:500]


def _create_merged_memory(
    db: MemoirsDB,
    *,
    a_type: str,
    b_type: str,
    a_content: str,
    b_content: str,
    now: str,
) -> str:
    """Insert a fresh memoria carrying the merged summary. Returns its id."""
    merged_type = a_type or b_type or "fact"
    merged_content = _summarise_merge(a_content or "", b_content or "")
    mid = stable_id("mem", merged_type, merged_content, now)
    h = content_hash(merged_content)
    db.conn.execute(
        """
        INSERT INTO memories (
            id, type, content, content_hash, importance, confidence,
            score, usage_count, user_signal, valid_from, metadata_json,
            created_at, updated_at
        )
        VALUES (?, ?, ?, ?, 3, 0.6, 0, 0, 0, ?, '{}', ?, ?)
        ON CONFLICT(content_hash) WHERE archived_at IS NULL DO NOTHING
        """,
        (mid, merged_type, merged_content, h, now, now, now),
    )
    # If a row already shared this content_hash (rare but possible) keep
    # whichever id won the race.
    row = db.conn.execute(
        "SELECT id FROM memories WHERE content_hash = ? AND archived_at IS NULL",
        (h,),
    ).fetchone()
    return row["id"] if row else mid


def resolve_conflict(
    db: MemoirsDB,
    conflict_id: int,
    *,
    action: str,
    notes: Optional[str] = None,
) -> dict[str, Any]:
    """Apply a resolution action to a conflict.

    Returns a small report describing what changed (which memoria was
    archived, the new merged memory id, etc.). Raises ``ValueError`` for
    unknown actions or when the conflict has already been resolved (we do
    not re-resolve to keep the audit trail honest — explicit reopen is a
    future migration).
    """
    if action not in RESOLVE_ACTIONS:
        raise ValueError(
            f"unknown action {action!r}; expected one of {list(RESOLVE_ACTIONS)}"
        )
    row = get_conflict(db, conflict_id)
    if not row:
        raise ValueError(f"conflict {conflict_id} not found")
    if row["status"] != "pending":
        raise ValueError(
            f"conflict {conflict_id} already {row['status']}; cannot re-resolve"
        )

    now = utc_now()
    archived: list[str] = []
    new_memory_id: Optional[str] = None
    new_status = _ACTION_TO_STATUS[action]
    a_id = row["memory_a_id"]
    b_id = row["memory_b_id"]

    with db.conn:
        if action == "keep_a":
            if _archive_memory(db, b_id, reason=f"conflict {conflict_id}: keep_a", now=now):
                archived.append(b_id)
        elif action == "keep_b":
            if _archive_memory(db, a_id, reason=f"conflict {conflict_id}: keep_b", now=now):
                archived.append(a_id)
        elif action == "merge":
            new_memory_id = _create_merged_memory(
                db,
                a_type=row.get("a_type") or "",
                b_type=row.get("b_type") or "",
                a_content=row.get("a_content") or "",
                b_content=row.get("b_content") or "",
                now=now,
            )
            for mid in (a_id, b_id):
                if _archive_memory(
                    db, mid,
                    reason=f"conflict {conflict_id}: merged into {new_memory_id}",
                    now=now,
                ):
                    archived.append(mid)
        # keep_both and dismiss: pure status update, no memoria touches.

        # Encode the new memory id at the head of resolution_notes so the UI
        # can surface it without a follow-up roundtrip; user notes follow.
        composed_notes_parts: list[str] = []
        if new_memory_id:
            composed_notes_parts.append(f"merged_memory_id={new_memory_id}")
        if notes:
            composed_notes_parts.append(notes)
        composed_notes = " | ".join(composed_notes_parts) if composed_notes_parts else None

        db.conn.execute(
            "UPDATE memory_conflicts SET status = ?, resolution_notes = ?, "
            "  resolved_at = ? WHERE id = ?",
            (new_status, composed_notes, now, int(conflict_id)),
        )

    return {
        "conflict_id": int(conflict_id),
        "status": new_status,
        "archived": archived,
        "new_memory_id": new_memory_id,
        "notes": composed_notes,
    }


__all__ = [
    "RESOLVE_ACTIONS",
    "record_conflict",
    "list_conflicts",
    "get_conflict",
    "resolve_conflict",
]
