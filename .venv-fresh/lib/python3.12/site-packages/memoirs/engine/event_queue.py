"""Durable event queue for cross-component async work (P0-4 GAP).

The :data:`event_queue` table (migration 001) was sitting idle: it had a schema
but no producer / consumer. This module fills that gap with a small, focused
API on top of the existing table — every ingester now drops a
``messages_ingested`` row when it finishes, the memory engine drops a
``memory_promoted`` row when it persists a new memoria via the ADD path, and
the sleep scheduler drains the queue with a per-event-type dispatch table.

Design choices:

* **No new schema.** We reuse ``event_queue (id, event_type, payload_json,
  status, error, created_at, processed_at)``. ``status`` cycles through
  ``pending → processing → done|failed`` so concurrent consumers don't double-
  process the same row.
* **Lock via UPDATE.** ``dequeue_batch`` flips the selected rows to
  ``processing`` inside a single transaction. A second concurrent caller will
  not see those rows on its own ``SELECT … WHERE status='pending'`` scan
  because SQLite serializes the writes (busy_timeout = 30s).
* **Soft requeue.** ``requeue_failed`` resets ``status='pending'`` for failed
  rows older than ``max_age_hours``. Useful when a downstream model (Gemma)
  was unavailable temporarily and the queue piled up.
* **Idempotent tests.** Every helper takes ``db`` first, mirroring the rest
  of the engine surface, and uses parameterized SQL.

Coexistence with :mod:`memory_engine`: the legacy
``enqueue_event`` / ``process_event_queue`` helpers there remain for
back-compat (and for the older event types ``conversation_closed`` /
``manual_memory_added`` that have a built-in dispatcher). New callers should
use this module — it exposes a richer surface (``mark_done``, ``mark_failed``,
stats, requeue) and accepts arbitrary handler tables.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from ..core.ids import utc_now
from ..db import MemoirsDB

log = logging.getLogger("memoirs.event_queue")


VALID_STATUSES = ("pending", "processing", "done", "failed")


# ---------------------------------------------------------------------------
# Producer side
# ---------------------------------------------------------------------------


def enqueue(
    db: MemoirsDB,
    *,
    event_type: str,
    payload: dict,
) -> int:
    """Persist a new ``pending`` event and return its id.

    ``payload`` must be JSON-serializable. We commit immediately so producers
    (ingesters, the memory engine, the API surface) aren't bound to the
    consumer's transaction.
    """
    if not isinstance(event_type, str) or not event_type:
        raise ValueError("event_type must be a non-empty string")
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    cur = db.conn.execute(
        "INSERT INTO event_queue (event_type, payload_json, status, created_at) "
        "VALUES (?, ?, 'pending', ?)",
        (event_type, json.dumps(payload, ensure_ascii=False, sort_keys=True), utc_now()),
    )
    db.conn.commit()
    return int(cur.lastrowid)


# ---------------------------------------------------------------------------
# Consumer side
# ---------------------------------------------------------------------------


def dequeue_batch(db: MemoirsDB, *, limit: int = 50) -> list[dict[str, Any]]:
    """Atomically claim up to ``limit`` pending events.

    Returns a list of ``{id, event_type, payload, created_at}`` dicts. The
    rows are flipped to ``status='processing'`` in the same transaction so a
    second concurrent caller never sees the same id.

    Caller is then expected to call :func:`mark_done` or :func:`mark_failed`
    on each row.
    """
    if limit <= 0:
        return []
    out: list[dict[str, Any]] = []
    with db.conn:
        rows = db.conn.execute(
            "SELECT id, event_type, payload_json, created_at "
            "FROM event_queue WHERE status = 'pending' ORDER BY id LIMIT ?",
            (int(limit),),
        ).fetchall()
        for r in rows:
            db.conn.execute(
                "UPDATE event_queue SET status = 'processing' WHERE id = ? AND status = 'pending'",
                (r["id"],),
            )
            try:
                payload = json.loads(r["payload_json"]) if r["payload_json"] else {}
            except json.JSONDecodeError:
                payload = {"_raw": r["payload_json"]}
            out.append(
                {
                    "id": int(r["id"]),
                    "event_type": str(r["event_type"]),
                    "payload": payload,
                    "created_at": r["created_at"],
                }
            )
    return out


def mark_done(
    db: MemoirsDB,
    event_id: int,
    *,
    result: Any = None,
) -> None:
    """Mark an event as ``done``. ``result`` is currently ignored at the
    schema level (no result column) but accepted for API symmetry / future
    use; we log it at DEBUG so operators can correlate via logs.
    """
    if result is not None:
        log.debug("event %s done: %r", event_id, result)
    db.conn.execute(
        "UPDATE event_queue SET status = 'done', processed_at = ?, error = NULL WHERE id = ?",
        (utc_now(), int(event_id)),
    )
    db.conn.commit()


def mark_failed(
    db: MemoirsDB,
    event_id: int,
    *,
    error: str,
) -> None:
    """Mark an event as ``failed`` with the given error string."""
    db.conn.execute(
        "UPDATE event_queue SET status = 'failed', processed_at = ?, error = ? WHERE id = ?",
        (utc_now(), str(error), int(event_id)),
    )
    db.conn.commit()


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


Handler = Callable[[MemoirsDB, dict], Any]


def process_pending(
    db: MemoirsDB,
    *,
    batch_size: int = 50,
    handlers: dict[str, Handler] | None = None,
) -> dict[str, Any]:
    """Drain up to ``batch_size`` pending events through ``handlers``.

    ``handlers`` maps ``event_type`` → callable ``(db, payload) -> Any``. Each
    handler runs inside its own try/except: a failure marks just that row
    ``failed`` and the rest of the batch keeps going.

    Unknown event types (no handler registered) are marked ``done`` with no
    side effects — the queue is durable so we don't want unknown events
    piling up forever; if a downstream component cares, it can re-queue.
    """
    handlers = handlers or {}
    batch = dequeue_batch(db, limit=batch_size)
    processed = 0
    failed = 0
    skipped = 0
    by_type: dict[str, int] = {}
    for ev in batch:
        etype = ev["event_type"]
        by_type[etype] = by_type.get(etype, 0) + 1
        handler = handlers.get(etype)
        if handler is None:
            mark_done(db, ev["id"], result={"skipped": True})
            skipped += 1
            continue
        try:
            result = handler(db, ev["payload"])
            mark_done(db, ev["id"], result=result)
            processed += 1
        except Exception as exc:  # noqa: BLE001 — all handler errors are reported
            log.exception("event_queue handler failed: id=%s type=%s", ev["id"], etype)
            mark_failed(db, ev["id"], error=f"{type(exc).__name__}: {exc}")
            failed += 1
    return {
        "processed": processed,
        "failed": failed,
        "skipped": skipped,
        "by_type": by_type,
        "batch_size": len(batch),
    }


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def requeue_failed(db: MemoirsDB, *, max_age_hours: int = 24) -> int:
    """Reset ``failed`` rows older than ``max_age_hours`` back to ``pending``.

    Returns the number of rows touched. The error message is preserved so
    operators can still see why it failed last time.
    """
    if max_age_hours < 0:
        raise ValueError("max_age_hours must be >= 0")
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=int(max_age_hours))).isoformat(
        timespec="seconds"
    )
    cur = db.conn.execute(
        "UPDATE event_queue SET status = 'pending', processed_at = NULL "
        "WHERE status = 'failed' AND COALESCE(processed_at, created_at) <= ?",
        (cutoff,),
    )
    db.conn.commit()
    return int(cur.rowcount or 0)


def get_stats(db: MemoirsDB) -> dict[str, Any]:
    """Return per-status counts plus the age (in seconds) of the oldest
    pending event. Useful for ``memoirs events stats`` and the MCP surface.
    """
    counts = {s: 0 for s in VALID_STATUSES}
    rows = db.conn.execute(
        "SELECT status, COUNT(*) AS n FROM event_queue GROUP BY status"
    ).fetchall()
    for r in rows:
        counts[str(r["status"])] = int(r["n"])

    oldest_pending_age_seconds: float | None = None
    row = db.conn.execute(
        "SELECT created_at FROM event_queue WHERE status = 'pending' "
        "ORDER BY id ASC LIMIT 1"
    ).fetchone()
    if row and row["created_at"]:
        try:
            ts = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            oldest_pending_age_seconds = max(
                0.0, (datetime.now(timezone.utc) - ts).total_seconds()
            )
        except ValueError:
            oldest_pending_age_seconds = None

    total = sum(counts.values())
    return {
        "counts": counts,
        "total": total,
        "oldest_pending_age_seconds": oldest_pending_age_seconds,
    }


def list_events(
    db: MemoirsDB,
    *,
    status: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List queue rows for inspection (CLI / MCP). Newest-first."""
    if status is not None and status not in VALID_STATUSES:
        raise ValueError(
            f"unknown status {status!r}; expected one of {list(VALID_STATUSES)}"
        )
    if status:
        rows = db.conn.execute(
            "SELECT id, event_type, status, error, created_at, processed_at "
            "FROM event_queue WHERE status = ? ORDER BY id DESC LIMIT ?",
            (status, int(limit)),
        ).fetchall()
    else:
        rows = db.conn.execute(
            "SELECT id, event_type, status, error, created_at, processed_at "
            "FROM event_queue ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Convenience helper used by ingesters
# ---------------------------------------------------------------------------


def enqueue_messages_ingested(
    db: MemoirsDB,
    *,
    conversation_id: str | None,
    source_id: int | str | None,
    message_count: int,
    extra: dict[str, Any] | None = None,
) -> int | None:
    """Enqueue a ``messages_ingested`` event after a successful ingest.

    Returns the new event id, or ``None`` when ``message_count <= 0`` (in
    which case we do nothing — empty ingests are noise).
    """
    if not message_count or message_count <= 0:
        return None
    payload: dict[str, Any] = {
        "conversation_id": conversation_id,
        "source_id": source_id,
        "message_count": int(message_count),
    }
    if extra:
        payload.update(extra)
    return enqueue(db, event_type="messages_ingested", payload=payload)
