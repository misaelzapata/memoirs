"""Tests for the durable event_queue (P0-4).

Covers the new ``memoirs.engine.event_queue`` API plus the auto-enqueue hook
wired into ``watch.ingest_path`` and ``apply_decision``'s ADD path.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from memoirs.engine import event_queue as eq


# ---------------------------------------------------------------------------
# Core queue API
# ---------------------------------------------------------------------------


def test_enqueue_then_dequeue_round_trip(tmp_db):
    eid = eq.enqueue(tmp_db, event_type="messages_ingested", payload={"a": 1})
    assert isinstance(eid, int) and eid > 0

    batch = eq.dequeue_batch(tmp_db, limit=10)
    assert len(batch) == 1
    row = batch[0]
    assert row["id"] == eid
    assert row["event_type"] == "messages_ingested"
    assert row["payload"] == {"a": 1}
    # After dequeue the row is in 'processing' (locked).
    status = tmp_db.conn.execute(
        "SELECT status FROM event_queue WHERE id = ?", (eid,)
    ).fetchone()["status"]
    assert status == "processing"


def test_dequeue_locks_via_processing_no_double_take(tmp_db):
    """A second dequeue must not see rows already flipped to 'processing'."""
    eq.enqueue(tmp_db, event_type="x", payload={})
    eq.enqueue(tmp_db, event_type="x", payload={})
    first = eq.dequeue_batch(tmp_db, limit=10)
    assert len(first) == 2
    second = eq.dequeue_batch(tmp_db, limit=10)
    assert second == []  # both already locked


def test_mark_done_updates_status_and_clears_error(tmp_db):
    eid = eq.enqueue(tmp_db, event_type="x", payload={})
    eq.dequeue_batch(tmp_db, limit=1)
    eq.mark_done(tmp_db, eid, result={"ok": True})
    row = tmp_db.conn.execute(
        "SELECT status, processed_at, error FROM event_queue WHERE id = ?", (eid,)
    ).fetchone()
    assert row["status"] == "done"
    assert row["processed_at"] is not None
    assert row["error"] is None


def test_mark_failed_records_error(tmp_db):
    eid = eq.enqueue(tmp_db, event_type="x", payload={})
    eq.dequeue_batch(tmp_db, limit=1)
    eq.mark_failed(tmp_db, eid, error="boom")
    row = tmp_db.conn.execute(
        "SELECT status, error, processed_at FROM event_queue WHERE id = ?", (eid,)
    ).fetchone()
    assert row["status"] == "failed"
    assert row["error"] == "boom"
    assert row["processed_at"] is not None


def test_process_pending_dispatches_by_handler(tmp_db):
    seen: list[tuple[str, dict]] = []

    def h_a(db, payload):
        seen.append(("a", payload))
        return {"handled": "a"}

    def h_b(db, payload):
        seen.append(("b", payload))
        return {"handled": "b"}

    eq.enqueue(tmp_db, event_type="a", payload={"k": 1})
    eq.enqueue(tmp_db, event_type="b", payload={"k": 2})
    eq.enqueue(tmp_db, event_type="unknown", payload={})

    out = eq.process_pending(
        tmp_db, batch_size=10, handlers={"a": h_a, "b": h_b}
    )
    assert out["processed"] == 2
    assert out["failed"] == 0
    assert out["skipped"] == 1  # unknown event_type
    assert sorted(seen) == sorted([("a", {"k": 1}), ("b", {"k": 2})])
    # All three rows should now be 'done' (unknown gets marked done too).
    statuses = [
        r["status"]
        for r in tmp_db.conn.execute("SELECT status FROM event_queue").fetchall()
    ]
    assert statuses.count("done") == 3


def test_process_pending_marks_failed_on_handler_exception(tmp_db):
    def boom(db, payload):
        raise RuntimeError("handler exploded")

    eq.enqueue(tmp_db, event_type="x", payload={})
    out = eq.process_pending(tmp_db, batch_size=10, handlers={"x": boom})
    assert out["processed"] == 0
    assert out["failed"] == 1
    row = tmp_db.conn.execute(
        "SELECT status, error FROM event_queue ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row["status"] == "failed"
    assert "handler exploded" in row["error"]


def test_requeue_failed_resets_to_pending(tmp_db):
    eid = eq.enqueue(tmp_db, event_type="x", payload={})
    eq.dequeue_batch(tmp_db, limit=1)
    eq.mark_failed(tmp_db, eid, error="nope")
    n = eq.requeue_failed(tmp_db, max_age_hours=0)
    assert n == 1
    status = tmp_db.conn.execute(
        "SELECT status FROM event_queue WHERE id = ?", (eid,)
    ).fetchone()["status"]
    assert status == "pending"
    # And it can be re-dequeued.
    again = eq.dequeue_batch(tmp_db, limit=10)
    assert len(again) == 1
    assert again[0]["id"] == eid


def test_get_stats_shape_and_oldest_pending(tmp_db):
    # Empty.
    stats = eq.get_stats(tmp_db)
    assert set(stats.keys()) == {"counts", "total", "oldest_pending_age_seconds"}
    assert stats["counts"] == {"pending": 0, "processing": 0, "done": 0, "failed": 0}
    assert stats["total"] == 0
    assert stats["oldest_pending_age_seconds"] is None

    # With one pending.
    eq.enqueue(tmp_db, event_type="x", payload={})
    stats = eq.get_stats(tmp_db)
    assert stats["counts"]["pending"] == 1
    assert stats["total"] == 1
    assert isinstance(stats["oldest_pending_age_seconds"], float)
    assert stats["oldest_pending_age_seconds"] >= 0.0


def test_enqueue_messages_ingested_no_op_on_zero(tmp_db):
    """The convenience helper must not flood the queue with empty events."""
    out = eq.enqueue_messages_ingested(
        tmp_db, conversation_id="c1", source_id=1, message_count=0
    )
    assert out is None
    assert (
        tmp_db.conn.execute("SELECT COUNT(*) AS c FROM event_queue").fetchone()["c"]
        == 0
    )


def test_enqueue_messages_ingested_records_payload(tmp_db):
    eid = eq.enqueue_messages_ingested(
        tmp_db,
        conversation_id="conv-1",
        source_id=42,
        message_count=3,
        extra={"importer": "claude_code"},
    )
    assert isinstance(eid, int)
    row = tmp_db.conn.execute(
        "SELECT event_type, payload_json FROM event_queue WHERE id = ?", (eid,)
    ).fetchone()
    assert row["event_type"] == "messages_ingested"
    payload = json.loads(row["payload_json"])
    assert payload["conversation_id"] == "conv-1"
    assert payload["source_id"] == 42
    assert payload["message_count"] == 3
    assert payload["importer"] == "claude_code"


# ---------------------------------------------------------------------------
# Hooks: ingest + memory_engine ADD path
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(o) for o in lines) + "\n")


def test_ingest_claude_code_path_enqueues_messages_ingested(tmp_db, tmp_path, monkeypatch):
    """Real flow: watcher.ingest_path on a synthetic claude_code session leaves
    at least one ``messages_ingested`` event in the queue."""
    from memoirs.watch import ingest_path
    from memoirs.ingesters import claude_code

    fake_root = tmp_path / "claude_projects"
    project_dir = fake_root / "-home-x-foo"
    project_dir.mkdir(parents=True)
    monkeypatch.setattr(claude_code, "CLAUDE_CODE_PROJECTS", fake_root)

    jsonl = project_dir / "session-uuid.jsonl"
    _write_jsonl(jsonl, [
        {"type": "user", "message": {"role": "user", "content": "hello"}, "uuid": "u1"},
        {"type": "assistant",
         "message": {"role": "assistant",
                     "content": [{"type": "text", "text": "world"}]},
         "uuid": "u2"},
    ])
    ingest_path(tmp_db, jsonl, reporter=lambda *_: None)
    rows = tmp_db.conn.execute(
        "SELECT event_type, payload_json FROM event_queue WHERE event_type = 'messages_ingested'"
    ).fetchall()
    assert len(rows) >= 1
    payload = json.loads(rows[0]["payload_json"])
    assert payload["message_count"] >= 1
    assert payload["source_id"] is not None


def test_ingest_idempotent_no_duplicate_events_on_replay(tmp_db, tmp_path, monkeypatch):
    """Re-ingesting an unchanged file must not enqueue another event — the
    watcher only fires when delta > 0."""
    from memoirs.watch import ingest_path
    from memoirs.ingesters import claude_code

    fake_root = tmp_path / "claude_projects"
    project_dir = fake_root / "-home-x-foo2"
    project_dir.mkdir(parents=True)
    monkeypatch.setattr(claude_code, "CLAUDE_CODE_PROJECTS", fake_root)

    jsonl = project_dir / "session.jsonl"
    _write_jsonl(jsonl, [
        {"type": "user", "message": {"role": "user", "content": "hi"}, "uuid": "u1"},
    ])
    ingest_path(tmp_db, jsonl, reporter=lambda *_: None)
    ingest_path(tmp_db, jsonl, reporter=lambda *_: None)  # replay
    n = tmp_db.conn.execute(
        "SELECT COUNT(*) AS c FROM event_queue WHERE event_type = 'messages_ingested'"
    ).fetchone()["c"]
    assert n == 1


def test_apply_decision_add_enqueues_memory_promoted(tmp_db):
    """The ADD branch of apply_decision must drop a memory_promoted event."""
    from memoirs.engine.memory_engine import apply_decision, Decision
    from memoirs.engine.gemma import Candidate

    cand = Candidate(
        type="fact",
        content="The watcher is the producer for messages_ingested events.",
        importance=4,
        confidence=0.9,
    )
    decision = Decision(action="ADD", reason="test")
    out = apply_decision(tmp_db, cand, decision)
    assert out["action"] == "ADD"
    rows = tmp_db.conn.execute(
        "SELECT event_type, payload_json FROM event_queue "
        "WHERE event_type = 'memory_promoted'"
    ).fetchall()
    assert len(rows) == 1
    payload = json.loads(rows[0]["payload_json"])
    assert payload["memory_id"] == out["memory_id"]
    assert payload["type"] == "fact"


# ---------------------------------------------------------------------------
# Performance: dequeue 1000 rows fast (<50ms)
# ---------------------------------------------------------------------------


def test_dequeue_1000_under_50ms(tmp_db):
    # Use the raw connection to load the table fast — bypassing enqueue's
    # per-row commit so the perf test isn't dominated by fsync.
    rows = [(f"t{i}", json.dumps({"i": i}), "pending", "2026-04-27T00:00:00+00:00")
            for i in range(1000)]
    tmp_db.conn.executemany(
        "INSERT INTO event_queue (event_type, payload_json, status, created_at) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )
    tmp_db.conn.commit()
    t0 = time.perf_counter()
    batch = eq.dequeue_batch(tmp_db, limit=1000)
    dt = (time.perf_counter() - t0) * 1000.0
    assert len(batch) == 1000
    # Generous bound — local SQLite should easily clear this. Bump if a
    # slower CI starts flaking.
    assert dt < 50.0, f"dequeue 1000 took {dt:.2f}ms (>50ms)"


# ---------------------------------------------------------------------------
# list_events / requeue edge cases
# ---------------------------------------------------------------------------


def test_list_events_filters_by_status(tmp_db):
    a = eq.enqueue(tmp_db, event_type="x", payload={})
    eq.dequeue_batch(tmp_db, limit=1)
    eq.mark_failed(tmp_db, a, error="x")
    eq.enqueue(tmp_db, event_type="y", payload={})
    failed = eq.list_events(tmp_db, status="failed", limit=10)
    pending = eq.list_events(tmp_db, status="pending", limit=10)
    assert len(failed) == 1 and failed[0]["status"] == "failed"
    assert len(pending) == 1 and pending[0]["status"] == "pending"


def test_list_events_unknown_status_raises(tmp_db):
    with pytest.raises(ValueError):
        eq.list_events(tmp_db, status="bogus")
