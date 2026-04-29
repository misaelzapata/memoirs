"""Per-ingester event_queue hooks (P0-4 GAP).

Each source-specific ingester (claude_code, cursor, the markdown / jsonl /
chatgpt-zip dispatcher in importers.py) is supposed to drop one
``messages_ingested`` event into ``event_queue`` per conversation that gained
new messages. These tests pin that contract and cover the idempotency story:
re-running an ingester on an unchanged file must NOT enqueue duplicates.
"""
from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from memoirs.engine import event_queue as eq
from memoirs.ingesters import claude_code, cursor, importers


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _pending_events(db) -> list[dict]:
    rows = db.conn.execute(
        "SELECT id, event_type, payload_json FROM event_queue "
        "WHERE status = 'pending' ORDER BY id"
    ).fetchall()
    out = []
    for r in rows:
        out.append(
            {
                "id": int(r["id"]),
                "event_type": str(r["event_type"]),
                "payload": json.loads(r["payload_json"]),
            }
        )
    return out


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    with path.open("w") as f:
        for obj in lines:
            f.write(json.dumps(obj) + "\n")


# ---------------------------------------------------------------------------
# claude_code ingester
# ---------------------------------------------------------------------------


def test_claude_code_ingest_emits_one_event_per_conversation(tmp_db, tmp_path: Path):
    """A single Claude Code session JSONL → 1 conversation → 1 event."""
    jsonl = tmp_path / "session-uuid.jsonl"
    _write_jsonl(
        jsonl,
        [
            {"type": "user", "message": {"role": "user", "content": "hi"}, "uuid": "u1"},
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "yo"}]},
                "uuid": "u2",
            },
        ],
    )

    out = claude_code.ingest_claude_code_jsonl(jsonl, tmp_db)

    assert out["conversations"] == 1
    assert out["messages"] == 2
    assert out["events_enqueued"] == 1
    assert out["new_messages"] == 2

    events = _pending_events(tmp_db)
    assert len(events) == 1
    ev = events[0]
    assert ev["event_type"] == "messages_ingested"
    assert ev["payload"]["message_count"] == 2
    assert ev["payload"]["conversation_id"]
    assert ev["payload"]["importer"] == "claude_code"
    assert ev["payload"]["source_kind"] == "claude_code"
    assert ev["payload"]["source_id"] is not None


def test_claude_code_ingest_is_idempotent(tmp_db, tmp_path: Path):
    """Re-running the same JSONL must not produce a second event."""
    jsonl = tmp_path / "session-uuid.jsonl"
    _write_jsonl(
        jsonl,
        [
            {"type": "user", "message": {"role": "user", "content": "hi"}, "uuid": "u1"},
        ],
    )

    first = claude_code.ingest_claude_code_jsonl(jsonl, tmp_db)
    assert first["events_enqueued"] == 1
    assert len(_pending_events(tmp_db)) == 1

    second = claude_code.ingest_claude_code_jsonl(jsonl, tmp_db)
    assert second["events_enqueued"] == 0
    assert second["new_messages"] == 0
    # Still exactly one pending event (the original).
    assert len(_pending_events(tmp_db)) == 1


def test_claude_code_ingest_appended_messages_emit_one_more_event(
    tmp_db, tmp_path: Path
):
    """Adding new messages to an already-ingested file emits exactly one event."""
    jsonl = tmp_path / "session-uuid.jsonl"
    _write_jsonl(
        jsonl,
        [{"type": "user", "message": {"role": "user", "content": "hi"}, "uuid": "u1"}],
    )
    claude_code.ingest_claude_code_jsonl(jsonl, tmp_db)
    assert len(_pending_events(tmp_db)) == 1

    # Append a new message → re-ingest.
    _write_jsonl(
        jsonl,
        [
            {"type": "user", "message": {"role": "user", "content": "hi"}, "uuid": "u1"},
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": "world"},
                "uuid": "u2",
            },
        ],
    )
    out = claude_code.ingest_claude_code_jsonl(jsonl, tmp_db)
    assert out["events_enqueued"] == 1
    assert out["new_messages"] == 1

    pending = _pending_events(tmp_db)
    assert len(pending) == 2
    # The latest event reports just the new row (delta = 1).
    assert pending[-1]["payload"]["message_count"] == 1


# ---------------------------------------------------------------------------
# cursor ingester
# ---------------------------------------------------------------------------


def _make_cursor_db(path: Path, *, blobs: dict[str, list[dict]]) -> None:
    """Create a minimal Cursor state.vscdb with one or more chat blobs.

    ``blobs`` maps a candidate key (e.g. the chatdata key) to a list of
    message dicts that will be wrapped in a Cursor-style payload.
    """
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE ItemTable (key TEXT PRIMARY KEY, value TEXT)"
        )
        for key, msgs in blobs.items():
            payload = {"tabs": [{"messages": msgs}]}
            conn.execute(
                "INSERT INTO ItemTable (key, value) VALUES (?, ?)",
                (key, json.dumps(payload)),
            )
        conn.commit()
    finally:
        conn.close()


def test_cursor_ingest_emits_one_event_per_conversation(tmp_db, tmp_path: Path):
    """Two chat blobs in a Cursor state.vscdb → 2 conversations → 2 events."""
    db_path = tmp_path / "state.vscdb"
    _make_cursor_db(
        db_path,
        blobs={
            "workbench.panel.aichat.view.aichat.chatdata": [
                {"role": "user", "text": "alpha", "id": "a1"},
                {"role": "assistant", "text": "alpha-reply", "id": "a2"},
            ],
            "aiService.prompts": [
                {"role": "user", "text": "beta", "id": "b1"},
            ],
        },
    )

    out = cursor.ingest_cursor_state(db_path, tmp_db)
    assert out["conversations"] == 2
    assert out["events_enqueued"] == 2
    assert out["new_messages"] == 3

    events = _pending_events(tmp_db)
    assert len(events) == 2
    for ev in events:
        assert ev["event_type"] == "messages_ingested"
        assert ev["payload"]["importer"] == "cursor"
        assert ev["payload"]["source_kind"] == "cursor"
    # Distinct conversation_ids per blob.
    cids = {ev["payload"]["conversation_id"] for ev in events}
    assert len(cids) == 2


def test_cursor_ingest_idempotent_on_rerun(tmp_db, tmp_path: Path):
    db_path = tmp_path / "state.vscdb"
    _make_cursor_db(
        db_path,
        blobs={
            "workbench.panel.aichat.view.aichat.chatdata": [
                {"role": "user", "text": "x", "id": "x1"},
            ],
        },
    )
    cursor.ingest_cursor_state(db_path, tmp_db)
    assert len(_pending_events(tmp_db)) == 1

    out2 = cursor.ingest_cursor_state(db_path, tmp_db)
    assert out2["events_enqueued"] == 0
    assert len(_pending_events(tmp_db)) == 1


# ---------------------------------------------------------------------------
# importers.py — chatgpt zip / markdown / jsonl
# ---------------------------------------------------------------------------


def _make_chatgpt_zip(path: Path, n_conversations: int) -> None:
    payload = []
    for i in range(n_conversations):
        payload.append(
            {
                "id": f"conv-{i}",
                "title": f"Conversation {i}",
                "create_time": 1700000000 + i,
                "mapping": {
                    f"n{i}-1": {
                        "message": {
                            "id": f"m-{i}-1",
                            "author": {"role": "user"},
                            "create_time": 1,
                            "content": {"parts": [f"hello-{i}"]},
                        }
                    },
                    f"n{i}-2": {
                        "message": {
                            "id": f"m-{i}-2",
                            "author": {"role": "assistant"},
                            "create_time": 2,
                            "content": {"parts": [f"reply-{i}"]},
                        }
                    },
                },
            }
        )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("conversations.json", json.dumps(payload))


def test_chatgpt_zip_emits_one_event_per_conversation(tmp_db, tmp_path: Path):
    """A ChatGPT export zip with N conversations → N events."""
    zip_path = tmp_path / "chatgpt-export.zip"
    _make_chatgpt_zip(zip_path, n_conversations=3)

    out = importers.ingest_file_with_events(zip_path, tmp_db, importer="chatgpt")
    assert out["conversations"] == 3
    assert out["events_enqueued"] == 3
    assert out["new_messages"] == 6  # 2 messages each

    events = _pending_events(tmp_db)
    assert len(events) == 3
    cids = {ev["payload"]["conversation_id"] for ev in events}
    assert len(cids) == 3
    for ev in events:
        assert ev["event_type"] == "messages_ingested"
        assert ev["payload"]["source_kind"] == "chatgpt"
        assert ev["payload"]["message_count"] == 2


def test_markdown_single_file_emits_one_event(tmp_db, tmp_path: Path):
    """A standalone .md file → 1 conversation → 1 event."""
    md = tmp_path / "note.md"
    md.write_text(
        "\n".join(
            [
                "Some prose preamble.",
                json.dumps({"role": "user", "content": "what is X?"}),
                json.dumps({"role": "assistant", "content": "X is..."}),
            ]
        )
    )

    out = importers.ingest_file_with_events(md, tmp_db)
    assert out["conversations"] == 1
    assert out["events_enqueued"] == 1
    assert out["new_messages"] >= 1  # prose doc + 2 jsonl msgs

    events = _pending_events(tmp_db)
    assert len(events) == 1
    assert events[0]["event_type"] == "messages_ingested"
    assert events[0]["payload"]["source_kind"] == "markdown"


def test_jsonl_ingester_emits_one_event_per_conversation(tmp_db, tmp_path: Path):
    """A standalone .jsonl file (not under ~/.claude/projects) → 1 conv → 1 event."""
    jsonl = tmp_path / "log.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "ping"}),
                json.dumps({"role": "assistant", "content": "pong"}),
            ]
        )
    )

    out = importers.ingest_file_with_events(jsonl, tmp_db)
    assert out["conversations"] == 1
    assert out["events_enqueued"] == 1
    assert out["new_messages"] == 2

    events = _pending_events(tmp_db)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["source_kind"] == "jsonl"
    assert payload["message_count"] == 2


def test_ingest_file_idempotent_no_duplicate_events(tmp_db, tmp_path: Path):
    """Re-running ingest_file_with_events on an unchanged file enqueues nothing new."""
    jsonl = tmp_path / "log.jsonl"
    jsonl.write_text(json.dumps({"role": "user", "content": "x"}))
    importers.ingest_file_with_events(jsonl, tmp_db)
    first_count = len(_pending_events(tmp_db))
    assert first_count == 1

    out = importers.ingest_file_with_events(jsonl, tmp_db)
    assert out["events_enqueued"] == 0
    assert len(_pending_events(tmp_db)) == first_count


# ---------------------------------------------------------------------------
# Behavioral guarantee: enqueue failure must NOT break the ingest path
# ---------------------------------------------------------------------------


def test_ingest_swallows_event_queue_failure(tmp_db, tmp_path: Path, monkeypatch):
    """A queue failure must not bubble up; the save itself still succeeds."""
    jsonl = tmp_path / "session-uuid.jsonl"
    _write_jsonl(
        jsonl,
        [{"type": "user", "message": {"role": "user", "content": "hi"}, "uuid": "u1"}],
    )

    from memoirs.ingesters import importers as imp

    def boom(*args, **kwargs):
        raise RuntimeError("queue down")

    monkeypatch.setattr(
        "memoirs.engine.event_queue.enqueue_messages_ingested", boom
    )

    out = claude_code.ingest_claude_code_jsonl(jsonl, tmp_db)
    # Ingest itself still completed — the failure is swallowed at the boundary.
    assert out["conversations"] == 1
    assert out["messages"] == 1
    assert out["events_enqueued"] == 0
    # Conversation row exists in DB.
    n_convs = tmp_db.conn.execute(
        "SELECT COUNT(*) AS n FROM conversations"
    ).fetchone()["n"]
    assert n_convs == 1
