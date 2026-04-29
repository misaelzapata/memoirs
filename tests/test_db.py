"""Tests for db.py: schema init, migrations, ingest_event."""
import json
import sqlite3


def test_init_creates_all_tables(tmp_db):
    rows = tmp_db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    names = {r[0] for r in rows}
    expected = {
        "sources", "conversations", "messages", "attachments", "import_runs",
        "memory_candidates", "memories", "memory_embeddings",
        "entities", "relationships", "memory_entities", "event_queue",
    }
    assert expected.issubset(names)


def test_init_sets_user_version(tmp_db):
    from memoirs.migrations import target_version
    v = tmp_db.conn.execute("PRAGMA user_version").fetchone()[0]
    assert v == target_version()


def test_init_idempotent(tmp_db):
    from memoirs.migrations import target_version
    tmp_db.init()
    tmp_db.init()
    v = tmp_db.conn.execute("PRAGMA user_version").fetchone()[0]
    assert v == target_version()


def test_wal_journal_mode(tmp_db):
    mode = tmp_db.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_ingest_event_creates_row(tmp_db):
    result = tmp_db.ingest_event({
        "type": "chat_message",
        "source": "test",
        "conversation_id": "conv-test",
        "message_id": "msg-1",
        "role": "user",
        "content": "hello world",
    })
    assert result["action"] == "inserted"
    n = tmp_db.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert n == 1


def test_ingest_event_idempotent_by_message_id(tmp_db):
    """Re-ingesting same message_id updates, not duplicates."""
    payload = {
        "type": "chat_message", "source": "test",
        "conversation_id": "conv-test", "message_id": "msg-1",
        "role": "user", "content": "first",
    }
    tmp_db.ingest_event(payload)
    payload["content"] = "updated content"
    result = tmp_db.ingest_event(payload)
    assert result["action"] == "updated"
    n = tmp_db.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert n == 1
    content = tmp_db.conn.execute("SELECT content FROM messages").fetchone()[0]
    assert content == "updated content"


def test_status_returns_counts(tmp_db):
    s = tmp_db.status()
    assert "conversations" in s
    assert "active_messages" in s
    assert "sources" in s
    assert s["active_messages"] == 0
