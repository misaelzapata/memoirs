"""Tests for the Claude.ai official export ingester (P4-2)."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from memoirs.ingesters.claude_export import (
    ImportStats,
    SOURCE_KIND,
    import_claude_export,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "claude_export_sample"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_export_dir(root: Path, payload: list[dict]) -> Path:
    """Materialize a synthetic export directory with `conversations.json`."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "conversations.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def _write_export_zip(zip_path: Path, payload: list[dict], *, nested: bool = False) -> Path:
    """Build a synthetic export zip containing `conversations.json`."""
    member = "wrapper/conversations.json" if nested else "conversations.json"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(member, json.dumps(payload))
        archive.writestr("users.json", json.dumps({"uuid": "acct-aaaa"}))
    return zip_path


def _row_counts(db) -> dict[str, int]:
    return {
        "sources": db.conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
        "conversations": db.conn.execute(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0],
        "messages": db.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
        "active_messages": db.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE is_active = 1"
        ).fetchone()[0],
    }


# ---------------------------------------------------------------------------
# fixture-based integration test (uses the on-disk sample)
# ---------------------------------------------------------------------------


def test_import_fixture_directory_exact_counts(tmp_db):
    stats = import_claude_export(FIXTURE_DIR, tmp_db)
    assert isinstance(stats, ImportStats)
    assert stats.conversations == 2
    # 2 messages in conv #1 + 3 messages in conv #2 = 5
    assert stats.messages == 5
    assert stats.skipped_conversations == 0

    counts = _row_counts(tmp_db)
    assert counts["sources"] == 1
    assert counts["conversations"] == 2
    assert counts["active_messages"] == 5

    # Source registered with the right kind.
    kind = tmp_db.conn.execute(
        "SELECT kind FROM sources WHERE uri = ?", (str(FIXTURE_DIR.resolve()),)
    ).fetchone()[0]
    assert kind == SOURCE_KIND


# ---------------------------------------------------------------------------
# core behavior
# ---------------------------------------------------------------------------


def test_import_directory_synthetic(tmp_db, tmp_path):
    payload = [
        {
            "uuid": "c-1",
            "name": "Solo",
            "created_at": "2026-01-01T00:00:00Z",
            "chat_messages": [
                {"uuid": "m-1", "sender": "human", "text": "hi"},
                {"uuid": "m-2", "sender": "assistant", "text": "hello"},
            ],
        }
    ]
    export = _write_export_dir(tmp_path / "export", payload)
    stats = import_claude_export(export, tmp_db)
    assert stats.conversations == 1
    assert stats.messages == 2

    rows = tmp_db.conn.execute(
        "SELECT role, content FROM messages ORDER BY ordinal"
    ).fetchall()
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert [r["content"] for r in rows] == ["hi", "hello"]


def test_import_zip_archive(tmp_db, tmp_path):
    payload = [
        {
            "uuid": "z-1",
            "name": "Zipped",
            "chat_messages": [
                {"uuid": "z-m1", "sender": "human", "text": "ping"},
                {
                    "uuid": "z-m2",
                    "sender": "assistant",
                    "content": [{"type": "text", "text": "pong"}],
                },
            ],
        }
    ]
    zip_path = _write_export_zip(tmp_path / "export.zip", payload)
    stats = import_claude_export(zip_path, tmp_db)
    assert stats.conversations == 1
    assert stats.messages == 2

    # Source URI should be the absolute zip path.
    src_uri = tmp_db.conn.execute(
        "SELECT uri FROM sources WHERE kind = ?", (SOURCE_KIND,)
    ).fetchone()[0]
    assert src_uri == str(zip_path.resolve())


def test_import_zip_with_nested_conversations_json(tmp_db, tmp_path):
    """Some users zip a wrapping folder; the parser should still find conversations.json."""
    payload = [
        {
            "uuid": "n-1",
            "name": "Nested",
            "chat_messages": [
                {"uuid": "n-m1", "sender": "human", "text": "ok"},
            ],
        }
    ]
    zip_path = _write_export_zip(tmp_path / "nested.zip", payload, nested=True)
    stats = import_claude_export(zip_path, tmp_db)
    assert stats.conversations == 1
    assert stats.messages == 1


# ---------------------------------------------------------------------------
# idempotency
# ---------------------------------------------------------------------------


def test_reimport_same_export_does_not_duplicate(tmp_db):
    """Running the importer twice on the same fixture must yield the same row counts."""
    import_claude_export(FIXTURE_DIR, tmp_db)
    first = _row_counts(tmp_db)
    import_claude_export(FIXTURE_DIR, tmp_db)
    second = _row_counts(tmp_db)
    assert first == second, f"second import duplicated rows: {first} vs {second}"


def test_reimport_zip_does_not_duplicate(tmp_db, tmp_path):
    payload = [
        {
            "uuid": "dup-1",
            "name": "Dup",
            "chat_messages": [
                {"uuid": "dup-m1", "sender": "human", "text": "x"},
                {"uuid": "dup-m2", "sender": "assistant", "text": "y"},
            ],
        }
    ]
    zip_path = _write_export_zip(tmp_path / "dup.zip", payload)
    import_claude_export(zip_path, tmp_db)
    before = _row_counts(tmp_db)
    import_claude_export(zip_path, tmp_db)
    after = _row_counts(tmp_db)
    assert before == after


# ---------------------------------------------------------------------------
# defensive parsing
# ---------------------------------------------------------------------------


def test_skips_messages_with_empty_content(tmp_db, tmp_path):
    payload = [
        {
            "uuid": "e-1",
            "chat_messages": [
                {"uuid": "e-m1", "sender": "human", "text": ""},          # empty
                {"uuid": "e-m2", "sender": "human", "text": "   "},       # whitespace
                {"uuid": "e-m3", "sender": "assistant", "text": "real"},  # kept
            ],
        }
    ]
    export = _write_export_dir(tmp_path / "ex", payload)
    stats = import_claude_export(export, tmp_db)
    assert stats.conversations == 1
    assert stats.messages == 1
    assert stats.skipped_messages == 2


def test_skips_conversation_without_uuid(tmp_db, tmp_path):
    payload = [
        {"name": "no uuid", "chat_messages": [{"sender": "human", "text": "hi"}]},
        {
            "uuid": "ok-1",
            "chat_messages": [{"uuid": "m1", "sender": "human", "text": "hi"}],
        },
    ]
    export = _write_export_dir(tmp_path / "ex", payload)
    stats = import_claude_export(export, tmp_db)
    assert stats.conversations == 1
    assert stats.skipped_conversations == 1


def test_unknown_keys_are_ignored(tmp_db, tmp_path):
    payload = [
        {
            "uuid": "u-1",
            "name": "weird",
            "future_key_we_dont_know": {"deep": [1, 2, 3]},
            "chat_messages": [
                {
                    "uuid": "u-m1",
                    "sender": "human",
                    "text": "hi",
                    "another_unknown": "ignore me",
                },
            ],
        }
    ]
    export = _write_export_dir(tmp_path / "ex", payload)
    stats = import_claude_export(export, tmp_db)
    assert stats.conversations == 1
    assert stats.messages == 1


def test_role_mapping_human_to_user(tmp_db, tmp_path):
    payload = [
        {
            "uuid": "r-1",
            "chat_messages": [
                {"uuid": "r-m1", "sender": "human", "text": "Q"},
                {"uuid": "r-m2", "sender": "assistant", "text": "A"},
            ],
        }
    ]
    export = _write_export_dir(tmp_path / "ex", payload)
    import_claude_export(export, tmp_db)
    roles = [
        r["role"]
        for r in tmp_db.conn.execute(
            "SELECT role FROM messages ORDER BY ordinal"
        ).fetchall()
    ]
    assert roles == ["user", "assistant"]


def test_content_array_with_tool_use_block(tmp_db, tmp_path):
    payload = [
        {
            "uuid": "t-1",
            "chat_messages": [
                {
                    "uuid": "t-m1",
                    "sender": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me search."},
                        {"type": "tool_use", "name": "web_search"},
                        {"type": "unknown_block", "stuff": 1},
                    ],
                }
            ],
        }
    ]
    export = _write_export_dir(tmp_path / "ex", payload)
    stats = import_claude_export(export, tmp_db)
    assert stats.messages == 1
    content = tmp_db.conn.execute("SELECT content FROM messages").fetchone()[0]
    assert "Let me search." in content
    assert "[tool_use: web_search]" in content


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


def test_malformed_json_raises(tmp_db, tmp_path):
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "conversations.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        import_claude_export(bad_dir, tmp_db)


def test_missing_conversations_json_raises(tmp_db, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        import_claude_export(empty_dir, tmp_db)


def test_zip_without_conversations_json_raises(tmp_db, tmp_path):
    zip_path = tmp_path / "wrong.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("readme.txt", "hello")
    with pytest.raises(ValueError, match="conversations.json"):
        import_claude_export(zip_path, tmp_db)


def test_payload_not_list_raises(tmp_db, tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "conversations.json").write_text(
        json.dumps({"not": "a list"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not a JSON array"):
        import_claude_export(bad, tmp_db)


def test_nonexistent_path_raises(tmp_db, tmp_path):
    with pytest.raises(FileNotFoundError):
        import_claude_export(tmp_path / "does-not-exist", tmp_db)
