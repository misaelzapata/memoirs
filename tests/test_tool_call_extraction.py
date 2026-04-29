"""Tests for automatic ``tool_call`` memory extraction (post-P1-8 gap closure).

Covers :mod:`memoirs.engine.tool_call_extract`:

* :func:`extract_tool_calls_from_message` — pure parser over single
  Anthropic-style messages (text-only, single tool_use+tool_result, error
  results, oversized result truncation).
* :func:`record_tool_calls_for_conversation` — orchestrator that walks a
  whole conversation, persists one memory per invocation, dedupes on
  re-run, and honors the ``MEMOIRS_EXTRACT_TOOL_CALLS`` opt-out.
* CLI integration — ``memoirs extract`` over a seeded conversation
  yields the expected ``type='tool_call'`` memorias in the database.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from memoirs.db import MemoirsDB
from memoirs.engine import memory_engine as me
from memoirs.engine import tool_call_extract as tce
from memoirs.models import RawConversation, RawMessage


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _assistant_with_tool_use(tool_id: str, name: str, inp: dict) -> dict:
    """Build a Claude Code-shaped JSONL line carrying one ``tool_use`` block."""
    return {
        "type": "assistant",
        "uuid": f"u-{tool_id}",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": inp,
                }
            ],
        },
    }


def _user_with_tool_result(
    tool_id: str, content: str | list, *, is_error: bool = False
) -> dict:
    """Build a Claude Code-shaped JSONL line carrying one ``tool_result`` block."""
    block: dict = {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "content": content,
    }
    if is_error:
        block["is_error"] = True
    return {
        "type": "user",
        "uuid": f"u-result-{tool_id}",
        "message": {
            "role": "user",
            "content": [block],
        },
    }


def _seed_conversation(
    db: MemoirsDB,
    *,
    conv_id: str = "conv_test",
    raw_messages: list[dict],
) -> str:
    """Persist a conversation directly via ``store_conversations``.

    ``raw_messages`` are Claude Code-shaped dicts; their flattened content
    feeds ``RawMessage.content`` and the originals land in ``raw_json``.
    """
    from memoirs.core.normalize import flatten_content

    msgs: list[RawMessage] = []
    for i, raw in enumerate(raw_messages):
        inner = raw.get("message", {}) if isinstance(raw, dict) else {}
        role = inner.get("role") or raw.get("type") or "user"
        content = flatten_content(inner.get("content"))
        msgs.append(
            RawMessage(
                role=role,
                content=content or "(empty)",
                ordinal=i,
                external_id=raw.get("uuid"),
                metadata={},
                raw=raw,
            )
        )
    conv = RawConversation(
        external_id=conv_id,
        title="t",
        source_kind="claude_code",
        source_uri=f"mem://{conv_id}",
        messages=msgs,
        created_at=None,
        metadata={},
    )
    # ``save_conversations`` upserts both rows; the resulting conversation
    # id is derived deterministically from (source, external_id).
    n_conv, _ = db.save_conversations(
        [conv],
        source_name=f"test-{conv_id}",
        source_kind="claude_code",
        source_uri=f"mem://{conv_id}",
        hash_value=None,
        mtime_ns=None,
        size_bytes=None,
    )
    assert n_conv == 1
    row = db.conn.execute(
        "SELECT id FROM conversations WHERE external_id = ?", (conv_id,),
    ).fetchone()
    return row["id"]


# ---------------------------------------------------------------------------
# extract_tool_calls_from_message — single-message parser
# ---------------------------------------------------------------------------


def test_parser_returns_event_for_paired_tool_use_and_result():
    """One message that carries both blocks yields exactly one event."""
    msg = {
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "bash",
                    "input": {"cmd": "ls"},
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_1",
                    "content": "file1\nfile2",
                },
            ],
        }
    }
    events = tce.extract_tool_calls_from_message(msg)
    assert len(events) == 1
    ev = events[0]
    assert ev.tool_name == "bash"
    assert ev.args == {"cmd": "ls"}
    assert ev.status == "success"
    assert ev.result_summary == "file1\nfile2"
    assert ev.tool_use_id == "tu_1"


def test_parser_returns_empty_for_message_without_tool_blocks():
    """A plain text message must produce no events."""
    msg = {
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
        }
    }
    assert tce.extract_tool_calls_from_message(msg) == []
    # Same for str content (legal Anthropic shape):
    assert tce.extract_tool_calls_from_message({"message": {"content": "hi"}}) == []
    # And for a totally unrelated dict:
    assert tce.extract_tool_calls_from_message({"foo": "bar"}) == []


def test_parser_marks_tool_use_without_result_as_cancelled():
    """An assistant tool_use with no co-located tool_result is `cancelled`."""
    msg = _assistant_with_tool_use("tu_x", "Read", {"path": "/etc/hosts"})
    events = tce.extract_tool_calls_from_message(msg)
    assert len(events) == 1
    assert events[0].status == "cancelled"
    assert events[0].result_summary == ""
    assert events[0].tool_name == "Read"


def test_parser_truncates_long_results_and_includes_sha256():
    """A result longer than 500 chars must be capped at 200 + sha256 suffix."""
    payload = "y" * 1200
    msg = {
        "message": {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_long", "name": "bash", "input": {}},
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_long",
                    "content": payload,
                },
            ],
        }
    }
    events = tce.extract_tool_calls_from_message(msg)
    assert len(events) == 1
    summary = events[0].result_summary
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    # Head is exactly SUMMARY_MAX chars of the original.
    assert summary.startswith("y" * tce.SUMMARY_MAX)
    assert "[truncated, sha256=" in summary
    assert digest in summary
    # Sanity: full payload never appears in the summary.
    assert payload not in summary


def test_parser_status_error_when_is_error_true():
    """``is_error=True`` on tool_result flips status to 'error'."""
    msg = {
        "message": {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_e", "name": "Read", "input": {"path": "/missing"}},
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_e",
                    "content": "ENOENT: no such file",
                    "is_error": True,
                },
            ],
        }
    }
    events = tce.extract_tool_calls_from_message(msg)
    assert len(events) == 1
    assert events[0].status == "error"


# ---------------------------------------------------------------------------
# record_tool_calls_for_conversation — full conversation orchestration
# ---------------------------------------------------------------------------


def test_record_tool_calls_writes_one_memory_per_pair(tmp_db):
    """Conv with 3 msgs (2 of which carry tool calls) → 2 type='tool_call' rows."""
    raw = [
        # ordinal 0: user prompt (no tools)
        {
            "type": "user",
            "uuid": "u0",
            "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        },
        # ordinal 1: assistant issues tool_use #1
        _assistant_with_tool_use("tu_a", "bash", {"cmd": "ls"}),
        # ordinal 2: user echoes tool_result #1 + assistant issues #2 in same line
        _user_with_tool_result("tu_a", "file1\nfile2"),
        _assistant_with_tool_use("tu_b", "Read", {"path": "/etc/hosts"}),
        _user_with_tool_result("tu_b", "127.0.0.1 localhost"),
    ]
    cid = _seed_conversation(tmp_db, conv_id="conv_pair", raw_messages=raw)

    inserted = tce.record_tool_calls_for_conversation(tmp_db, cid)
    assert inserted == 2

    rows = tmp_db.conn.execute(
        "SELECT tool_name, tool_status, tool_args_json FROM memories "
        "WHERE type='tool_call' AND archived_at IS NULL "
        "ORDER BY tool_name"
    ).fetchall()
    assert [r["tool_name"] for r in rows] == ["Read", "bash"]
    assert all(r["tool_status"] == "success" for r in rows)
    assert json.loads(rows[1]["tool_args_json"]) == {"cmd": "ls"}


def test_record_tool_calls_is_idempotent(tmp_db):
    """Running twice must not duplicate memorias."""
    raw = [
        _assistant_with_tool_use("tu_1", "bash", {"cmd": "pwd"}),
        _user_with_tool_result("tu_1", "/home/u"),
        _assistant_with_tool_use("tu_2", "bash", {"cmd": "id"}),
        _user_with_tool_result("tu_2", "uid=1000"),
    ]
    cid = _seed_conversation(tmp_db, conv_id="conv_idem", raw_messages=raw)

    n1 = tce.record_tool_calls_for_conversation(tmp_db, cid)
    n2 = tce.record_tool_calls_for_conversation(tmp_db, cid)
    assert n1 == 2
    assert n2 == 0  # second pass: nothing new

    total = tmp_db.conn.execute(
        "SELECT COUNT(*) AS c FROM memories WHERE type='tool_call' AND archived_at IS NULL"
    ).fetchone()["c"]
    assert total == 2


def test_record_tool_calls_skipped_when_env_off(tmp_db, monkeypatch):
    """``MEMOIRS_EXTRACT_TOOL_CALLS=off`` must short-circuit cleanly."""
    raw = [
        _assistant_with_tool_use("tu_skip", "bash", {"cmd": "echo hi"}),
        _user_with_tool_result("tu_skip", "hi"),
    ]
    cid = _seed_conversation(tmp_db, conv_id="conv_off", raw_messages=raw)

    monkeypatch.setenv("MEMOIRS_EXTRACT_TOOL_CALLS", "off")
    inserted = tce.record_tool_calls_for_conversation(tmp_db, cid)
    assert inserted == 0

    rows = tmp_db.conn.execute(
        "SELECT COUNT(*) AS c FROM memories WHERE type='tool_call' AND archived_at IS NULL"
    ).fetchone()
    assert rows["c"] == 0

    # Sanity: turning it back on extracts as expected.
    monkeypatch.setenv("MEMOIRS_EXTRACT_TOOL_CALLS", "on")
    inserted = tce.record_tool_calls_for_conversation(tmp_db, cid)
    assert inserted == 1


def test_record_tool_calls_marks_orphan_use_as_cancelled(tmp_db):
    """A tool_use with no matching tool_result lands as ``status='cancelled'``."""
    raw = [
        _assistant_with_tool_use("tu_orphan", "bash", {"cmd": "sleep 9999"}),
        # No result for tu_orphan in the conversation.
    ]
    cid = _seed_conversation(tmp_db, conv_id="conv_orphan", raw_messages=raw)

    inserted = tce.record_tool_calls_for_conversation(tmp_db, cid)
    assert inserted == 1
    row = tmp_db.conn.execute(
        "SELECT tool_status, tool_name FROM memories WHERE type='tool_call' "
        "AND json_extract(metadata_json, '$.conversation_id') = ?",
        (cid,),
    ).fetchone()
    assert row["tool_status"] == "cancelled"
    assert row["tool_name"] == "bash"


def test_record_tool_calls_propagates_error_status(tmp_db):
    """``is_error=True`` is preserved through the orchestrator."""
    raw = [
        _assistant_with_tool_use("tu_err", "Read", {"path": "/nope"}),
        _user_with_tool_result("tu_err", "ENOENT", is_error=True),
    ]
    cid = _seed_conversation(tmp_db, conv_id="conv_err", raw_messages=raw)
    inserted = tce.record_tool_calls_for_conversation(tmp_db, cid)
    assert inserted == 1
    row = tmp_db.conn.execute(
        "SELECT tool_status FROM memories WHERE type='tool_call'"
    ).fetchone()
    assert row["tool_status"] == "error"


def test_record_tool_calls_no_op_for_unknown_conversation(tmp_db):
    """Empty / nonexistent conversation_id must return 0 without raising."""
    assert tce.record_tool_calls_for_conversation(tmp_db, "") == 0
    assert tce.record_tool_calls_for_conversation(tmp_db, "does-not-exist") == 0


# ---------------------------------------------------------------------------
# CLI integration — ``memoirs extract`` end-to-end
# ---------------------------------------------------------------------------


def test_cli_extract_records_tool_calls_for_seeded_conv(tmp_path: Path, monkeypatch):
    """``memoirs extract`` on a DB with a tool-using conversation must end with
    matching ``type='tool_call'`` rows in the corpus.

    We invoke the CLI as a subprocess so the env-var path runs through the
    real ``argparse`` + dispatch pipeline. Gemma may or may not be loaded
    on the host; either way the tool-call extraction hook runs.
    """
    db_path = tmp_path / "memoirs.sqlite"
    db = MemoirsDB(db_path)
    db.init()

    raw = [
        # Need at least 3 messages so ``extract_pending`` considers it.
        {
            "type": "user",
            "uuid": "u0",
            "message": {"role": "user", "content": [{"type": "text", "text": "go"}]},
        },
        _assistant_with_tool_use("tu_cli_1", "bash", {"cmd": "echo hi"}),
        _user_with_tool_result("tu_cli_1", "hi"),
        _assistant_with_tool_use("tu_cli_2", "Read", {"path": "/etc/hostname"}),
        _user_with_tool_result("tu_cli_2", "myhost"),
    ]
    _seed_conversation(db, conv_id="conv_cli", raw_messages=raw)
    db.close()

    env = os.environ.copy()
    env["MEMOIRS_EXTRACT_TOOL_CALLS"] = "on"
    # Disable Gemma so the test is fast and deterministic — we only care
    # about the tool_call hook here.
    env["MEMOIRS_DISABLE_GEMMA"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "memoirs", "--db", str(db_path), "extract", "--limit", "5"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    payload = json.loads(proc.stdout)
    assert payload.get("tool_calls_extracted") == 2
    assert payload.get("tool_call_conversations") == 1

    # Verify the corpus shape directly.
    db2 = MemoirsDB(db_path)
    db2.init()
    try:
        rows = db2.conn.execute(
            "SELECT tool_name, tool_status FROM memories "
            "WHERE type='tool_call' AND archived_at IS NULL "
            "ORDER BY tool_name"
        ).fetchall()
    finally:
        db2.close()
    assert [r["tool_name"] for r in rows] == ["Read", "bash"]
    assert all(r["tool_status"] == "success" for r in rows)


def test_cli_extract_respects_env_off(tmp_path: Path):
    """With ``MEMOIRS_EXTRACT_TOOL_CALLS=off`` the CLI must NOT write tool_call rows."""
    db_path = tmp_path / "memoirs.sqlite"
    db = MemoirsDB(db_path)
    db.init()

    raw = [
        {
            "type": "user",
            "uuid": "u0",
            "message": {"role": "user", "content": [{"type": "text", "text": "go"}]},
        },
        _assistant_with_tool_use("tu_off", "bash", {"cmd": "id"}),
        _user_with_tool_result("tu_off", "uid=1000"),
    ]
    _seed_conversation(db, conv_id="conv_off_cli", raw_messages=raw)
    db.close()

    env = os.environ.copy()
    env["MEMOIRS_EXTRACT_TOOL_CALLS"] = "off"
    env["MEMOIRS_DISABLE_GEMMA"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "memoirs", "--db", str(db_path), "extract", "--limit", "5"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload.get("tool_calls_extracted", 0) == 0

    db2 = MemoirsDB(db_path)
    db2.init()
    try:
        n = db2.conn.execute(
            "SELECT COUNT(*) AS c FROM memories WHERE type='tool_call'"
        ).fetchone()["c"]
    finally:
        db2.close()
    assert n == 0
