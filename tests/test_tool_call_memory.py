"""Tests for tool-call memory (P1-8 GAP).

Covers:
* ``record_tool_call`` writes a row with type='tool_call' and the four
  tool_* columns populated correctly.
* JSON serialization: nested dicts in args, large results hash but don't
  bloat ``tool_args_json``, status='error' and 'cancelled' persist.
* Validation: empty tool_name and unknown status raise ``ValueError``.
* Search by ``type='tool_call'`` only returns tool_call rows.
* ``summarize_tool_calls`` groups by tool_name with correct counts and
  success_rate.
* ``get_tool_calls_for_conversation`` filters by metadata.conversation_id.
* MCP ``mcp_record_tool_call`` end-to-end via the dispatcher.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from memoirs.engine import memory_engine as me
from memoirs.mcp import tools as mcp_tools


# ---------------------------------------------------------------------------
# record_tool_call — happy path + columns
# ---------------------------------------------------------------------------

def test_record_tool_call_persists_row(tmp_db):
    mid = me.record_tool_call(
        tmp_db,
        tool_name="bash",
        args={"cmd": "ls"},
        result="file1\nfile2",
    )
    assert mid.startswith("mem_")

    row = tmp_db.conn.execute(
        "SELECT type, content, tool_name, tool_args_json, tool_result_hash, "
        "tool_status, importance FROM memories WHERE id = ?",
        (mid,),
    ).fetchone()
    assert row is not None
    assert row["type"] == "tool_call"
    assert row["tool_name"] == "bash"
    # args_json round-trips
    assert json.loads(row["tool_args_json"]) == {"cmd": "ls"}
    # result_hash is sha256[:16] of utf-8 result bytes
    expected_hash = hashlib.sha256(b"file1\nfile2").hexdigest()[:16]
    assert row["tool_result_hash"] == expected_hash
    assert row["tool_status"] == "success"
    # default importance is 2
    assert row["importance"] == 2
    # human content is greppable
    assert "bash(" in row["content"]
    assert "→" in row["content"]


def test_record_tool_call_serializes_nested_args(tmp_db):
    """Nested dicts and lists must round-trip through tool_args_json."""
    nested = {"options": {"verbose": True, "depth": 3}, "paths": ["/a", "/b"]}
    mid = me.record_tool_call(
        tmp_db,
        tool_name="grep",
        args=nested,
        result="match",
    )
    row = tmp_db.conn.execute(
        "SELECT tool_args_json FROM memories WHERE id = ?", (mid,),
    ).fetchone()
    assert json.loads(row["tool_args_json"]) == nested


def test_record_tool_call_large_result_hashes_but_does_not_bloat(tmp_db):
    """A huge result must be reduced to a 16-char hex hash; no portion of it
    is stored in ``tool_args_json`` or ``content``.
    """
    huge = "x" * 200_000
    mid = me.record_tool_call(
        tmp_db,
        tool_name="bash",
        args={"cmd": "cat big"},
        result=huge,
    )
    row = tmp_db.conn.execute(
        "SELECT content, tool_args_json, tool_result_hash FROM memories WHERE id = ?",
        (mid,),
    ).fetchone()
    assert len(row["tool_result_hash"]) == 16
    assert row["tool_result_hash"] == hashlib.sha256(huge.encode("utf-8")).hexdigest()[:16]
    # The huge payload must NOT live anywhere on the row.
    assert huge not in (row["content"] or "")
    assert huge not in (row["tool_args_json"] or "")
    # Sanity: row content stays small.
    assert len(row["content"]) < 500


def test_record_tool_call_status_error_persists(tmp_db):
    mid = me.record_tool_call(
        tmp_db,
        tool_name="Read",
        args={"path": "/missing"},
        result="ENOENT",
        status="error",
    )
    row = tmp_db.conn.execute(
        "SELECT tool_status, content FROM memories WHERE id = ?", (mid,),
    ).fetchone()
    assert row["tool_status"] == "error"
    assert "error" in row["content"].lower()


def test_record_tool_call_status_cancelled_persists(tmp_db):
    mid = me.record_tool_call(
        tmp_db,
        tool_name="Bash",
        args={"cmd": "sleep 30"},
        result=None,
        status="cancelled",
    )
    row = tmp_db.conn.execute(
        "SELECT tool_status FROM memories WHERE id = ?", (mid,),
    ).fetchone()
    assert row["tool_status"] == "cancelled"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_record_tool_call_requires_tool_name(tmp_db):
    """Calling with type=tool_call (implicit) but no tool_name must raise."""
    with pytest.raises(ValueError, match="tool_name"):
        me.record_tool_call(
            tmp_db, tool_name="", args={}, result="ok",
        )
    with pytest.raises(ValueError, match="tool_name"):
        me.record_tool_call(
            tmp_db, tool_name="   ", args={}, result="ok",
        )


def test_record_tool_call_invalid_status_raises(tmp_db):
    with pytest.raises(ValueError, match="invalid tool_status"):
        me.record_tool_call(
            tmp_db, tool_name="bash", args={}, result="ok", status="weird",
        )


def test_tool_call_type_is_in_valid_memory_types():
    assert "tool_call" in me._VALID_MEMORY_TYPES


# ---------------------------------------------------------------------------
# Filtering and aggregation
# ---------------------------------------------------------------------------

def test_search_by_type_returns_only_tool_calls(tmp_db):
    # Seed a couple of tool calls + a regular memoria.
    me.record_tool_call(tmp_db, tool_name="bash", args={"cmd": "ls"}, result="a")
    me.record_tool_call(tmp_db, tool_name="grep", args={"pat": "x"}, result="b")
    tmp_db.conn.execute(
        "INSERT INTO memories (id, type, content, content_hash, importance, "
        "confidence, score, usage_count, user_signal, valid_from, "
        "metadata_json, created_at, updated_at) "
        "VALUES ('mem_seed_fact', 'fact', 'pi=3.14', 'h_seed_fact', 3, 0.9, "
        "0, 0, 0, '2026-01-01', '{}', '2026-01-01', '2026-01-01')"
    )
    tmp_db.conn.commit()

    rows = tmp_db.conn.execute(
        "SELECT type FROM memories WHERE type = 'tool_call' AND archived_at IS NULL"
    ).fetchall()
    assert len(rows) == 2
    assert all(r["type"] == "tool_call" for r in rows)


def test_summarize_tool_calls_groups_correctly(tmp_db):
    # bash: 2 success + 1 error  (success_rate ≈ 0.667)
    me.record_tool_call(tmp_db, tool_name="bash", args={"cmd": "a"}, result="1")
    me.record_tool_call(tmp_db, tool_name="bash", args={"cmd": "b"}, result="2")
    me.record_tool_call(
        tmp_db, tool_name="bash", args={"cmd": "c"}, result="boom", status="error",
    )
    # grep: 1 success + 1 cancelled
    me.record_tool_call(tmp_db, tool_name="grep", args={"pat": "x"}, result="hit")
    me.record_tool_call(
        tmp_db, tool_name="grep", args={"pat": "y"}, result=None, status="cancelled",
    )

    stats = me.summarize_tool_calls(tmp_db)
    by_name = {s.tool_name: s for s in stats}
    assert set(by_name) == {"bash", "grep"}
    assert by_name["bash"].count == 3
    assert by_name["bash"].success_count == 2
    assert by_name["bash"].error_count == 1
    assert abs(by_name["bash"].success_rate - 2 / 3) < 1e-6
    assert by_name["grep"].count == 2
    assert by_name["grep"].cancelled_count == 1

    # Filter by name returns only one.
    only_bash = me.summarize_tool_calls(tmp_db, "bash")
    assert len(only_bash) == 1
    assert only_bash[0].tool_name == "bash"


def test_get_tool_calls_for_conversation_filters(tmp_db):
    me.record_tool_call(
        tmp_db, tool_name="bash", args={"cmd": "ls"}, result="ok",
        conversation_id="conv_A",
    )
    me.record_tool_call(
        tmp_db, tool_name="grep", args={"pat": "x"}, result="hit",
        conversation_id="conv_A",
    )
    me.record_tool_call(
        tmp_db, tool_name="bash", args={"cmd": "pwd"}, result="/",
        conversation_id="conv_B",
    )
    # No conversation_id at all
    me.record_tool_call(tmp_db, tool_name="bash", args={"cmd": "id"}, result="root")

    rows_a = me.get_tool_calls_for_conversation(tmp_db, "conv_A")
    assert len(rows_a) == 2
    assert {r["tool_name"] for r in rows_a} == {"bash", "grep"}

    rows_b = me.get_tool_calls_for_conversation(tmp_db, "conv_B")
    assert len(rows_b) == 1

    rows_none = me.get_tool_calls_for_conversation(tmp_db, "")
    assert rows_none == []


# ---------------------------------------------------------------------------
# MCP wrapper end-to-end
# ---------------------------------------------------------------------------

def test_mcp_record_tool_call_end_to_end(tmp_db):
    args = {
        "tool_name": "bash",
        "args": {"cmd": "ls -la"},
        "result_summary": "10 files",
        "status": "success",
        "conversation_id": "conv_xyz",
        "importance": 3,
    }
    payload = mcp_tools.call_tool(tmp_db, "mcp_record_tool_call", args)
    assert payload["ok"] is True
    mid = payload["memory_id"]
    assert mid.startswith("mem_")
    assert "bash(" in payload["content"]

    # The row exists with the expected columns.
    row = tmp_db.conn.execute(
        "SELECT tool_name, tool_status, tool_args_json, importance, metadata_json "
        "FROM memories WHERE id = ?",
        (mid,),
    ).fetchone()
    assert row["tool_name"] == "bash"
    assert row["tool_status"] == "success"
    assert json.loads(row["tool_args_json"]) == {"cmd": "ls -la"}
    assert row["importance"] == 3
    assert json.loads(row["metadata_json"])["conversation_id"] == "conv_xyz"


def test_mcp_record_tool_call_validates_required(tmp_db):
    with pytest.raises(ValueError, match="tool_name"):
        mcp_tools.call_tool(
            tmp_db, "mcp_record_tool_call",
            {"tool_name": "", "args": {}, "result_summary": "x"},
        )
    with pytest.raises(ValueError, match="result_summary"):
        mcp_tools.call_tool(
            tmp_db, "mcp_record_tool_call",
            {"tool_name": "bash", "args": {}},
        )


def test_mcp_record_tool_call_schema_registered():
    """The new tool must appear in TOOL_SCHEMAS so tools/list advertises it."""
    names = {t["name"] for t in mcp_tools.TOOL_SCHEMAS}
    assert "mcp_record_tool_call" in names
    schema = next(t for t in mcp_tools.TOOL_SCHEMAS if t["name"] == "mcp_record_tool_call")
    required = set(schema["inputSchema"]["required"])
    assert {"tool_name", "args", "result_summary"}.issubset(required)
