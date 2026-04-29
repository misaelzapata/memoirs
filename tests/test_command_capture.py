"""Tests for command/tool_call context capture + ``memoirs commands`` CLI.

Covers:

* ``tool_call_extract.record_tool_calls_for_conversation`` enriches
  ``metadata_json`` with ``cwd`` / ``project_name`` / ``message_ordinal``
  / ``timestamp`` (read from the conversation + raw message).
* ``memoirs commands list`` filters by ``--project`` and ``--tool`` and
  emits a clean (no ``[tool_use:`` noise) table.
* ``memoirs commands stats`` aggregates by tool with success_rate.
* ``memoirs commands replay`` reconstructs the original Bash/Read args.
* ``thread_resume.resume_thread`` does **not** leak ``[tool_use:``
  fragments into ``last_decisions`` / ``pending_actions`` and surfaces
  ``recent_tool_calls`` from the corpus.
* ``memoirs current`` end-to-end prints the cleaned table.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from memoirs.db import MemoirsDB
from memoirs.engine import thread_resume as tr
from memoirs.engine import tool_call_extract as tce
from memoirs.models import RawConversation, RawMessage


def _assistant_with_tool_use(
    tool_id: str, name: str, inp: dict,
    *, cwd: str | None = None, ts: str | None = None,
) -> dict:
    obj: dict = {
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
    if cwd is not None:
        obj["cwd"] = cwd
    if ts is not None:
        obj["timestamp"] = ts
    return obj


def _user_with_tool_result(tool_id: str, content: str, *, is_error: bool = False) -> dict:
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
        "message": {"role": "user", "content": [block]},
    }


def _seed_conversation(
    db: MemoirsDB,
    *,
    conv_id: str,
    raw_messages: list[dict],
    cwd: str | None = None,
    project_name: str | None = None,
) -> str:
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
                created_at=raw.get("timestamp"),
                metadata={"cwd": raw.get("cwd")} if raw.get("cwd") else {},
                raw=raw,
            )
        )
    conv_md: dict = {"format": "claude_code_jsonl"}
    if cwd:
        conv_md["cwd"] = cwd
    if project_name:
        conv_md["project_name"] = project_name
    conv = RawConversation(
        external_id=conv_id,
        title="t",
        source_kind="claude_code",
        source_uri=f"mem://{conv_id}",
        messages=msgs,
        created_at=None,
        metadata=conv_md,
    )
    db.save_conversations(
        [conv],
        source_name=f"test-{conv_id}",
        source_kind="claude_code",
        source_uri=f"mem://{conv_id}",
        hash_value=None,
        mtime_ns=None,
        size_bytes=None,
    )
    row = db.conn.execute(
        "SELECT id FROM conversations WHERE external_id = ?", (conv_id,),
    ).fetchone()
    return row["id"]


# ---------------------------------------------------------------------------
# 1. Context capture in tool_call metadata_json
# ---------------------------------------------------------------------------


def test_extract_persists_cwd_and_project_in_metadata(tmp_db):
    raw = [
        _assistant_with_tool_use(
            "tu_a", "Bash", {"command": "ls memoirs/migrations"},
            cwd="/home/misael/Desktop/projects/memoirs",
            ts="2026-04-28T14:32:00Z",
        ),
        _user_with_tool_result("tu_a", "001.sql\n002.sql"),
    ]
    cid = _seed_conversation(
        tmp_db, conv_id="conv_ctx", raw_messages=raw,
        cwd="/home/misael/Desktop/projects/memoirs",
    )
    inserted = tce.record_tool_calls_for_conversation(tmp_db, cid)
    assert inserted == 1

    row = tmp_db.conn.execute(
        "SELECT metadata_json FROM memories WHERE type='tool_call' "
        "AND archived_at IS NULL"
    ).fetchone()
    md = json.loads(row["metadata_json"])
    assert md["cwd"] == "/home/misael/Desktop/projects/memoirs"
    assert md["project_name"] == "memoirs"
    assert md["conversation_id"] == cid
    assert md["message_ordinal"] == 0
    assert "timestamp" in md and md["timestamp"]


def test_extract_falls_back_to_conversation_cwd(tmp_db):
    """When a per-message cwd is absent, the conversation-level cwd wins."""
    raw = [
        _assistant_with_tool_use("tu_b", "Read", {"file_path": "/x/y/z.py"}),
        _user_with_tool_result("tu_b", "print('hi')"),
    ]
    cid = _seed_conversation(
        tmp_db, conv_id="conv_fb", raw_messages=raw,
        cwd="/home/misael/Desktop/projects/gocracker",
    )
    tce.record_tool_calls_for_conversation(tmp_db, cid)
    row = tmp_db.conn.execute(
        "SELECT metadata_json FROM memories WHERE type='tool_call'"
    ).fetchone()
    md = json.loads(row["metadata_json"])
    assert md["cwd"] == "/home/misael/Desktop/projects/gocracker"
    assert md["project_name"] == "gocracker"


# ---------------------------------------------------------------------------
# 2. memoirs commands list / stats / replay
# ---------------------------------------------------------------------------


def _run_cli(db_path: Path, *args: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "memoirs", "--db", str(db_path), *args],
        capture_output=True, text=True, check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _seed_two_projects(tmp_db: MemoirsDB) -> None:
    raw_a = [
        _assistant_with_tool_use(
            "tu_1", "Bash", {"command": "pytest -x"},
            cwd="/home/u/Desktop/projects/memoirs",
            ts="2026-04-28T14:00:00Z",
        ),
        _user_with_tool_result("tu_1", "ok"),
        _assistant_with_tool_use(
            "tu_2", "Read", {"file_path": "/home/u/Desktop/projects/memoirs/README.md"},
            cwd="/home/u/Desktop/projects/memoirs",
            ts="2026-04-28T14:01:00Z",
        ),
        _user_with_tool_result("tu_2", "# Memoirs"),
    ]
    cid_a = _seed_conversation(
        tmp_db, conv_id="conv_a", raw_messages=raw_a,
        cwd="/home/u/Desktop/projects/memoirs",
    )
    tce.record_tool_calls_for_conversation(tmp_db, cid_a)

    raw_b = [
        _assistant_with_tool_use(
            "tu_3", "Bash", {"command": "git push origin main"},
            cwd="/home/u/Desktop/projects/gocracker",
            ts="2026-04-28T13:45:00Z",
        ),
        _user_with_tool_result("tu_3", "Auth failed", is_error=True),
    ]
    cid_b = _seed_conversation(
        tmp_db, conv_id="conv_b", raw_messages=raw_b,
        cwd="/home/u/Desktop/projects/gocracker",
    )
    tce.record_tool_calls_for_conversation(tmp_db, cid_b)


def test_commands_list_filters_by_project(tmp_db, tmp_path):
    _seed_two_projects(tmp_db)
    tmp_db.close()
    db_path = Path(tmp_db.path) if hasattr(tmp_db, "path") else None
    if db_path is None:
        db_path = Path(str(tmp_db.conn.execute("PRAGMA database_list").fetchone()["file"]))

    rc, out, err = _run_cli(db_path, "commands", "list", "--project", "memoirs", "--json")
    assert rc == 0, err
    data = json.loads(out)
    assert len(data) == 2
    assert all(d["metadata"]["project_name"] == "memoirs" for d in data)
    assert {d["tool_name"] for d in data} == {"Bash", "Read"}


def test_commands_list_filters_by_tool(tmp_db, tmp_path):
    _seed_two_projects(tmp_db)
    db_path = Path(str(tmp_db.path)) if hasattr(tmp_db, "path") else None
    tmp_db.close()
    if db_path is None:
        # fallback: construct from same tmp_path
        db_path = next(tmp_path.glob("memoirs.sqlite"), None) or (tmp_path / "memoirs.sqlite")

    rc, out, err = _run_cli(db_path, "commands", "list", "--tool", "bash", "--json")
    assert rc == 0, err
    data = json.loads(out)
    assert all(d["tool_name"].lower() == "bash" for d in data)
    assert len(data) == 2


def test_commands_list_table_has_no_tool_use_noise(tmp_db):
    _seed_two_projects(tmp_db)
    db_path = tmp_db.path if hasattr(tmp_db, "path") else None
    tmp_db.close()
    rc, out, err = _run_cli(Path(str(db_path)), "commands", "list", "--limit", "10")
    assert rc == 0, err
    assert "[tool_use" not in out
    assert "[tool_result" not in out
    # tool names appear as a column header value
    assert "Bash" in out and "Read" in out


def test_commands_stats_aggregates(tmp_db):
    _seed_two_projects(tmp_db)
    db_path = tmp_db.path
    tmp_db.close()
    rc, out, err = _run_cli(Path(str(db_path)), "commands", "stats", "--json")
    assert rc == 0, err
    data = json.loads(out)
    by_tool = {d["tool_name"]: d for d in data}
    assert by_tool["Bash"]["count"] == 2
    assert by_tool["Bash"]["success_count"] == 1
    assert by_tool["Bash"]["error_count"] == 1
    assert by_tool["Bash"]["success_rate"] == 0.5
    assert by_tool["Read"]["count"] == 1
    assert by_tool["Read"]["success_rate"] == 1.0


def test_commands_replay_reconstructs_bash_command(tmp_db):
    raw = [
        _assistant_with_tool_use(
            "tu_r", "Bash", {"command": "ls -la /tmp"},
            cwd="/home/u/Desktop/projects/memoirs",
        ),
        _user_with_tool_result("tu_r", "x"),
    ]
    cid = _seed_conversation(
        tmp_db, conv_id="conv_replay", raw_messages=raw,
        cwd="/home/u/Desktop/projects/memoirs",
    )
    tce.record_tool_calls_for_conversation(tmp_db, cid)
    row = tmp_db.conn.execute(
        "SELECT id FROM memories WHERE type='tool_call' AND tool_name='Bash'"
    ).fetchone()
    full_id = row["id"]
    db_path = tmp_db.path
    tmp_db.close()
    rc, out, err = _run_cli(Path(str(db_path)), "commands", "replay", full_id[:12])
    assert rc == 0, err
    assert out.strip() == "ls -la /tmp"


# ---------------------------------------------------------------------------
# 3. resume_thread / current — drop tool_use noise
# ---------------------------------------------------------------------------


def test_resume_thread_strips_tool_use_noise_from_pending(tmp_db):
    raw = [
        {
            "type": "user",
            "uuid": "u-prose",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "we need to update README"}],
            },
        },
        _assistant_with_tool_use(
            "tu_n", "TodoWrite",
            {"todos": [{"task": "fix flaky test_decide_action"}]},
            cwd="/home/u/Desktop/projects/memoirs",
        ),
        _user_with_tool_result("tu_n", "ok"),
    ]
    cid = _seed_conversation(
        tmp_db, conv_id="conv_noise", raw_messages=raw,
        cwd="/home/u/Desktop/projects/memoirs",
    )
    tce.record_tool_calls_for_conversation(tmp_db, cid)

    payload = tr.resume_thread(tmp_db, cid, generate_if_missing=False)
    pending = payload.get("pending_actions") or []
    decisions = payload.get("last_decisions") or []
    for line in pending + decisions:
        assert "[tool_use" not in line
        assert "[tool_result" not in line


def test_resume_thread_surfaces_recent_tool_calls_from_corpus(tmp_db):
    raw = [
        _assistant_with_tool_use(
            "tu_c1", "Bash", {"command": "ls"},
            cwd="/home/u/Desktop/projects/memoirs",
            ts="2026-04-28T14:00:00Z",
        ),
        _user_with_tool_result("tu_c1", "file"),
        _assistant_with_tool_use(
            "tu_c2", "Read", {"file_path": "/x/README.md"},
            cwd="/home/u/Desktop/projects/memoirs",
            ts="2026-04-28T14:01:00Z",
        ),
        _user_with_tool_result("tu_c2", "..."),
    ]
    cid = _seed_conversation(
        tmp_db, conv_id="conv_recent", raw_messages=raw,
        cwd="/home/u/Desktop/projects/memoirs",
    )
    tce.record_tool_calls_for_conversation(tmp_db, cid)

    payload = tr.resume_thread(tmp_db, cid, generate_if_missing=False)
    recent = payload.get("recent_tool_calls") or []
    tools = {r["tool_name"] for r in recent}
    assert {"Bash", "Read"}.issubset(tools)
    for r in recent:
        assert "[tool_use" not in (r.get("content") or "")


def test_resume_thread_uses_task_memories_for_pending(tmp_db):
    raw = [
        _assistant_with_tool_use("tu_t", "Bash", {"command": "echo hi"}),
        _user_with_tool_result("tu_t", "hi"),
    ]
    cid = _seed_conversation(tmp_db, conv_id="conv_tasks", raw_messages=raw)

    md = json.dumps({"conversation_id": cid})
    tmp_db.conn.execute(
        "INSERT INTO memories (id, type, content, content_hash, importance, "
        "  confidence, score, usage_count, user_signal, valid_from, "
        "  metadata_json, created_at, updated_at) "
        "VALUES (?, 'task', ?, ?, 3, 0.9, 0, 0, 0, ?, ?, ?, ?)",
        (
            "mem_task_1", "fix flaky test_decide_action", "h_task_1",
            "2026-04-28T14:00:00Z", md,
            "2026-04-28T14:00:00Z", "2026-04-28T14:00:00Z",
        ),
    )
    tmp_db.conn.commit()

    payload = tr.resume_thread(tmp_db, cid, generate_if_missing=False)
    pending = payload.get("pending_actions") or []
    assert any("flaky test_decide_action" in p for p in pending)


def test_current_cmd_table_has_no_raw_tool_use(tmp_db, tmp_path, monkeypatch):
    """End-to-end: ``memoirs current`` doesn't print ``[tool_use:`` lines.

    Drives through ``_print_resume_payload`` after we hand it a payload
    that exercises the renderer (no need to set up a real Claude Code
    JSONL — we test the print helper directly, which is what `current`
    invokes).
    """
    import io
    from contextlib import redirect_stdout
    from memoirs.cli import _print_resume_payload

    payload = {
        "conversation_id": "conv_render",
        "summary": "Testing render.",
        "generated_at": None,
        "message_count_at_summary": 4,
        "salient_memories": [],
        "last_decisions": ["decided to switch curator to Qwen3"],
        "pending_actions": ["fix flaky test_decide_action"],
        "recent_tool_calls": [
            {
                "id": "mem_x", "tool_name": "Bash", "tool_status": "success",
                "tool_args_json": json.dumps({"command": "ls memoirs/migrations/"}),
                "content": "Bash(...) → ok",
                "created_at": "2026-04-28T14:32:00Z",
                "metadata_json": json.dumps({
                    "project_name": "memoirs",
                    "timestamp": "2026-04-28T14:32:00Z",
                }),
            },
            {
                "id": "mem_y", "tool_name": "Bash", "tool_status": "error",
                "tool_args_json": json.dumps({"command": "git push origin main"}),
                "content": "Bash(...) → ERROR",
                "created_at": "2026-04-28T13:45:00Z",
                "metadata_json": json.dumps({
                    "project_name": "gocracker",
                    "timestamp": "2026-04-28T13:45:00Z",
                }),
            },
        ],
        "project_context": None,
    }
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_resume_payload(payload)
    out = buf.getvalue()
    assert "[tool_use" not in out
    assert "[tool_result" not in out
    assert "ls memoirs/migrations/" in out
    assert "git push origin main" in out
    assert "Recent commands" in out
