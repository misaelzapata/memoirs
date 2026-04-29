"""Tests for the auto-resume thread feature (P-resume).

Covers:
* migration 011 round-trip (table + indexes appear / disappear cleanly).
* ``detect_idle_conversations`` only returns conversations older than the
  cutoff and skips those with a fresh summary.
* ``generate_thread_summary`` with a mocked curator persists the row and
  returns the parsed payload.
* ``resume_thread`` returns the documented shape and generates on-the-fly
  when no summary row exists yet.
* The sleep cron job caps work at ``DEFAULT_MAX_CONVS_PER_TICK`` and
  short-circuits idempotently on already-summarized conversations.
* The MCP tool dispatches end-to-end with a mock LLM.
* Auto-detect helpers (``encode_cwd_for_claude``,
  ``find_latest_jsonl_for_cwd``, ``find_conversation_id_for_cwd``) handle
  the Claude Code project layout.
* ``memoirs current`` CLI smoke test on a synthetic DB.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from memoirs import migrations
from memoirs.db import MemoirsDB, content_hash, stable_id, utc_now
from memoirs.engine import thread_resume
from memoirs.engine.thread_resume import (
    DEFAULT_IDLE_MINUTES,
    DEFAULT_MAX_CONVS_PER_TICK,
    detect_idle_conversations,
    encode_cwd_for_claude,
    find_conversation_id_for_cwd,
    find_latest_jsonl_for_cwd,
    generate_thread_summary,
    latest_thread_summary,
    resume_thread,
    sleep_thread_summaries_job,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _seed_conversation(
    db: MemoirsDB,
    *,
    conv_id: str,
    last_message_minutes_ago: float,
    n_messages: int = 6,
    title: str = "test",
    metadata: dict | None = None,
) -> str:
    """Insert a source + conversation + N messages, with the most recent message
    backdated by ``last_message_minutes_ago`` minutes.
    """
    src_uri = f"test://{conv_id}"
    now = datetime.now(timezone.utc)
    db.conn.execute(
        "INSERT OR IGNORE INTO sources (uri, kind, name, created_at, updated_at) "
        "VALUES (?, 'test', ?, ?, ?)",
        (src_uri, conv_id, now.isoformat(), now.isoformat()),
    )
    src_row = db.conn.execute(
        "SELECT id FROM sources WHERE uri = ?", (src_uri,),
    ).fetchone()
    md = metadata or {}
    db.conn.execute(
        "INSERT OR IGNORE INTO conversations "
        "(id, source_id, external_id, title, created_at, updated_at, "
        " message_count, metadata_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            conv_id, src_row["id"], conv_id, title,
            now.isoformat(), now.isoformat(),
            n_messages,
            json.dumps(md, ensure_ascii=False),
        ),
    )

    base = now - timedelta(minutes=last_message_minutes_ago + n_messages)
    for i in range(n_messages):
        ts = (base + timedelta(minutes=i)).isoformat()
        if i == n_messages - 1:
            # Force the last message to land exactly at ``last_message_minutes_ago``.
            ts = (now - timedelta(minutes=last_message_minutes_ago)).isoformat()
        role = "user" if i % 2 == 0 else "assistant"
        content = (
            f"msg {i} role={role} -- "
            "Working on memoirs auto-resume; decided to use Qwen for the curator. "
            "TODO: implement migration 011 and wire MCP. Pending: review fix #1."
        )
        mid = stable_id("msg", conv_id, str(i))
        db.conn.execute(
            "INSERT OR IGNORE INTO messages "
            "(id, conversation_id, role, content, ordinal, created_at, "
            " content_hash, raw_json, is_active, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, '{}', 1, ?, ?)",
            (
                mid, conv_id, role, content, i, ts,
                content_hash(content), ts, ts,
            ),
        )
    db.conn.commit()
    return conv_id


@pytest.fixture
def synth_db(tmp_path: Path) -> MemoirsDB:
    db = MemoirsDB(tmp_path / "thread_resume.sqlite")
    db.init()
    yield db
    db.close()


# ---------------------------------------------------------------------------
# Migration 011 round-trip
# ---------------------------------------------------------------------------

def test_migration_011_thread_summaries_round_trip(tmp_path: Path):
    """011 must add ``thread_summaries`` + the two indexes, accept INSERTs,
    and survive a down/up round-trip without data loss on neighboring tables.
    """
    db_path = tmp_path / "m11.sqlite"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        migrations.run_pending_migrations(conn)
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "thread_summaries" in tables

        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(thread_summaries)"
        ).fetchall()}
        assert {
            "id", "conversation_id", "summary", "generated_at",
            "message_count_at_summary", "last_message_ts",
            "pending_actions_json", "salient_entity_ids_json", "user_id",
        } <= cols

        idx = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='thread_summaries'"
            ).fetchall()
        }
        assert "idx_thread_summaries_conv" in idx
        assert "idx_thread_summaries_recent" in idx

        # Insert a synthetic row + verify retrieval.
        conn.execute(
            "INSERT INTO thread_summaries "
            "(conversation_id, summary, generated_at, message_count_at_summary, "
            " last_message_ts, pending_actions_json, salient_entity_ids_json) "
            "VALUES ('conv_1', 'durable summary', '2026-04-28T00:00:00+00:00', "
            " 12, '2026-04-27T23:59:00+00:00', '[\"todo\"]', '[]')"
        )
        conn.commit()
        row = conn.execute(
            "SELECT summary, message_count_at_summary FROM thread_summaries "
            "WHERE conversation_id = 'conv_1'"
        ).fetchone()
        assert row is not None
        assert row["summary"] == "durable summary"
        assert row["message_count_at_summary"] == 12

        # Down + back up.
        target = migrations.target_version()
        steps_back = max(1, target - 10)
        migrations.rollback(conn, steps=steps_back)
        tables_after = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "thread_summaries" not in tables_after

        migrations.run_pending_migrations(conn)
        tables_again = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "thread_summaries" in tables_again
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# detect_idle_conversations
# ---------------------------------------------------------------------------

def test_detect_idle_only_returns_paused_conversations(synth_db: MemoirsDB):
    """Idle conversations (last activity > cutoff) appear; recent ones do not."""
    _seed_conversation(synth_db, conv_id="conv_idle", last_message_minutes_ago=120)
    _seed_conversation(synth_db, conv_id="conv_recent", last_message_minutes_ago=5)

    result = detect_idle_conversations(synth_db, idle_minutes=DEFAULT_IDLE_MINUTES)
    ids = {entry["conversation_id"] for entry in result}
    assert "conv_idle" in ids
    assert "conv_recent" not in ids

    # Each entry exposes the documented fields.
    idle_entry = next(e for e in result if e["conversation_id"] == "conv_idle")
    assert idle_entry["message_count"] >= 1
    assert idle_entry["age_minutes"] > DEFAULT_IDLE_MINUTES
    assert idle_entry["last_message_ts"]


def test_detect_skips_conversations_with_fresh_summary(synth_db: MemoirsDB):
    """Once we record a summary newer than the latest message, the
    conversation should drop out of the idle queue (idempotency)."""
    _seed_conversation(synth_db, conv_id="conv_done", last_message_minutes_ago=200)

    first = detect_idle_conversations(synth_db, idle_minutes=DEFAULT_IDLE_MINUTES)
    assert any(e["conversation_id"] == "conv_done" for e in first)

    # Persist a synthetic summary with a newer last_message_ts.
    future_ts = datetime.now(timezone.utc).isoformat()
    synth_db.conn.execute(
        "INSERT INTO thread_summaries "
        "(conversation_id, summary, generated_at, message_count_at_summary, "
        " last_message_ts, pending_actions_json, salient_entity_ids_json) "
        "VALUES (?, ?, ?, ?, ?, '[]', '[]')",
        ("conv_done", "already covered", future_ts, 6, future_ts),
    )
    synth_db.conn.commit()

    second = detect_idle_conversations(synth_db, idle_minutes=DEFAULT_IDLE_MINUTES)
    assert not any(e["conversation_id"] == "conv_done" for e in second)


# ---------------------------------------------------------------------------
# generate_thread_summary (mock LLM)
# ---------------------------------------------------------------------------

class _MockLLM:
    """Stand-in for llama-cpp's Llama. Returns a canned summary blob."""

    def __init__(self, text: str):
        self._text = text
        self.calls = 0

    def create_completion(self, prompt: str, max_tokens: int = 256, **kwargs):
        self.calls += 1
        return {"choices": [{"text": self._text}]}

    # gemma_summarize_project's _build helper calls _count_tokens(llm, ...).
    # The real helper falls back to a char/4 estimate on Exception, so we
    # don't need a tokenizer here — but do expose ``n_ctx`` so the
    # neighbor-block budget logic doesn't blow up.
    def n_ctx(self) -> int:
        return 4096

    def tokenize(self, text):  # pragma: no cover — used only when called
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")
        return list(range(max(1, len(text) // 4)))


def test_generate_thread_summary_persists_with_mock_llm(synth_db: MemoirsDB, monkeypatch):
    """With a curator that returns a valid summary, the row is persisted."""
    cid = _seed_conversation(synth_db, conv_id="conv_gen", last_message_minutes_ago=90)

    fake_text = (
        "Memoirs team is implementing auto-resume for paused chat threads. "
        "Decision: use Qwen as curator and persist summaries to a new "
        "thread_summaries table; pending: review fix #1 and bench latency."
    )
    mock_llm = _MockLLM(fake_text)

    from memoirs.engine import curator as curator_mod
    monkeypatch.setattr(curator_mod, "_have_curator", lambda: True)

    row = generate_thread_summary(synth_db, cid, llm=mock_llm)
    assert row is not None
    assert row["conversation_id"] == cid
    assert row["summary"]
    # Should be the LLM's text (or a trimmed version of it).
    assert "auto-resume" in row["summary"].lower() or "memoirs" in row["summary"].lower()
    assert row["message_count_at_summary"] >= 1
    assert isinstance(row.get("pending_actions"), list)
    # The mock LLM should have been invoked.
    assert mock_llm.calls >= 1


def test_generate_thread_summary_falls_back_to_heuristic(synth_db: MemoirsDB, monkeypatch):
    """When the LLM is unavailable (or rejects every retry) we still get a
    non-empty summary via the heuristic fallback."""
    cid = _seed_conversation(synth_db, conv_id="conv_fb", last_message_minutes_ago=200)

    from memoirs.engine import curator as curator_mod
    monkeypatch.setattr(curator_mod, "_have_curator", lambda: False)

    row = generate_thread_summary(synth_db, cid)
    assert row is not None
    assert row["summary"]
    # The heuristic fallback should be deterministic on length: the helper
    # caps at SUMMARY_MAX_CHARS so we just sanity-check non-empty.
    assert len(row["summary"]) <= 500


# ---------------------------------------------------------------------------
# resume_thread
# ---------------------------------------------------------------------------

def test_resume_thread_returns_documented_shape(synth_db: MemoirsDB):
    """Even without a persisted summary, ``resume_thread`` produces the
    documented payload by generating one via the heuristic path."""
    cid = _seed_conversation(synth_db, conv_id="conv_shape", last_message_minutes_ago=120)

    payload = resume_thread(synth_db, cid)
    assert payload["conversation_id"] == cid
    assert "summary" in payload
    assert "salient_memories" in payload
    assert "last_decisions" in payload
    assert "pending_actions" in payload
    # heuristic fallback must produce a non-empty summary on a non-empty conv
    assert payload["summary"]
    assert isinstance(payload["last_decisions"], list)
    assert isinstance(payload["pending_actions"], list)


def test_resume_thread_generates_on_the_fly_when_missing(synth_db: MemoirsDB):
    cid = _seed_conversation(synth_db, conv_id="conv_otf", last_message_minutes_ago=60)
    # Sanity: no thread_summaries row exists yet.
    assert latest_thread_summary(synth_db, cid) is None

    payload = resume_thread(synth_db, cid, generate_if_missing=True)
    assert payload["summary"]
    # ... and now a row exists.
    persisted = latest_thread_summary(synth_db, cid)
    assert persisted is not None
    assert persisted["summary"] == payload["summary"]


def test_resume_thread_no_generate_returns_empty_summary(synth_db: MemoirsDB):
    cid = _seed_conversation(synth_db, conv_id="conv_noop", last_message_minutes_ago=60)
    payload = resume_thread(synth_db, cid, generate_if_missing=False)
    assert payload["summary"] is None
    # ...but pending_actions still derived from messages
    assert isinstance(payload["pending_actions"], list)


# ---------------------------------------------------------------------------
# Sleep cron cap
# ---------------------------------------------------------------------------

def test_sleep_thread_summaries_caps_per_tick(synth_db: MemoirsDB, monkeypatch):
    """The cron job must not summarize more than ``max_convs`` per tick."""
    # Seed 12 idle conversations.
    for i in range(12):
        _seed_conversation(
            synth_db, conv_id=f"conv_cron_{i:02d}",
            last_message_minutes_ago=120 + i,
        )

    # Disable the LLM path — heuristic fallback is enough for the cap test.
    from memoirs.engine import curator as curator_mod
    monkeypatch.setattr(curator_mod, "_have_curator", lambda: False)

    result = sleep_thread_summaries_job(
        synth_db, idle_minutes=DEFAULT_IDLE_MINUTES, max_convs=10,
    )
    assert result["summarized"] <= 10
    assert result["max_convs_per_tick"] == 10
    # Detected count is bounded by the requested limit (max_convs * 2).
    assert result["detected"] >= result["summarized"]


def test_sleep_thread_summaries_idempotent_on_rerun(synth_db: MemoirsDB, monkeypatch):
    cid = _seed_conversation(synth_db, conv_id="conv_idem", last_message_minutes_ago=200)

    from memoirs.engine import curator as curator_mod
    monkeypatch.setattr(curator_mod, "_have_curator", lambda: False)

    first = sleep_thread_summaries_job(synth_db, idle_minutes=DEFAULT_IDLE_MINUTES)
    assert first["summarized"] == 1

    # Second run: nothing new because the summary's last_message_ts is now
    # >= the conversation's latest message.
    second = sleep_thread_summaries_job(synth_db, idle_minutes=DEFAULT_IDLE_MINUTES)
    assert second["summarized"] == 0


# ---------------------------------------------------------------------------
# Auto-detect (cwd → JSONL → conversation_id)
# ---------------------------------------------------------------------------

def test_encode_cwd_for_claude_handles_absolute_paths():
    assert encode_cwd_for_claude("/home/x/code") == "-home-x-code"
    # Trailing slash still encodes consistently.
    assert encode_cwd_for_claude("/home/x/code/") == "-home-x-code"


def test_find_latest_jsonl_returns_most_recent(tmp_path: Path, monkeypatch):
    """Auto-detect picks the JSONL with the highest mtime in the encoded dir."""
    fake_root = tmp_path / "claude_root"
    encoded = encode_cwd_for_claude(tmp_path)
    proj = fake_root / encoded
    proj.mkdir(parents=True)
    older = proj / "session-old.jsonl"
    newer = proj / "session-new.jsonl"
    older.write_text("{}\n", encoding="utf-8")
    newer.write_text("{}\n", encoding="utf-8")
    # Backdate the older one explicitly.
    import os as _os
    older_ts = time.time() - 3600
    _os.utime(older, (older_ts, older_ts))

    result = find_latest_jsonl_for_cwd(tmp_path, claude_root=fake_root)
    assert result is not None
    assert result.name == "session-new.jsonl"


def test_find_conversation_id_for_cwd(synth_db: MemoirsDB, tmp_path: Path, monkeypatch):
    """Auto-detect end-to-end: cwd → JSONL → conversation by external_id."""
    fake_root = tmp_path / "claude_root"
    encoded = encode_cwd_for_claude(tmp_path)
    proj = fake_root / encoded
    proj.mkdir(parents=True)

    session_id = "session-auto-001"
    jsonl = proj / f"{session_id}.jsonl"
    jsonl.write_text("{}\n", encoding="utf-8")

    # Seed a matching conversation row (external_id == session_id).
    src_uri = str(jsonl)
    now_iso = utc_now()
    synth_db.conn.execute(
        "INSERT OR IGNORE INTO sources (uri, kind, name, created_at, updated_at) "
        "VALUES (?, 'claude_code', ?, ?, ?)",
        (src_uri, session_id, now_iso, now_iso),
    )
    src_row = synth_db.conn.execute(
        "SELECT id FROM sources WHERE uri = ?", (src_uri,),
    ).fetchone()
    synth_db.conn.execute(
        "INSERT OR IGNORE INTO conversations "
        "(id, source_id, external_id, title, created_at, updated_at, "
        " message_count, metadata_json) "
        "VALUES (?, ?, ?, ?, ?, ?, 0, '{}')",
        ("conv_auto", src_row["id"], session_id, "auto", now_iso, now_iso),
    )
    synth_db.conn.commit()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(thread_resume, "find_latest_jsonl_for_cwd",
                        lambda cwd=None: jsonl)

    cid = find_conversation_id_for_cwd(synth_db)
    assert cid == "conv_auto"


# ---------------------------------------------------------------------------
# MCP tool end-to-end
# ---------------------------------------------------------------------------

def test_mcp_resume_thread_with_mock_llm(synth_db: MemoirsDB, monkeypatch):
    """Dispatch the MCP tool with an explicit conv id + mock curator."""
    from memoirs.mcp import tools as mcp_tools
    from memoirs.engine import curator as curator_mod

    cid = _seed_conversation(synth_db, conv_id="conv_mcp", last_message_minutes_ago=180)

    fake_text = (
        "Memoirs team rolling out auto-resume threads in conv_mcp; "
        "decided to ship migration 011 and the new MCP wrapper. "
        "Pending: bench Qwen latency over 100-message conversations."
    )
    mock_llm = _MockLLM(fake_text)
    monkeypatch.setattr(curator_mod, "_have_curator", lambda: True)
    # Patch _get_llm so the engine module picks up our mock.
    monkeypatch.setattr(curator_mod, "_get_llm", lambda: mock_llm)

    result = mcp_tools.call_tool(
        synth_db, "mcp_resume_thread",
        {"conversation_id": cid, "salient_limit": 4},
    )
    assert result["ok"] is True
    assert result["conversation_id"] == cid
    assert result["summary"]
    assert "salient_memories" in result
    assert "pending_actions" in result


def test_mcp_resume_thread_auto_detect_no_jsonl(synth_db: MemoirsDB, tmp_path: Path,
                                                 monkeypatch):
    """When no conversation_id is supplied AND no JSONL matches the cwd we
    return ok=False with a reason, NOT a hard exception."""
    from memoirs.mcp import tools as mcp_tools

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(thread_resume, "find_latest_jsonl_for_cwd",
                        lambda cwd=None: None)

    result = mcp_tools.call_tool(synth_db, "mcp_resume_thread", {})
    assert result["ok"] is False
    assert "reason" in result


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------

def _run_cli(*cli_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "memoirs", *cli_args],
        capture_output=True, text=True, check=False,
    )


def test_cli_current_smoke_with_no_match(tmp_path: Path):
    """``memoirs current`` on a fresh DB with no JSONL match returns 0 and
    a friendly message rather than crashing."""
    db_path = tmp_path / "cli_current.sqlite"
    # Initialize first so migrations are applied.
    init_result = _run_cli("--db", str(db_path), "init")
    assert init_result.returncode == 0, init_result.stderr

    result = _run_cli("--db", str(db_path), "current", "--json")
    # We may or may not be in a Claude Code project — but the command must
    # exit 0 either way (no JSONL is not a failure).
    assert result.returncode == 0, result.stderr
    # Output is JSON (object) when --json passed.
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)


def test_cli_resume_explicit_conversation(tmp_path: Path):
    """``memoirs resume <conv_id>`` with a synthetic conversation prints
    the resume payload and exits 0."""
    db_path = tmp_path / "cli_resume.sqlite"
    db = MemoirsDB(db_path)
    db.init()
    try:
        _seed_conversation(db, conv_id="conv_cli_explicit",
                           last_message_minutes_ago=180)
    finally:
        db.close()

    result = _run_cli("--db", str(db_path), "resume", "conv_cli_explicit", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["conversation_id"] == "conv_cli_explicit"
    assert payload["summary"]


# ---------------------------------------------------------------------------
# Sleep cron registration
# ---------------------------------------------------------------------------

def test_thread_summaries_job_registered_in_sleep():
    """The new job must be wired into the sleep scheduler's JOB_NAMES + _JOB_FNS."""
    from memoirs.engine import sleep_consolidation as sc
    assert "thread_summaries" in sc.JOB_NAMES
    assert "thread_summaries" in sc._JOB_FNS
