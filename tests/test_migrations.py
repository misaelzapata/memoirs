"""Tests for the versioned migration system.

Covers:
* discovery (numeric ordering, well-formed metadata)
* fresh DB → applies all pending migrations → user_version == target
* idempotence (running migrate twice is a no-op)
* round-trip: up then down restores user_version
* legacy DB (tables already exist, user_version=0) is upgraded without error
* fresh DB schema after the runner == schema produced by ``MemoirsDB.init()``
* CLI ``memoirs db version`` / ``memoirs db migrate`` / ``--rollback``
"""
from __future__ import annotations

import importlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from memoirs import migrations
from memoirs.db import MemoirsDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_raw(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _schema_snapshot(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Return a deterministic structural summary of the DB schema.

    Keyed by table; value is the sorted list of ``(column, type, notnull,
    pk)`` tuples plus the sorted list of indexes for that table. This is
    what the acceptance test compares.
    """
    out: dict[str, list[str]] = {}
    tables = [
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    ]
    for table in tables:
        cols = sorted(
            f"{r['name']}|{r['type']}|notnull={r['notnull']}|pk={r['pk']}|dflt={r['dflt_value']}"
            for r in conn.execute(f"PRAGMA table_info({table})").fetchall()
        )
        idx = sorted(
            r[0] for r in conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' "
                "AND tbl_name = ? AND sql IS NOT NULL",
                (table,),
            ).fetchall()
        )
        out[table] = cols + [f"INDEX:{s}" for s in idx]
    return out


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_discover_finds_baseline():
    migs = migrations.discover_migrations()
    assert migs, "expected at least one migration"
    versions = [m.version for m in migs]
    assert versions == sorted(versions), "migrations must be sorted by version"
    assert versions[0] == 1, "baseline must be version 1"
    baseline = migs[0]
    assert baseline.name == "initial"
    assert callable(baseline.up) and callable(baseline.down)


def test_target_version_matches_last_migration():
    migs = migrations.discover_migrations()
    assert migrations.target_version() == migs[-1].version


# ---------------------------------------------------------------------------
# Apply / idempotence / rollback
# ---------------------------------------------------------------------------

def test_run_pending_on_fresh_db_applies_all(tmp_path: Path):
    db_path = tmp_path / "fresh.sqlite"
    conn = _open_raw(db_path)
    try:
        assert migrations.current_version(conn) == 0
        applied = migrations.run_pending_migrations(conn)
        assert applied == [m.version for m in migrations.discover_migrations()]
        assert migrations.current_version(conn) == migrations.target_version()
    finally:
        conn.close()


def test_run_pending_is_idempotent(tmp_path: Path):
    db_path = tmp_path / "idem.sqlite"
    conn = _open_raw(db_path)
    try:
        migrations.run_pending_migrations(conn)
        first = migrations.current_version(conn)
        applied = migrations.run_pending_migrations(conn)
        assert applied == []
        assert migrations.current_version(conn) == first
    finally:
        conn.close()


def test_migrate_to_zero_then_back(tmp_path: Path):
    db_path = tmp_path / "roundtrip.sqlite"
    conn = _open_raw(db_path)
    try:
        migrations.run_pending_migrations(conn)
        target = migrations.target_version()
        all_versions = [m.version for m in migrations.discover_migrations()]

        rolled = migrations.migrate_to(conn, 0)
        assert all(v < 0 for v in rolled)
        assert migrations.current_version(conn) == 0

        applied = migrations.migrate_to(conn, target)
        assert applied == all_versions
        assert migrations.current_version(conn) == target
    finally:
        conn.close()


def test_rollback_one_step(tmp_path: Path):
    db_path = tmp_path / "rollback.sqlite"
    conn = _open_raw(db_path)
    try:
        migrations.run_pending_migrations(conn)
        all_versions = [m.version for m in migrations.discover_migrations()]
        previous = all_versions[-2] if len(all_versions) >= 2 else 0
        rolled = migrations.rollback(conn, steps=1)
        assert rolled == [-migrations.target_version()]
        assert migrations.current_version(conn) == previous
    finally:
        conn.close()


def test_migrate_to_unknown_version_raises(tmp_path: Path):
    db_path = tmp_path / "bad.sqlite"
    conn = _open_raw(db_path)
    try:
        migrations.run_pending_migrations(conn)
        with pytest.raises(ValueError):
            migrations.migrate_to(conn, 999)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Legacy DB compatibility
# ---------------------------------------------------------------------------

def test_legacy_db_with_tables_but_no_version_upgrades(tmp_path: Path):
    """Simulate a DB created by the pre-migrations code path: tables exist
    but user_version was never bumped (or was reset to 0). The runner must
    succeed without dropping data.
    """
    initial = importlib.import_module("memoirs.migrations.001_initial")

    db_path = tmp_path / "legacy.sqlite"
    conn = _open_raw(db_path)
    try:
        # Create the schema directly (simulating a legacy install).
        conn.executescript(initial.SCHEMA)
        # Force user_version back to 0 to simulate a DB from before the
        # migration system existed.
        conn.execute("PRAGMA user_version = 0")
        conn.commit()

        # Insert a sentinel row so we can detect data loss.
        conn.execute(
            "INSERT INTO sources (uri, kind, name, created_at, updated_at) "
            "VALUES ('legacy://x', 'test', 'x', '2026-01-01', '2026-01-01')"
        )
        conn.commit()

        applied = migrations.run_pending_migrations(conn)
        # All migrations should be re-applied (idempotently).
        assert migrations.current_version(conn) == migrations.target_version()
        assert applied == [m.version for m in migrations.discover_migrations()]

        # Sentinel data must survive.
        row = conn.execute("SELECT name FROM sources WHERE uri='legacy://x'").fetchone()
        assert row is not None and row["name"] == "x"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Acceptance: fresh DB final state == current schema
# ---------------------------------------------------------------------------

def test_fresh_migrate_matches_init_schema(tmp_path: Path):
    via_runner = tmp_path / "runner.sqlite"
    via_init = tmp_path / "init.sqlite"

    runner_conn = _open_raw(via_runner)
    try:
        migrations.run_pending_migrations(runner_conn)
        runner_snapshot = _schema_snapshot(runner_conn)
        runner_version = migrations.current_version(runner_conn)
    finally:
        runner_conn.close()

    db = MemoirsDB(via_init)
    db.init()
    try:
        init_snapshot = _schema_snapshot(db.conn)
        init_version = db.conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        db.close()

    assert runner_snapshot == init_snapshot
    assert runner_version == init_version == migrations.target_version()


def test_fresh_db_contains_all_expected_tables(tmp_path: Path):
    db = MemoirsDB(tmp_path / "tables.sqlite")
    db.init()
    try:
        rows = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        names = {r[0] for r in rows}
        expected = {
            "sources", "conversations", "messages", "attachments", "import_runs",
            "memory_candidates", "memories", "memory_embeddings",
            "entities", "relationships", "memory_entities", "event_queue",
        }
        assert expected.issubset(names)
    finally:
        db.close()


def test_auto_migrate_false_skips_init(tmp_path: Path):
    db = MemoirsDB(tmp_path / "noauto.sqlite", auto_migrate=False)
    db.init()  # should be a no-op
    try:
        version = db.conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == 0
        rows = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        assert rows == []
    finally:
        db.close()


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------

def _run_cli(*cli_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "memoirs", *cli_args],
        capture_output=True, text=True, check=False,
    )


def test_cli_db_version_on_fresh_db(tmp_path: Path):
    db_path = tmp_path / "cli.sqlite"
    result = _run_cli("--db", str(db_path), "db", "version")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["current"] == 0
    assert payload["target"] >= 1
    assert payload["pending"] == payload["target"]


def test_cli_db_migrate_then_idempotent(tmp_path: Path):
    db_path = tmp_path / "cli2.sqlite"

    first = _run_cli("--db", str(db_path), "db", "migrate")
    assert first.returncode == 0, first.stderr
    payload = json.loads(first.stdout)
    expected = [m.version for m in migrations.discover_migrations()]
    assert payload["applied"] == expected
    assert payload["current"] == migrations.target_version()

    second = _run_cli("--db", str(db_path), "db", "migrate")
    assert second.returncode == 0, second.stderr
    payload = json.loads(second.stdout)
    assert payload["applied"] == []
    assert payload["current"] == migrations.target_version()


def test_migration_004_adds_strength(tmp_path: Path):
    """004 must introduce ``strength`` + ``last_accessed_at`` on memories,
    plus an index for the latter, and backfill ``last_accessed_at`` from a
    pre-existing timestamp.
    """
    db_path = tmp_path / "m4.sqlite"
    conn = _open_raw(db_path)
    try:
        migrations.run_pending_migrations(conn)
        # Columns
        cols = {
            r[1]: dict(name=r[1], type=r[2], notnull=r[3], dflt=r[4])
            for r in conn.execute("PRAGMA table_info(memories)").fetchall()
        }
        assert "strength" in cols
        assert cols["strength"]["type"].upper() == "REAL"
        assert int(cols["strength"]["notnull"]) == 1
        assert "last_accessed_at" in cols
        # Index
        idx_names = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='memories'"
            ).fetchall()
        }
        assert "idx_memories_last_accessed" in idx_names

        # Backfill smoke test: insert a row through the canonical INSERT,
        # confirm strength defaults to 1.0 and the explicit backfill picks
        # last_used_at when present.
        conn.execute(
            """
            INSERT INTO memories (
                id, type, content, content_hash, importance, confidence,
                last_used_at, valid_from, created_at, updated_at
            ) VALUES (
                'mem_m4_seed', 'fact', 'seeded', 'h_m4', 3, 0.5,
                '2025-12-01T00:00:00+00:00', '2025-12-01T00:00:00+00:00',
                '2025-11-01T00:00:00+00:00', '2025-12-01T00:00:00+00:00'
            )
            """
        )
        # Re-run the backfill statement (it's idempotent for rows with NULL
        # last_accessed_at) — emulates what migration 004 would do on a
        # legacy DB after upgrade.
        conn.execute(
            """
            UPDATE memories
            SET last_accessed_at = COALESCE(
                last_used_at, updated_at, valid_from, created_at
            )
            WHERE last_accessed_at IS NULL
            """
        )
        row = conn.execute(
            "SELECT strength, last_accessed_at FROM memories WHERE id='mem_m4_seed'"
        ).fetchone()
        assert row["strength"] == 1.0
        assert row["last_accessed_at"] == "2025-12-01T00:00:00+00:00"
    finally:
        conn.close()


def test_migration_005_tool_call_round_trip(tmp_path: Path):
    """005 must add the four tool_* columns + idx_memories_tool_name, and
    its down() must reverse the change while preserving pre-existing data
    on the `memories` table.
    """
    db_path = tmp_path / "m5.sqlite"
    conn = _open_raw(db_path)
    try:
        # Forward to v5 only — guarantees we exercise the migration boundary.
        migrations.migrate_to(conn, 5)
        assert migrations.current_version(conn) == 5

        cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
        assert {
            "tool_name", "tool_args_json", "tool_result_hash", "tool_status"
        }.issubset(cols)

        idx_names = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='memories'"
            ).fetchall()
        }
        assert "idx_memories_tool_name" in idx_names

        # Seed a regular row so down() must preserve it.
        conn.execute(
            """
            INSERT INTO memories (
                id, type, content, content_hash, importance, confidence,
                valid_from, created_at, updated_at
            ) VALUES (
                'mem_pre_005', 'fact', 'pre-005 row', 'h_pre_005', 3, 0.5,
                '2026-01-01T00:00:00+00:00',
                '2026-01-01T00:00:00+00:00',
                '2026-01-01T00:00:00+00:00'
            )
            """
        )
        # Seed a tool_call row.
        conn.execute(
            """
            INSERT INTO memories (
                id, type, content, content_hash, importance, confidence,
                tool_name, tool_args_json, tool_result_hash, tool_status,
                valid_from, created_at, updated_at
            ) VALUES (
                'mem_tc_005', 'tool_call', 'bash(...) -> ok', 'h_tc_005', 2, 0.9,
                'bash', '{"cmd": "ls"}', 'deadbeefcafebabe', 'success',
                '2026-01-01T00:00:00+00:00',
                '2026-01-01T00:00:00+00:00',
                '2026-01-01T00:00:00+00:00'
            )
            """
        )
        conn.commit()

        # Roll back to v4 — table rebuild must drop the four columns + index.
        migrations.migrate_to(conn, 4)
        assert migrations.current_version(conn) == 4

        cols_after = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
        assert "tool_name" not in cols_after
        assert "tool_args_json" not in cols_after
        assert "tool_result_hash" not in cols_after
        assert "tool_status" not in cols_after

        idx_after = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='memories'"
            ).fetchall()
        }
        assert "idx_memories_tool_name" not in idx_after
        # The 004 index must survive the rebuild.
        assert "idx_memories_last_accessed" in idx_after

        # Pre-existing row must still be there after the rebuild.
        row = conn.execute(
            "SELECT id, type, content FROM memories WHERE id='mem_pre_005'"
        ).fetchone()
        assert row is not None and row["type"] == "fact"

        # Forward again to v5 — must succeed and re-add the columns.
        applied = migrations.migrate_to(conn, 5)
        assert 5 in applied
        cols_again = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
        assert "tool_name" in cols_again
    finally:
        conn.close()


def test_cli_db_rollback(tmp_path: Path):
    db_path = tmp_path / "cli3.sqlite"
    _run_cli("--db", str(db_path), "db", "migrate")

    result = _run_cli("--db", str(db_path), "db", "migrate", "--rollback")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    all_versions = [m.version for m in migrations.discover_migrations()]
    previous = all_versions[-2] if len(all_versions) >= 2 else 0
    assert payload["rolled_back"] == [migrations.target_version()]
    assert payload["current"] == previous


def test_migration_007_sleep_runs_table(tmp_path: Path):
    """007 adds the ``sleep_runs`` audit table for the sleep-time async
    consolidation scheduler (P1-4). Schema must include a started_at
    index so ``ORDER BY started_at DESC`` queries stay cheap, and the
    table must accept the JSON payload shape the scheduler produces.
    """
    db_path = tmp_path / "m7.sqlite"
    conn = _open_raw(db_path)
    try:
        migrations.run_pending_migrations(conn)
        # Schema
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "sleep_runs" in tables
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(sleep_runs)"
        ).fetchall()}
        assert {"id", "started_at", "finished_at", "jobs_json", "error"} <= cols
        idx = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='sleep_runs'"
        ).fetchall()}
        assert "idx_sleep_runs_started" in idx

        # Insert a synthetic row, round-trip through json.
        payload = json.dumps([
            {"name": "consolidate", "status": "ok", "duration_ms": 12.3,
             "result": {"processed": 5, "by_action": {"ADD": 5}}},
            {"name": "dedup", "status": "ok", "duration_ms": 3.1,
             "result": {"merged": 0, "contradictions": 0}},
        ])
        conn.execute(
            "INSERT INTO sleep_runs (started_at, finished_at, jobs_json) "
            "VALUES (?, ?, ?)",
            ("2026-04-27T00:00:00+00:00", "2026-04-27T00:00:05+00:00", payload),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, jobs_json FROM sleep_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        decoded = json.loads(row["jobs_json"])
        assert {j["name"] for j in decoded} == {"consolidate", "dedup"}

        # Round-trip the migration to confirm down() drops cleanly. Other
        # agents added migrations 008+, so we roll back through 7 specifically.
        target = migrations.target_version()
        steps_back = max(1, target - 6)
        migrations.rollback(conn, steps=steps_back)
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "sleep_runs" not in tables
        migrations.run_pending_migrations(conn)
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "sleep_runs" in tables
    finally:
        conn.close()


def test_migration_006_raptor_summary_tables(tmp_path: Path):
    """006 must create summary_nodes + summary_node_members + their indexes,
    survive a round-trip down/up, and accept INSERT/DELETE traffic.
    """
    db_path = tmp_path / "m6.sqlite"
    conn = _open_raw(db_path)
    try:
        migrations.run_pending_migrations(conn)
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "summary_nodes" in tables
        assert "summary_node_members" in tables

        idx_names = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name IN ('summary_nodes','summary_node_members')"
            ).fetchall()
        }
        assert "idx_summary_nodes_parent" in idx_names
        assert "idx_summary_nodes_level" in idx_names
        assert "idx_summary_nodes_scope" in idx_names
        assert "idx_summary_node_members_member" in idx_names

        # Insert a root + a leaf-member edge to confirm both tables accept
        # data per the documented schema.
        conn.execute(
            "INSERT INTO summary_nodes "
            "(id, level, content, child_count, scope_kind) "
            "VALUES ('sum_test_root', 1, 'root summary', 3, 'global')"
        )
        conn.execute(
            "INSERT INTO summary_node_members "
            "(node_id, member_kind, member_id, similarity) "
            "VALUES ('sum_test_root', 'memory', 'mem_x', 0.91)"
        )
        conn.commit()
        seeded = conn.execute(
            "SELECT id, child_count FROM summary_nodes WHERE id='sum_test_root'"
        ).fetchone()
        assert seeded is not None and int(seeded["child_count"]) == 3

        # Roll 006 (and any later) back to its predecessor and confirm tables
        # are dropped, then re-apply forward.
        previous = max(
            v for v in [m.version for m in migrations.discover_migrations()]
            if v < 6
        )
        migrations.migrate_to(conn, previous)
        tables_after = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "summary_nodes" not in tables_after
        assert "summary_node_members" not in tables_after

        applied = migrations.run_pending_migrations(conn)
        assert 6 in applied
        names = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "summary_nodes" in names
        assert "summary_node_members" in names
    finally:
        conn.close()


def test_migration_010_candidate_rejection_round_trip(tmp_path: Path):
    """010 adds the partial index ``idx_memory_candidates_status_reject``
    over ``memory_candidates(status) WHERE status='rejected'``. The
    ``rejection_reason`` column already existed since 001 so the migration
    must NOT drop it on rollback. We exercise both up() (index appears,
    column survives) and down() (index gone, column survives).
    """
    db_path = tmp_path / "m10.sqlite"
    conn = _open_raw(db_path)
    try:
        migrations.run_pending_migrations(conn)
        # Up: the partial index must exist on memory_candidates.
        idx_names = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='memory_candidates'"
            ).fetchall()
        }
        assert "idx_memory_candidates_status_reject" in idx_names

        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(memory_candidates)"
        ).fetchall()}
        assert "rejection_reason" in cols

        # Confirm we can write a rejected candidate row carrying a reason.
        conn.execute(
            "INSERT INTO memory_candidates (id, source_message_ids, type, content, "
            "  importance, confidence, entities, status, rejection_reason, "
            "  raw_json, created_at, updated_at) "
            "VALUES ('cand_010_a', '[]', 'fact', 'noise content', 1, 0.1, '[]', "
            "  'rejected', 'noise: code snippet', '{}', "
            "  '2026-04-27T00:00:00+00:00', '2026-04-27T00:00:00+00:00')"
        )
        conn.commit()
        row = conn.execute(
            "SELECT status, rejection_reason FROM memory_candidates "
            "WHERE id='cand_010_a'"
        ).fetchone()
        assert row["status"] == "rejected"
        assert row["rejection_reason"] == "noise: code snippet"

        # Down: index drops, column survives, seed row survives.
        # Roll back through any migrations stacked on top of 010 (e.g. 011+
        # added later) so this test stays focused on the 010 boundary.
        target = migrations.target_version()
        steps_back = max(1, target - 9)
        rolled = migrations.rollback(conn, steps=steps_back)
        assert -10 in rolled
        idx_after = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='memory_candidates'"
            ).fetchall()
        }
        assert "idx_memory_candidates_status_reject" not in idx_after
        # Column must still be present (it predates this migration).
        cols_after = {r[1] for r in conn.execute(
            "PRAGMA table_info(memory_candidates)"
        ).fetchall()}
        assert "rejection_reason" in cols_after
        survived = conn.execute(
            "SELECT id FROM memory_candidates WHERE id='cand_010_a'"
        ).fetchone()
        assert survived is not None

        # Re-apply forward: idempotent and the index reappears.
        applied = migrations.run_pending_migrations(conn)
        assert 10 in applied
        idx_again = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='memory_candidates'"
            ).fetchall()
        }
        assert "idx_memory_candidates_status_reject" in idx_again
    finally:
        conn.close()


def test_migration_011_thread_summaries_round_trip(tmp_path: Path):
    """011 introduces the ``thread_summaries`` table for the auto-resume
    thread feature (P-resume). The schema must include the documented
    columns + the two indexes (per-conversation lookup + most-recent),
    plus the ``UNIQUE(conversation_id, generated_at)`` constraint.
    """
    db_path = tmp_path / "m11.sqlite"
    conn = _open_raw(db_path)
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
        idx_names = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='thread_summaries'"
            ).fetchall()
        }
        assert "idx_thread_summaries_conv" in idx_names
        assert "idx_thread_summaries_recent" in idx_names

        # UNIQUE(conversation_id, generated_at) blocks duplicate inserts.
        conn.execute(
            "INSERT INTO thread_summaries "
            "(conversation_id, summary, generated_at, message_count_at_summary, "
            " last_message_ts, pending_actions_json, salient_entity_ids_json) "
            "VALUES ('cv_a', 'first', '2026-04-28T00:00:00+00:00', 5, "
            " '2026-04-27T23:59:00+00:00', '[]', '[]')"
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO thread_summaries "
                "(conversation_id, summary, generated_at, message_count_at_summary) "
                "VALUES ('cv_a', 'dup', '2026-04-28T00:00:00+00:00', 5)"
            )
        conn.rollback()

        # Round-trip: rollback to v10 drops the table, run_pending re-creates.
        target = migrations.target_version()
        steps_back = max(1, target - 10)
        rolled = migrations.rollback(conn, steps=steps_back)
        assert -11 in rolled
        tables_after = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "thread_summaries" not in tables_after
        applied = migrations.run_pending_migrations(conn)
        assert 11 in applied
    finally:
        conn.close()


def test_migration_009_memory_conflicts_round_trip(tmp_path: Path):
    """009 introduces ``memory_conflicts`` for the conflict resolution UI
    (P5-2). The schema must include the documented columns + the
    UNIQUE(memory_a_id, memory_b_id) constraint, plus an index on status.
    """
    db_path = tmp_path / "m9.sqlite"
    conn = _open_raw(db_path)
    try:
        migrations.run_pending_migrations(conn)
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "memory_conflicts" in tables
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(memory_conflicts)"
        ).fetchall()}
        assert {
            "id", "memory_a_id", "memory_b_id", "similarity", "detected_at",
            "detector", "reason", "status", "resolution_notes", "resolved_at",
        } <= cols
        idx_names = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='memory_conflicts'"
            ).fetchall()
        }
        assert "idx_conflicts_status" in idx_names

        # UNIQUE(a, b) blocks duplicate inserts with the same pair.
        conn.execute(
            "INSERT INTO memory_conflicts (memory_a_id, memory_b_id, similarity, "
            "  detected_at, detector, reason) "
            "VALUES ('mem_a', 'mem_b', 0.9, '2026-04-27T00:00:00+00:00', 'gemma', 'r')"
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO memory_conflicts (memory_a_id, memory_b_id, "
                "  detected_at) VALUES ('mem_a', 'mem_b', '2026-04-27T00:00:00+00:00')"
            )
        conn.rollback()

        # Round-trip: rollback to v8 drops the table, run_pending re-creates
        # it. We step back through any later migrations (010+) so the test
        # stays focused on the 009 boundary regardless of how many migrations
        # have been added on top.
        target = migrations.target_version()
        steps_back = max(1, target - 8)
        rolled = migrations.rollback(conn, steps=steps_back)
        assert -9 in rolled
        tables_after = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "memory_conflicts" not in tables_after
        applied = migrations.run_pending_migrations(conn)
        assert 9 in applied
    finally:
        conn.close()
