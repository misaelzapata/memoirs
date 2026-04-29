"""Tests for the conflict resolution surface (P5-2).

Covers:
- migration 009 round-trip + UNIQUE pair constraint
- ``record_conflict`` persistence + idempotent refresh
- ``list_conflicts`` status filtering + JOIN
- ``resolve_conflict`` for every action (keep_a, keep_b, keep_both, merge,
  dismiss) — including merge with a stub gemma_summarize so we don't pull
  llama-cpp into the test runner
- the FastAPI UI endpoints (list / detail / resolve)
- end-to-end CLI ``conflicts resolve``
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from memoirs.api.server import _build_app
from memoirs.core.ids import content_hash, utc_now
from memoirs.db import MemoirsDB
from memoirs.engine import conflicts as conflicts_mod
from memoirs.engine.conflicts import (
    list_conflicts,
    record_conflict,
    resolve_conflict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _seed_memory(db: MemoirsDB, mid: str, mtype: str, content: str) -> None:
    now = utc_now()
    db.conn.execute(
        """
        INSERT INTO memories (
            id, type, content, content_hash, importance, confidence,
            score, usage_count, user_signal, valid_from, metadata_json,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, 3, 0.7, 0.5, 0, 0, ?, '{}', ?, ?)
        """,
        (mid, mtype, content, content_hash(content), now, now, now),
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Bare DB with three memorias and one persisted conflict (mem_a vs mem_b)."""
    p = tmp_path / "memoirs.sqlite"
    db = MemoirsDB(p)
    db.init()
    with db.conn:
        _seed_memory(db, "mem_a", "preference", "User prefers Python.")
        _seed_memory(db, "mem_b", "preference", "User prefers Go.")
        _seed_memory(db, "mem_c", "fact", "memoirs uses sqlite-vec.")
    db.close()
    return p


@pytest.fixture
def db(db_path: Path) -> MemoirsDB:
    handle = MemoirsDB(db_path)
    handle.init()
    yield handle
    handle.close()


@pytest.fixture
def seeded_with_conflict(db: MemoirsDB) -> int:
    """Returns the conflict id for mem_a / mem_b."""
    return record_conflict(
        db,
        memory_a_id="mem_a",
        memory_b_id="mem_b",
        similarity=0.91,
        detector="gemma",
        reason="contradictory preferences",
    )


@pytest.fixture
def client(db_path: Path) -> TestClient:
    return TestClient(_build_app(db_path))


# ---------------------------------------------------------------------------
# Engine layer
# ---------------------------------------------------------------------------


def test_record_conflict_persists_and_normalises(db: MemoirsDB):
    """Pair (a, b) and (b, a) collapse to a single row thanks to the
    canonical ordering inside ``record_conflict``."""
    cid_1 = record_conflict(
        db, memory_a_id="mem_b", memory_b_id="mem_a",
        similarity=0.9, detector="gemma", reason="r1",
    )
    cid_2 = record_conflict(
        db, memory_a_id="mem_a", memory_b_id="mem_b",
        similarity=0.92, detector="gemma", reason="r2",
    )
    assert cid_1 == cid_2  # ON CONFLICT DO UPDATE — same pair, same row.
    rows = db.conn.execute(
        "SELECT memory_a_id, memory_b_id, similarity, reason, status "
        "FROM memory_conflicts"
    ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert (row["memory_a_id"], row["memory_b_id"]) == ("mem_a", "mem_b")
    # Refresh kept status pending and updated similarity / reason.
    assert row["status"] == "pending"
    assert abs(row["similarity"] - 0.92) < 1e-9
    assert row["reason"] == "r2"


def test_record_conflict_rejects_self_pair(db: MemoirsDB):
    with pytest.raises(ValueError):
        record_conflict(
            db, memory_a_id="mem_a", memory_b_id="mem_a", similarity=1.0,
            detector="gemma", reason="x",
        )


def test_list_conflicts_filters_by_status(db: MemoirsDB, seeded_with_conflict: int):
    pending = list_conflicts(db, status="pending")
    assert len(pending) == 1
    assert pending[0]["id"] == seeded_with_conflict
    # Joined memory contents must round-trip.
    assert pending[0]["a_content"] == "User prefers Python."
    assert pending[0]["b_content"] == "User prefers Go."

    resolved = list_conflicts(db, status="resolved_keep_a")
    assert resolved == []

    no_filter = list_conflicts(db, status=None)
    assert len(no_filter) == 1


def test_resolve_keep_a_archives_b(db: MemoirsDB, seeded_with_conflict: int):
    report = resolve_conflict(db, seeded_with_conflict, action="keep_a", notes="A wins")
    assert report["status"] == "resolved_keep_a"
    assert report["archived"] == ["mem_b"]
    assert report["new_memory_id"] is None

    a = db.conn.execute(
        "SELECT archived_at FROM memories WHERE id = 'mem_a'"
    ).fetchone()
    b = db.conn.execute(
        "SELECT archived_at, archive_reason FROM memories WHERE id = 'mem_b'"
    ).fetchone()
    assert a["archived_at"] is None
    assert b["archived_at"] is not None
    assert "keep_a" in (b["archive_reason"] or "")

    row = db.conn.execute(
        "SELECT status, resolution_notes FROM memory_conflicts WHERE id = ?",
        (seeded_with_conflict,),
    ).fetchone()
    assert row["status"] == "resolved_keep_a"
    assert "A wins" in (row["resolution_notes"] or "")


def test_resolve_keep_b_archives_a(db: MemoirsDB, seeded_with_conflict: int):
    resolve_conflict(db, seeded_with_conflict, action="keep_b")
    a = db.conn.execute("SELECT archived_at FROM memories WHERE id='mem_a'").fetchone()
    b = db.conn.execute("SELECT archived_at FROM memories WHERE id='mem_b'").fetchone()
    assert a["archived_at"] is not None
    assert b["archived_at"] is None


def test_resolve_keep_both_does_not_archive(
    db: MemoirsDB, seeded_with_conflict: int,
):
    resolve_conflict(db, seeded_with_conflict, action="keep_both")
    rows = db.conn.execute(
        "SELECT id, archived_at FROM memories WHERE id IN ('mem_a','mem_b')"
    ).fetchall()
    assert all(r["archived_at"] is None for r in rows)
    status = db.conn.execute(
        "SELECT status FROM memory_conflicts WHERE id = ?", (seeded_with_conflict,)
    ).fetchone()["status"]
    assert status == "resolved_keep_both"


def test_resolve_dismiss_does_not_archive(
    db: MemoirsDB, seeded_with_conflict: int,
):
    resolve_conflict(db, seeded_with_conflict, action="dismiss", notes="false alarm")
    rows = db.conn.execute(
        "SELECT id, archived_at FROM memories WHERE id IN ('mem_a','mem_b')"
    ).fetchall()
    assert all(r["archived_at"] is None for r in rows)
    row = db.conn.execute(
        "SELECT status, resolution_notes FROM memory_conflicts WHERE id = ?",
        (seeded_with_conflict,),
    ).fetchone()
    assert row["status"] == "dismissed"
    assert "false alarm" in (row["resolution_notes"] or "")


def test_resolve_merge_creates_new_memoria_and_archives_both(
    db: MemoirsDB, seeded_with_conflict: int, monkeypatch,
):
    """Merge: both memorias are archived and a fresh memoria is created. We
    stub Gemma so the test is hermetic and doesn't pull llama-cpp."""
    # Force the deterministic fallback path by reporting Gemma as unavailable.
    from memoirs.engine import curator as curator_mod
    monkeypatch.setattr(curator_mod, "_have_curator", lambda: False)

    report = resolve_conflict(db, seeded_with_conflict, action="merge", notes="combined")
    assert report["status"] == "resolved_merge"
    assert report["new_memory_id"] is not None
    assert sorted(report["archived"]) == ["mem_a", "mem_b"]

    new = db.conn.execute(
        "SELECT id, type, content, archived_at FROM memories WHERE id = ?",
        (report["new_memory_id"],),
    ).fetchone()
    assert new is not None
    assert new["archived_at"] is None
    assert "Python" in new["content"] and "Go" in new["content"]
    # Both originals are archived.
    rows = db.conn.execute(
        "SELECT id, archived_at FROM memories WHERE id IN ('mem_a','mem_b')"
    ).fetchall()
    assert all(r["archived_at"] is not None for r in rows)
    # resolution_notes carries the merged memory id pointer for the UI.
    notes = db.conn.execute(
        "SELECT resolution_notes FROM memory_conflicts WHERE id = ?",
        (seeded_with_conflict,),
    ).fetchone()["resolution_notes"]
    assert report["new_memory_id"] in (notes or "")


def test_resolve_unknown_action_raises(db: MemoirsDB, seeded_with_conflict: int):
    with pytest.raises(ValueError):
        resolve_conflict(db, seeded_with_conflict, action="vaporise")


def test_resolve_already_resolved_raises(db: MemoirsDB, seeded_with_conflict: int):
    resolve_conflict(db, seeded_with_conflict, action="dismiss")
    with pytest.raises(ValueError):
        resolve_conflict(db, seeded_with_conflict, action="keep_a")


# ---------------------------------------------------------------------------
# UI layer
# ---------------------------------------------------------------------------


def test_ui_conflicts_list_returns_html_with_table(
    client: TestClient, db_path: Path,
):
    db = MemoirsDB(db_path)
    db.init()
    record_conflict(
        db, memory_a_id="mem_a", memory_b_id="mem_b",
        similarity=0.91, detector="gemma", reason="ui-list",
    )
    db.close()

    r = client.get("/ui/conflicts")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    body = r.text
    assert "Conflicts" in body
    assert "mem_a" in body or "User prefers" in body
    assert "ui-list" in body or "pending" in body  # filter pill / detector reason


def test_ui_conflict_detail_renders_diff(
    client: TestClient, db_path: Path,
):
    db = MemoirsDB(db_path)
    db.init()
    cid = record_conflict(
        db, memory_a_id="mem_a", memory_b_id="mem_b",
        similarity=0.91, detector="gemma", reason="ui-diff",
    )
    db.close()

    r = client.get(f"/ui/conflicts/{cid}")
    assert r.status_code == 200
    body = r.text
    # difflib.HtmlDiff emits a <table class="diff" …> block.
    assert "diff" in body.lower()
    assert "Memory A" in body
    assert "Memory B" in body


def test_ui_conflict_detail_404_for_missing(client: TestClient):
    r = client.get("/ui/conflicts/99999")
    assert r.status_code == 404


def test_ui_conflict_resolve_archives_memory_b(
    client: TestClient, db_path: Path,
):
    db = MemoirsDB(db_path)
    db.init()
    cid = record_conflict(
        db, memory_a_id="mem_a", memory_b_id="mem_b",
        similarity=0.91, detector="gemma", reason="ui-resolve",
    )
    db.close()

    r = client.post(
        f"/ui/conflicts/{cid}/resolve",
        data={"action": "keep_a", "notes": "via http"},
    )
    assert r.status_code == 200
    assert "resolved_keep_a" in r.text

    db = MemoirsDB(db_path)
    db.init()
    try:
        b = db.conn.execute(
            "SELECT archived_at FROM memories WHERE id = 'mem_b'"
        ).fetchone()
        assert b["archived_at"] is not None
    finally:
        db.close()


def test_ui_conflict_resolve_rejects_invalid_action(
    client: TestClient, db_path: Path,
):
    db = MemoirsDB(db_path)
    db.init()
    cid = record_conflict(
        db, memory_a_id="mem_a", memory_b_id="mem_b",
        similarity=0.5, detector="gemma", reason="x",
    )
    db.close()
    r = client.post(
        f"/ui/conflicts/{cid}/resolve",
        data={"action": "deletes_everything"},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# CLI layer
# ---------------------------------------------------------------------------


def _run_cli(args: list[str], db_path: Path):
    cmd = [sys.executable, "-m", "memoirs", "--db", str(db_path), *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_cli_conflicts_resolve_end_to_end(db_path: Path):
    db = MemoirsDB(db_path)
    db.init()
    cid = record_conflict(
        db, memory_a_id="mem_a", memory_b_id="mem_b",
        similarity=0.91, detector="gemma", reason="cli-test",
    )
    db.close()

    # list returns the pending row
    out = _run_cli(["conflicts", "list", "--json"], db_path)
    assert out.returncode == 0, out.stderr
    listed = json.loads(out.stdout)
    assert any(int(r["id"]) == cid for r in listed)

    # show the row
    out = _run_cli(["conflicts", "show", str(cid), "--json"], db_path)
    assert out.returncode == 0, out.stderr
    shown = json.loads(out.stdout)
    assert int(shown["id"]) == cid
    assert shown["status"] == "pending"

    # resolve via CLI: keep_a archives mem_b
    out = _run_cli(
        ["conflicts", "resolve", str(cid),
         "--action", "keep_a", "--notes", "cli wins", "--json"],
        db_path,
    )
    assert out.returncode == 0, out.stderr
    report = json.loads(out.stdout)
    assert report["status"] == "resolved_keep_a"
    assert report["archived"] == ["mem_b"]

    db = MemoirsDB(db_path)
    db.init()
    try:
        b = db.conn.execute(
            "SELECT archived_at FROM memories WHERE id = 'mem_b'"
        ).fetchone()
        assert b["archived_at"] is not None
    finally:
        db.close()


def test_cli_conflicts_list_status_alias_resolved(db_path: Path):
    db = MemoirsDB(db_path)
    db.init()
    cid = record_conflict(
        db, memory_a_id="mem_a", memory_b_id="mem_b",
        similarity=0.5, detector="gemma", reason="alias",
    )
    resolve_conflict(db, cid, action="dismiss")
    cid2 = record_conflict(
        db, memory_a_id="mem_a", memory_b_id="mem_c",
        similarity=0.6, detector="gemma", reason="alias2",
    )
    resolve_conflict(db, cid2, action="keep_a")
    db.close()

    out = _run_cli(
        ["conflicts", "list", "--status", "resolved", "--json"], db_path,
    )
    assert out.returncode == 0, out.stderr
    listed = json.loads(out.stdout)
    statuses = {r["status"] for r in listed}
    # 'resolved' alias matches resolved_* but not 'dismissed'.
    assert statuses == {"resolved_keep_a"}


def test_record_conflict_keeps_user_resolution_after_redetection(
    db: MemoirsDB,
):
    """Sleep cycles re-detect the same pair; the persisted resolution must
    survive without being clobbered back to pending."""
    cid = record_conflict(
        db, memory_a_id="mem_a", memory_b_id="mem_b",
        similarity=0.9, detector="gemma", reason="first",
    )
    resolve_conflict(db, cid, action="keep_both", notes="ok")
    # Re-record (simulating a later sleep run flagging the same pair).
    record_conflict(
        db, memory_a_id="mem_b", memory_b_id="mem_a",
        similarity=0.95, detector="gemma", reason="second",
    )
    row = db.conn.execute(
        "SELECT status, similarity, reason FROM memory_conflicts WHERE id = ?",
        (cid,),
    ).fetchone()
    assert row["status"] == "resolved_keep_both"
    assert abs(row["similarity"] - 0.95) < 1e-9
    assert row["reason"] == "second"
