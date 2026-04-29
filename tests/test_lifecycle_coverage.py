"""Targeted coverage tests for memoirs/engine/lifecycle.py.

Focus areas (uncovered branches in the baseline):
  * calculate_decay (public alias)
  * promote_all / demote_all walkers
  * auto_merge_near_duplicates: same-type merge, different-type contradiction,
    embeddings_unavailable error path, dry_run guard
  * _flag_contradiction metadata mutation
  * Error branches in _within_days / _older_than_days (bad ISO inputs)
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone, timedelta

import pytest

from memoirs.config import EMBEDDING_DIM
from memoirs.engine import embeddings as emb
from memoirs.engine import lifecycle as lc


# ---------------------------------------------------------------------------
# Helpers (kept distinct from tests/test_lifecycle.py)
# ---------------------------------------------------------------------------


def _unit_vec(angle_rad: float) -> list[float]:
    v = [0.0] * EMBEDDING_DIM
    v[0] = math.cos(angle_rad)
    v[1] = math.sin(angle_rad)
    return v


def _seed(db, *, memory_id: str, mem_type: str = "fact", content: str = "x",
          score: float = 0.5, importance: int = 3, usage_count: int = 0,
          user_signal: float = 0.0, last_used_at: str | None = None,
          created_at: str | None = None, angle_rad: float | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    created_at = created_at or now
    db.conn.execute(
        "INSERT INTO memories (id, type, content, content_hash, importance, "
        "confidence, score, usage_count, user_signal, valid_from, metadata_json, "
        "created_at, updated_at, last_used_at) "
        "VALUES (?, ?, ?, 'h_'||?, ?, 0.5, ?, ?, ?, ?, '{}', ?, ?, ?)",
        (memory_id, mem_type, content, memory_id, importance, score,
         usage_count, user_signal, now, created_at, now, last_used_at),
    )
    if angle_rad is not None:
        emb._require_vec(db)
        blob = emb._pack(_unit_vec(angle_rad))
        db.conn.execute(
            "INSERT INTO memory_embeddings (memory_id, dim, embedding, model, created_at) "
            "VALUES (?, ?, ?, 'test', ?)",
            (memory_id, EMBEDDING_DIM, blob, now),
        )
        db.conn.execute("DELETE FROM vec_memories WHERE memory_id = ?", (memory_id,))
        db.conn.execute(
            "INSERT INTO vec_memories(memory_id, embedding) VALUES (?, ?)",
            (memory_id, blob),
        )
    db.conn.commit()


# ---------------------------------------------------------------------------
# calculate_decay (public alias) — covered the import-and-delegate branch
# ---------------------------------------------------------------------------


def test_calculate_decay_recent_memory_returns_high_score():
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    score = lc.calculate_decay({"last_used_at": now, "created_at": now})
    assert 0.99 <= score <= 1.0


def test_calculate_decay_old_memory_returns_low_score():
    old = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat(timespec="seconds")
    score = lc.calculate_decay({"created_at": old})
    assert score < 0.05


# ---------------------------------------------------------------------------
# Helper edge cases — bad ISO inputs hit the ValueError branch
# ---------------------------------------------------------------------------


def test_within_days_handles_bad_input():
    assert lc._within_days(None, 7) is False
    assert lc._within_days("not-an-iso", 7) is False


def test_older_than_days_handles_bad_input():
    assert lc._older_than_days(None, 60) is False
    assert lc._older_than_days("not-an-iso", 60) is False


# ---------------------------------------------------------------------------
# promote_all / demote_all walkers
# ---------------------------------------------------------------------------


def test_promote_all_promotes_eligible(tmp_db):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _seed(tmp_db, memory_id="p1", importance=3, usage_count=10, last_used_at=now)
    _seed(tmp_db, memory_id="p2", importance=2, usage_count=8, last_used_at=now)
    _seed(tmp_db, memory_id="p3", importance=1, usage_count=2, last_used_at=now)  # ineligible
    promoted = lc.promote_all(tmp_db)
    assert promoted == 2


def test_demote_all_demotes_eligible(tmp_db):
    old = (datetime.now(timezone.utc) - timedelta(days=70)).isoformat(timespec="seconds")
    recent = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _seed(tmp_db, memory_id="d1", importance=3, usage_count=0, created_at=old)
    _seed(tmp_db, memory_id="d2", importance=2, usage_count=0, created_at=old)
    _seed(tmp_db, memory_id="d3", importance=3, usage_count=5, created_at=old)  # used → skip
    _seed(tmp_db, memory_id="d4", importance=3, usage_count=0, created_at=recent)  # too young
    demoted = lc.demote_all(tmp_db)
    assert demoted == 2


# ---------------------------------------------------------------------------
# auto_merge_near_duplicates — happy + dry-run + contradiction + error paths
# ---------------------------------------------------------------------------


def test_auto_merge_handles_embeddings_unavailable(tmp_db, monkeypatch):
    """When embeddings can't run, return a clean error dict, no crash."""
    _seed(tmp_db, memory_id="m1", content="alpha")

    def boom(*args, **kwargs):
        raise emb.EmbeddingsUnavailable("simulated")

    monkeypatch.setattr(emb, "search_similar_memories", boom)
    out = lc.auto_merge_near_duplicates(tmp_db)
    assert out["merged"] == 0
    assert out["error"] == "embeddings_unavailable"


def test_auto_merge_dry_run_does_not_archive(tmp_db, monkeypatch):
    """With dry_run=True, near-dup detection still runs but no DB writes happen."""
    _seed(tmp_db, memory_id="m_keep", mem_type="fact", content="alpha", score=0.9, angle_rad=0.0)
    _seed(tmp_db, memory_id="m_drop", mem_type="fact", content="alpha v2", score=0.4, angle_rad=0.001)

    def fake_search(db, content, top_k=3):
        rows = db.conn.execute(
            "SELECT id, type, content, score FROM memories WHERE archived_at IS NULL"
        ).fetchall()
        return [
            {"id": r["id"], "type": r["type"], "content": r["content"],
             "score": r["score"], "similarity": 0.99}
            for r in rows
        ]

    monkeypatch.setattr(emb, "search_similar_memories", fake_search)
    out = lc.auto_merge_near_duplicates(tmp_db, dry_run=True)
    assert out["merged"] == 1
    assert out["dry_run"] is True
    # No row was actually archived.
    archived = tmp_db.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE archived_at IS NOT NULL"
    ).fetchone()[0]
    assert archived == 0


def test_auto_merge_archives_on_real_run(tmp_db, monkeypatch):
    """Same-type near-dup → loser archived, winner keeps usage_count + entity links."""
    _seed(tmp_db, memory_id="m_keep", mem_type="fact", content="alpha", score=0.9, angle_rad=0.0)
    _seed(tmp_db, memory_id="m_drop", mem_type="fact", content="alpha v2", score=0.4, angle_rad=0.001)
    # Give m_drop some usage so the merge transfers it.
    tmp_db.conn.execute("UPDATE memories SET usage_count = 7 WHERE id = 'm_drop'")
    tmp_db.conn.commit()

    def fake_search(db, content, top_k=3):
        rows = db.conn.execute(
            "SELECT id, type, content, score FROM memories WHERE archived_at IS NULL"
        ).fetchall()
        return [
            {"id": r["id"], "type": r["type"], "content": r["content"],
             "score": r["score"], "similarity": 0.99}
            for r in rows
        ]

    monkeypatch.setattr(emb, "search_similar_memories", fake_search)
    out = lc.auto_merge_near_duplicates(tmp_db)
    assert out["merged"] >= 1
    archived_row = tmp_db.conn.execute(
        "SELECT id, archived_at, archive_reason, superseded_by FROM memories WHERE id='m_drop'"
    ).fetchone()
    assert archived_row["archived_at"] is not None
    assert archived_row["superseded_by"] == "m_keep"
    # Winner inherited usage.
    keeper = tmp_db.conn.execute(
        "SELECT usage_count FROM memories WHERE id = 'm_keep'"
    ).fetchone()
    assert keeper["usage_count"] >= 7


def test_auto_merge_flags_contradiction_on_different_types(tmp_db, monkeypatch):
    """High similarity + different type → flagged in metadata, NOT merged."""
    _seed(tmp_db, memory_id="a1", mem_type="fact", content="alpha", score=0.9, angle_rad=0.0)
    _seed(tmp_db, memory_id="a2", mem_type="preference", content="alpha", score=0.4, angle_rad=0.001)

    def fake_search(db, content, top_k=3):
        rows = db.conn.execute(
            "SELECT id, type, content, score FROM memories WHERE archived_at IS NULL"
        ).fetchall()
        return [
            {"id": r["id"], "type": r["type"], "content": r["content"],
             "score": r["score"], "similarity": 0.99}
            for r in rows
        ]

    monkeypatch.setattr(emb, "search_similar_memories", fake_search)
    out = lc.auto_merge_near_duplicates(tmp_db)
    assert out["contradictions"] >= 1
    # Both memories still active (no archive).
    active_count = tmp_db.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE archived_at IS NULL"
    ).fetchone()[0]
    assert active_count == 2
    # Metadata json now has contradiction_with key on at least one.
    a1_meta = tmp_db.conn.execute(
        "SELECT metadata_json FROM memories WHERE id = 'a1'"
    ).fetchone()[0]
    assert "contradiction_with" in a1_meta


def test_auto_merge_skips_when_below_threshold(tmp_db, monkeypatch):
    """If top similarity < threshold, no action taken."""
    _seed(tmp_db, memory_id="b1", mem_type="fact", content="alpha", score=0.5, angle_rad=0.0)
    _seed(tmp_db, memory_id="b2", mem_type="fact", content="bravo", score=0.5, angle_rad=0.5)

    def fake_search(db, content, top_k=3):
        rows = db.conn.execute(
            "SELECT id, type, content, score FROM memories WHERE archived_at IS NULL"
        ).fetchall()
        return [
            {"id": r["id"], "type": r["type"], "content": r["content"],
             "score": r["score"], "similarity": 0.5}  # below threshold
            for r in rows
        ]

    monkeypatch.setattr(emb, "search_similar_memories", fake_search)
    out = lc.auto_merge_near_duplicates(tmp_db, threshold=0.92)
    assert out["merged"] == 0
    assert out["contradictions"] == 0


def test_auto_merge_respects_limit(tmp_db, monkeypatch):
    """`limit` caps the number of seed memories scanned."""
    for i in range(5):
        _seed(tmp_db, memory_id=f"l{i}", mem_type="fact", content=f"row {i}",
              score=0.5, angle_rad=i * 0.01)
    monkeypatch.setattr(emb, "search_similar_memories", lambda *a, **k: [])
    out = lc.auto_merge_near_duplicates(tmp_db, limit=2)
    assert out["scanned"] <= 2
