"""Tests for engine/lifecycle_decisions.py — EXPIRE / ARCHIVE generation
in the curator (P1-10 GAP fill).

Coverage:
  * should_expire heuristic vs Gemma routes (5 cases)
  * should_archive predicates (3 cases)
  * enrich_decision cascading: ADD + obsolete neighbor → secondary EXPIRE
  * enrich_decision MERGE → ARCHIVE when target stale
  * Gemma mock with contradictory=True triggers EXPIRE
  * apply_decision EXPIRE / ARCHIVE branches mutate the row correctly
  * End-to-end: ingest contradictory candidate → neighbor gains valid_to
  * End-to-end: stale neighbor archived during consolidation
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from memoirs.db import content_hash, stable_id, utc_now
from memoirs.engine import curator as curator_mod
from memoirs.engine import gemma as gemma_mod  # noqa: F401 — kept for legacy patch points
from memoirs.engine import lifecycle_decisions as lcd
from memoirs.engine import memory_engine
from memoirs.engine.curator import Candidate
from memoirs.engine.memory_engine import (
    Decision,
    apply_decision,
    decide_memory_action,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _seed(
    db,
    *,
    memory_id: str,
    mem_type: str = "fact",
    content: str = "old fact",
    score: float = 0.5,
    usage_count: int = 1,
    importance: int = 3,
    confidence: float = 0.5,
    age_days: float = 1.0,
    archived: bool = False,
) -> str:
    """Insert a memory N days old. Returns the inserted id."""
    now = datetime.now(timezone.utc)
    created = _iso(now - timedelta(days=age_days))
    archived_at = _iso(now) if archived else None
    h = "h_" + memory_id
    db.conn.execute(
        """
        INSERT INTO memories (
            id, type, content, content_hash, importance, confidence,
            score, usage_count, user_signal, valid_from, metadata_json,
            archived_at, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, '{}', ?, ?, ?)
        """,
        (
            memory_id, mem_type, content, h, importance, confidence,
            score, usage_count, created, archived_at, created, created,
        ),
    )
    db.conn.commit()
    return memory_id


def _row(db, mid: str) -> dict | None:
    r = db.conn.execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone()
    return dict(r) if r else None


# ---------------------------------------------------------------------------
# should_expire — heuristic and Gemma routes
# ---------------------------------------------------------------------------


def test_should_expire_heuristic_fact_old_similar_true():
    """fact + sim>0.8 + age>30d + use_gemma=False → True."""
    cand = Candidate(type="fact", content="The capital of France is Paris.")
    neighbor = {
        "id": "n1",
        "type": "fact",
        "content": "Paris is the capital of France.",
        "similarity": 0.92,
        "created_at": _iso(datetime.now(timezone.utc) - timedelta(days=60)),
    }
    expire, reason = lcd.should_expire(cand, neighbor, use_gemma=False)
    assert expire is True
    assert "sim" in reason.lower() and "fact" in reason


def test_should_expire_heuristic_preference_blocks_without_gemma():
    """type=preference is subjective; heuristic refuses to auto-expire."""
    cand = Candidate(type="preference", content="I like dark mode.")
    neighbor = {
        "id": "n2",
        "type": "preference",
        "content": "User prefers light mode.",
        "similarity": 0.95,
        "created_at": _iso(datetime.now(timezone.utc) - timedelta(days=60)),
    }
    expire, reason = lcd.should_expire(cand, neighbor, use_gemma=False)
    assert expire is False
    assert "preference" in reason


def test_should_expire_heuristic_recent_neighbor_blocks():
    """Even if similar, a young (<30d) neighbor isn't auto-expired."""
    cand = Candidate(type="fact", content="X is true.")
    neighbor = {
        "id": "n3",
        "type": "fact",
        "content": "X is true.",
        "similarity": 0.99,
        "created_at": _iso(datetime.now(timezone.utc) - timedelta(days=10)),
    }
    expire, reason = lcd.should_expire(cand, neighbor, use_gemma=False)
    assert expire is False
    assert "age" in reason


def test_should_expire_low_similarity_blocks():
    cand = Candidate(type="fact", content="A.")
    neighbor = {
        "id": "n4",
        "type": "fact",
        "content": "B.",
        "similarity": 0.5,
        "created_at": _iso(datetime.now(timezone.utc) - timedelta(days=120)),
    }
    expire, reason = lcd.should_expire(cand, neighbor, use_gemma=False)
    assert expire is False
    assert "similarity" in reason


def test_should_expire_gemma_contradictory_winner_a(monkeypatch):
    """Gemma reports contradictory + winner=A → EXPIRE neighbor."""
    monkeypatch.delenv("MEMOIRS_CURATOR_ENABLED", raising=False)
    monkeypatch.delenv("MEMOIRS_GEMMA_CURATOR", raising=False)

    def fake_resolve(a_content, b_content):
        return {
            "contradictory": True,
            "winner": "A",
            "reason": "candidate is fresher",
            "action": "MARK_CONFLICT",
        }

    monkeypatch.setattr(curator_mod, "_have_curator", lambda: True)
    monkeypatch.setattr(curator_mod, "curator_resolve_conflict", fake_resolve)

    cand = Candidate(type="preference", content="I like dark mode.")
    neighbor = {
        "id": "n5",
        "type": "preference",
        "content": "User prefers light mode.",
        "similarity": 0.4,  # heuristic would say no, but Gemma overrides
        "created_at": _iso(datetime.now(timezone.utc) - timedelta(days=2)),
    }
    expire, reason = lcd.should_expire(cand, neighbor)
    assert expire is True
    assert "gemma" in reason.lower()


def test_should_expire_gemma_winner_b_keeps_neighbor(monkeypatch):
    monkeypatch.delenv("MEMOIRS_CURATOR_ENABLED", raising=False)
    monkeypatch.delenv("MEMOIRS_GEMMA_CURATOR", raising=False)
    """Gemma says contradictory but neighbor wins → don't expire."""

    def fake_resolve(*a, **k):
        return {"contradictory": True, "winner": "B", "reason": "keep old", "action": "MARK_CONFLICT"}

    monkeypatch.setattr(curator_mod, "_have_curator", lambda: True)
    monkeypatch.setattr(curator_mod, "curator_resolve_conflict", fake_resolve)

    cand = Candidate(type="fact", content="x")
    neighbor = {
        "id": "n6", "type": "fact", "content": "y",
        "similarity": 0.9,
        "created_at": _iso(datetime.now(timezone.utc) - timedelta(days=120)),
    }
    expire, _ = lcd.should_expire(cand, neighbor)
    assert expire is False


# ---------------------------------------------------------------------------
# should_archive
# ---------------------------------------------------------------------------


def test_should_archive_unused_old_true():
    neighbor = {
        "id": "a1",
        "usage_count": 0,
        "score": 0.4,
        "created_at": _iso(datetime.now(timezone.utc) - timedelta(days=100)),
    }
    archive, reason = lcd.should_archive(neighbor)
    assert archive is True
    assert "unused" in reason


def test_should_archive_unused_recent_false():
    neighbor = {
        "id": "a2",
        "usage_count": 0,
        "score": 0.4,
        "created_at": _iso(datetime.now(timezone.utc) - timedelta(days=10)),
    }
    archive, _ = lcd.should_archive(neighbor)
    assert archive is False


def test_should_archive_used_does_not_archive():
    """usage_count=5 + age 100d → False (no low-score predicate either)."""
    neighbor = {
        "id": "a3",
        "usage_count": 5,
        "score": 0.5,
        "created_at": _iso(datetime.now(timezone.utc) - timedelta(days=100)),
    }
    archive, _ = lcd.should_archive(neighbor)
    assert archive is False


def test_should_archive_low_score_old_true():
    """score<0.2 + age>60d → True even if used."""
    neighbor = {
        "id": "a4",
        "usage_count": 3,
        "score": 0.1,
        "created_at": _iso(datetime.now(timezone.utc) - timedelta(days=70)),
    }
    archive, reason = lcd.should_archive(neighbor)
    assert archive is True
    assert "low score" in reason


# ---------------------------------------------------------------------------
# enrich_decision — cascade behavior
# ---------------------------------------------------------------------------


def test_enrich_decision_add_with_obsolete_neighbor_emits_secondary_expire():
    cand = Candidate(type="fact", content="The capital of Spain is Madrid.")
    base = Decision("ADD", reason="new memory")
    neighbors = [
        {
            "id": "obsolete",
            "type": "fact",
            "content": "Madrid is the capital of Spain.",
            "similarity": 0.93,
            "created_at": _iso(datetime.now(timezone.utc) - timedelta(days=120)),
            "usage_count": 5,
            "score": 0.5,
        }
    ]
    enriched = lcd.enrich_decision(base, cand, neighbors, use_gemma=False)
    assert enriched.action == "ADD"
    assert len(enriched.secondary_actions) == 1
    sec = enriched.secondary_actions[0]
    assert sec.action == "EXPIRE"
    assert sec.target_memory_id == "obsolete"


def test_enrich_decision_merge_with_stale_target_becomes_archive():
    cand = Candidate(type="fact", content="x")
    base = Decision("MERGE", target_memory_id="stale-target", reason="dup")
    neighbors = [
        {
            "id": "stale-target",
            "type": "fact",
            "content": "x",
            "similarity": 0.95,
            "created_at": _iso(datetime.now(timezone.utc) - timedelta(days=120)),
            "usage_count": 0,
            "score": 0.05,
        }
    ]
    enriched = lcd.enrich_decision(base, cand, neighbors, use_gemma=False)
    assert enriched.action == "ARCHIVE"
    assert enriched.target_memory_id == "stale-target"
    assert "stale" in enriched.reason or "archive" in enriched.reason.lower()


def test_enrich_decision_ignore_passthrough():
    base = Decision("IGNORE", reason="empty")
    cand = Candidate(type="fact", content="")
    enriched = lcd.enrich_decision(base, cand, [{"id": "n", "score": 0.0,
                                                 "usage_count": 0,
                                                 "created_at": _iso(datetime.now(timezone.utc) - timedelta(days=100))}])
    assert enriched.action == "IGNORE"
    assert not getattr(enriched, "secondary_actions", [])


def test_enrich_decision_no_neighbors_passthrough():
    base = Decision("ADD", reason="new")
    cand = Candidate(type="fact", content="hello")
    enriched = lcd.enrich_decision(base, cand, [])
    assert enriched.action == "ADD"
    assert not enriched.secondary_actions


def test_enrich_decision_mock_gemma_contradictory(monkeypatch):
    monkeypatch.delenv("MEMOIRS_CURATOR_ENABLED", raising=False)
    monkeypatch.delenv("MEMOIRS_GEMMA_CURATOR", raising=False)
    """Mock Gemma with contradictory=True → enrich_decision generates EXPIRE."""

    def fake_resolve(a, b):
        return {"contradictory": True, "winner": "A", "reason": "candidate fresh", "action": "MARK_CONFLICT"}

    monkeypatch.setattr(curator_mod, "_have_curator", lambda: True)
    monkeypatch.setattr(curator_mod, "curator_resolve_conflict", fake_resolve)

    cand = Candidate(type="preference", content="I prefer dark mode.")
    base = Decision("ADD")
    neighbors = [
        {
            "id": "old-pref",
            "type": "preference",
            "content": "User likes light mode.",
            "similarity": 0.5,
            "created_at": _iso(datetime.now(timezone.utc) - timedelta(days=5)),
            "usage_count": 2,
            "score": 0.4,
        }
    ]
    enriched = lcd.enrich_decision(base, cand, neighbors)
    assert enriched.action == "ADD"
    assert len(enriched.secondary_actions) == 1
    assert enriched.secondary_actions[0].action == "EXPIRE"
    assert enriched.secondary_actions[0].target_memory_id == "old-pref"


def test_enrich_decision_caps_secondary_count():
    """No more than 3 secondary actions emitted per ingest."""
    cand = Candidate(type="fact", content="The capital of France is Paris.")
    base = Decision("ADD")
    neighbors = [
        {
            "id": f"old-{i}",
            "type": "fact",
            "content": "Paris is the capital of France.",
            "similarity": 0.95,
            "created_at": _iso(datetime.now(timezone.utc) - timedelta(days=120)),
            "usage_count": 1,
            "score": 0.4,
        }
        for i in range(10)
    ]
    enriched = lcd.enrich_decision(base, cand, neighbors, use_gemma=False)
    assert len(enriched.secondary_actions) <= 3


# ---------------------------------------------------------------------------
# apply_decision — EXPIRE / ARCHIVE branches
# ---------------------------------------------------------------------------


def test_apply_decision_expire_sets_valid_to_and_status(tmp_db):
    target_id = _seed(tmp_db, memory_id="target1", content="old", age_days=60)
    cand = Candidate(type="fact", content="new content")
    decision = Decision("EXPIRE", target_memory_id=target_id, reason="superseded")
    res = apply_decision(tmp_db, cand, decision)
    assert res["expired_id"] == target_id

    row = _row(tmp_db, target_id)
    assert row is not None
    assert row["valid_to"] is not None
    meta = json.loads(row["metadata_json"] or "{}")
    assert meta.get("status") == "expired"
    assert meta.get("expire_reason") == "superseded"


def test_apply_decision_archive_sets_archived_at(tmp_db):
    target_id = _seed(tmp_db, memory_id="target2", age_days=120)
    cand = Candidate(type="fact", content="x")
    decision = Decision("ARCHIVE", target_memory_id=target_id, reason="stale")
    res = apply_decision(tmp_db, cand, decision)
    assert res["archived_id"] == target_id

    row = _row(tmp_db, target_id)
    assert row is not None
    assert row["archived_at"] is not None
    assert row["archive_reason"] == "stale"
    meta = json.loads(row["metadata_json"] or "{}")
    assert meta.get("archive_rule") == "stale"


def test_apply_decision_runs_secondary_actions(tmp_db):
    """A primary ADD with a secondary EXPIRE produces both effects."""
    neighbor_id = _seed(
        tmp_db, memory_id="nbr1", content="old fact", age_days=60,
    )
    cand = Candidate(type="fact", content="brand new fact")
    primary = Decision(action="ADD", reason="new")
    primary.secondary_actions = [
        Decision(action="EXPIRE", target_memory_id=neighbor_id, reason="superseded"),
    ]
    res = apply_decision(tmp_db, cand, primary)
    assert "memory_id" in res
    assert res["secondary_results"][0]["expired_id"] == neighbor_id

    row = _row(tmp_db, neighbor_id)
    assert row["valid_to"] is not None


# ---------------------------------------------------------------------------
# Integration: decide_memory_action calls enrich_decision
# ---------------------------------------------------------------------------


def test_decide_memory_action_emits_secondary_expire(tmp_db, monkeypatch):
    """End-to-end: candidate similar to old neighbor → ADD with secondary EXPIRE.

    Uses the heuristic path (curator=off) so we don't depend on Gemma.
    """
    monkeypatch.setenv("MEMOIRS_GEMMA_CURATOR", "off")

    neighbor_id = _seed(
        tmp_db,
        memory_id="oldfact",
        mem_type="fact",
        content="The capital of France is Lyon.",  # outdated
        age_days=120,
    )

    # Stub out neighbor gathering: we control the similarity score directly.
    def fake_neighbors(db, candidate, *, top_k=5):
        return [{
            "id": neighbor_id,
            "type": "fact",
            "content": "The capital of France is Lyon.",
            "similarity": 0.95,
            "created_at": _iso(datetime.now(timezone.utc) - timedelta(days=120)),
            "usage_count": 1,
            "score": 0.4,
        }]

    monkeypatch.setattr(memory_engine, "_gather_curator_neighbors", fake_neighbors)

    # Avoid the semantic-duplicate detector (which would force CONTRADICTION
    # on type-mismatch — we want a clean ADD path).
    monkeypatch.setattr(memory_engine, "detect_semantic_duplicate", lambda *a, **k: None)

    cand = Candidate(type="fact", content="The capital of France is Paris.")
    decision = decide_memory_action(tmp_db, cand)
    assert decision.action == "ADD"
    assert any(s.action == "EXPIRE" and s.target_memory_id == neighbor_id
               for s in decision.secondary_actions), \
           f"expected EXPIRE secondary, got {decision.secondary_actions!r}"


def test_consolidate_with_secondary_archive_marks_neighbor(tmp_db, monkeypatch):
    """Integration: ingest → consolidate → neighbor archived via cascade."""
    monkeypatch.setenv("MEMOIRS_GEMMA_CURATOR", "off")

    stale_id = _seed(
        tmp_db, memory_id="stale", mem_type="fact",
        content="ancient unused fact",
        usage_count=0, age_days=120, score=0.05,
    )

    def fake_neighbors(db, candidate, *, top_k=5):
        return [{
            "id": stale_id,
            "type": "fact",
            "content": "unrelated content",
            "similarity": 0.3,  # not similar enough to expire
            "created_at": _iso(datetime.now(timezone.utc) - timedelta(days=120)),
            "usage_count": 0,
            "score": 0.05,
        }]

    monkeypatch.setattr(memory_engine, "_gather_curator_neighbors", fake_neighbors)
    monkeypatch.setattr(memory_engine, "detect_semantic_duplicate", lambda *a, **k: None)

    cand = Candidate(type="fact", content="totally different fact")
    decision = decide_memory_action(tmp_db, cand)
    apply_decision(tmp_db, cand, decision)

    row = _row(tmp_db, stale_id)
    assert row["archived_at"] is not None, "stale neighbor should be archived"
