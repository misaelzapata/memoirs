"""Targeted coverage tests for memoirs/engine/memory_engine.py.

Focus areas (uncovered branches in the baseline):
  * recompute_all_scores
  * detect_exact_duplicate / detect_semantic_duplicate
  * decide_memory_action: IGNORE, UPDATE, MERGE, CONTRADICTION, ADD
  * apply_decision: ADD path, UPDATE path, CONTRADICTION path, target missing
  * archive_low_value_memories: empty corpus + dynamic + static threshold
  * expire_old_memories
  * create_memory_version: success + missing target
  * enqueue_event / process_event_queue dispatch + failure handling
  * _resolve_retrieval_mode: unknown value falls back to hybrid
  * _detect_conflicting + _resolve_conflicts (without Gemma loaded)
  * _compress_context + _summary_for edge cases
  * assemble_context end-to-end + time-travel branch
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone, timedelta

import pytest

from memoirs.config import EMBEDDING_DIM
from memoirs.engine import embeddings as emb
from memoirs.engine import memory_engine as me
from memoirs.engine.curator import Candidate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(angle_rad: float) -> list[float]:
    v = [0.0] * EMBEDDING_DIM
    v[0] = math.cos(angle_rad)
    v[1] = math.sin(angle_rad)
    return v


def _seed(db, *, memory_id: str, mem_type: str = "fact", content: str = "x",
          score: float = 0.5, importance: int = 3, confidence: float = 0.5,
          usage_count: int = 0, archived: bool = False,
          valid_to: str | None = None, angle_rad: float | None = None,
          created_at: str | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    created_at = created_at or now
    archived_at = "2026-01-01T00:00:00+00:00" if archived else None
    db.conn.execute(
        "INSERT INTO memories (id, type, content, content_hash, importance, "
        "confidence, score, usage_count, user_signal, valid_from, valid_to, "
        "archived_at, metadata_json, created_at, updated_at) "
        "VALUES (?, ?, ?, 'h_'||?, ?, ?, ?, ?, 0, ?, ?, ?, '{}', ?, ?)",
        (memory_id, mem_type, content, memory_id, importance, confidence, score,
         usage_count, now, valid_to, archived_at, created_at, now),
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
# Score recompute pass
# ---------------------------------------------------------------------------


def test_recompute_all_scores_walks_active(tmp_db):
    _seed(tmp_db, memory_id="r1", importance=5)
    _seed(tmp_db, memory_id="r2", importance=1)
    _seed(tmp_db, memory_id="r3", archived=True)
    n = me.recompute_all_scores(tmp_db)
    assert n == 2  # archived skipped
    s1 = tmp_db.conn.execute("SELECT score FROM memories WHERE id='r1'").fetchone()[0]
    s2 = tmp_db.conn.execute("SELECT score FROM memories WHERE id='r2'").fetchone()[0]
    assert s1 > s2  # higher importance wins


# ---------------------------------------------------------------------------
# Dedup helpers
# ---------------------------------------------------------------------------


def test_detect_exact_duplicate_hits_and_misses(tmp_db):
    from memoirs.core.ids import content_hash
    h = content_hash("alpha")
    tmp_db.conn.execute(
        "INSERT INTO memories (id, type, content, content_hash, importance, "
        "confidence, score, usage_count, user_signal, valid_from, metadata_json, "
        "created_at, updated_at) VALUES "
        "('m1', 'fact', 'alpha', ?, 3, 0.5, 0.5, 0, 0, "
        "'2026-04-27T00:00:00+00:00', '{}', "
        "'2026-04-27T00:00:00+00:00', '2026-04-27T00:00:00+00:00')",
        (h,),
    )
    tmp_db.conn.commit()
    hit = me.detect_exact_duplicate(tmp_db, "alpha")
    assert hit is not None and hit["id"] == "m1"
    miss = me.detect_exact_duplicate(tmp_db, "beta")
    assert miss is None


def test_detect_semantic_duplicate_uses_embeddings(tmp_db, monkeypatch):
    _seed(tmp_db, memory_id="m1", content="alpha", angle_rad=0.0)
    monkeypatch.setattr(emb, "embed_text", lambda t: _unit_vec(0.0))
    hit = me.detect_semantic_duplicate(tmp_db, "alpha", threshold=0.5)
    assert hit is not None
    assert hit["id"] == "m1"


# ---------------------------------------------------------------------------
# decide_memory_action — every branch
# ---------------------------------------------------------------------------


def test_decide_action_empty_content_ignored(tmp_db):
    cand = Candidate(type="fact", content="   ", importance=3, confidence=0.5)
    decision = me.decide_memory_action(tmp_db, cand)
    assert decision.action == "IGNORE"


def test_decide_action_exact_dup_returns_update(tmp_db):
    # NOTE: content must clear ``should_skip_extraction``'s 20-char minimum,
    # otherwise the curator REJECTs as noise before reaching this branch.
    seed_content = "user prefers the alpha approach"
    from memoirs.core.ids import content_hash
    h = content_hash(seed_content)
    tmp_db.conn.execute(
        "INSERT INTO memories (id, type, content, content_hash, importance, "
        "confidence, score, usage_count, user_signal, valid_from, metadata_json, "
        "created_at, updated_at) VALUES "
        "('m1', 'fact', ?, ?, 3, 0.5, 0.5, 0, 0, "
        "'2026-04-27T00:00:00+00:00', '{}', "
        "'2026-04-27T00:00:00+00:00', '2026-04-27T00:00:00+00:00')",
        (seed_content, h,),
    )
    tmp_db.conn.commit()
    cand = Candidate(type="fact", content=seed_content, importance=4, confidence=0.7)
    decision = me.decide_memory_action(tmp_db, cand)
    assert decision.action == "UPDATE"
    assert decision.target_memory_id == "m1"


def test_decide_action_semantic_same_type_returns_merge(tmp_db, monkeypatch):
    # Content must clear ``should_skip_extraction``'s 20-char minimum.
    _seed(tmp_db, memory_id="m1", mem_type="fact",
          content="user prefers the alpha approach", angle_rad=0.0)
    monkeypatch.setattr(emb, "embed_text", lambda t: _unit_vec(0.0))
    cand = Candidate(type="fact",
                     content="user prefers the alpha approach v2",
                     importance=3, confidence=0.5)
    decision = me.decide_memory_action(tmp_db, cand)
    # Same-type + high similarity: P1-10 enrich path may resolve to MERGE or
    # UPDATE depending on neighbor staleness. Both refine the existing row.
    assert decision.action in {"MERGE", "UPDATE"}
    assert decision.target_memory_id == "m1"


def test_decide_action_semantic_diff_type_returns_contradiction(tmp_db, monkeypatch):
    # NOTE: content must clear ``should_skip_extraction``'s 20-char minimum,
    # otherwise the curator REJECTs as noise before reaching the
    # semantic-duplicate branch we're exercising here. The conftest autouse
    # fixture forces MEMOIRS_GEMMA_CURATOR=off so the heuristic path is
    # deterministic regardless of whether Qwen3 GGUF is installed locally.
    _seed(tmp_db, memory_id="m1", mem_type="preference",
          content="user prefers the alpha approach", angle_rad=0.0)
    monkeypatch.setattr(emb, "embed_text", lambda t: _unit_vec(0.0))
    cand = Candidate(type="fact",
                     content="the alpha approach is the new standard",
                     importance=3, confidence=0.5)
    decision = me.decide_memory_action(tmp_db, cand)
    assert decision.action == "CONTRADICTION"


def test_decide_action_no_dup_returns_add(tmp_db, monkeypatch):
    monkeypatch.setattr(emb, "embed_text", lambda t: _unit_vec(0.0))
    monkeypatch.setattr(emb, "find_semantic_duplicates", lambda *a, **k: [])
    # Content must clear ``should_skip_extraction``'s 20-char minimum.
    cand = Candidate(type="fact",
                     content="user wants a brand-new memory line",
                     importance=3, confidence=0.5)
    decision = me.decide_memory_action(tmp_db, cand)
    assert decision.action == "ADD"


# ---------------------------------------------------------------------------
# apply_decision
# ---------------------------------------------------------------------------


def test_apply_decision_ignore_is_noop(tmp_db):
    cand = Candidate(type="fact", content="x")
    result = me.apply_decision(tmp_db, cand, me.Decision("IGNORE", reason="test"))
    assert result == {"action": "IGNORE", "reason": "test"}


def test_apply_decision_update_bumps_confidence(tmp_db, monkeypatch):
    _seed(tmp_db, memory_id="m1", confidence=0.5, importance=3, usage_count=0,
          angle_rad=0.0)
    monkeypatch.setattr(emb, "embed_text", lambda t: _unit_vec(0.0))
    # Bypass zettelkasten side effects.
    monkeypatch.setattr(me, "_maybe_link_memory", lambda *a, **k: None)
    cand = Candidate(type="fact", content="alpha", importance=4, confidence=0.9)
    decision = me.Decision("UPDATE", target_memory_id="m1", reason="dup")
    result = me.apply_decision(tmp_db, cand, decision)
    assert result["memory_id"] == "m1"
    row = tmp_db.conn.execute(
        "SELECT confidence, importance, usage_count FROM memories WHERE id='m1'"
    ).fetchone()
    assert row["confidence"] > 0.5
    assert row["importance"] == 4
    assert row["usage_count"] == 1


def test_apply_decision_contradiction_writes_metadata(tmp_db):
    _seed(tmp_db, memory_id="m1")
    cand = Candidate(type="fact", content="conflicting story", importance=3, confidence=0.5)
    decision = me.Decision("CONTRADICTION", target_memory_id="m1", reason="diff")
    result = me.apply_decision(tmp_db, cand, decision)
    assert result["memory_id"] == "m1"
    meta = tmp_db.conn.execute(
        "SELECT metadata_json FROM memories WHERE id='m1'"
    ).fetchone()[0]
    assert "contradiction" in meta


# ---------------------------------------------------------------------------
# archive_low_value_memories + expire_old_memories
# ---------------------------------------------------------------------------


def test_archive_low_value_empty_corpus_returns_zero(tmp_db):
    assert me.archive_low_value_memories(tmp_db) == 0


def test_archive_low_value_dynamic_threshold(tmp_db):
    old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat(timespec="seconds")
    for i in range(10):
        _seed(tmp_db, memory_id=f"a{i}", score=i / 10.0, created_at=old)
    archived = me.archive_low_value_memories(tmp_db, percentile=0.20)
    # bottom 20% of 10 rows = 2 rows scored < score_at_idx_2
    assert archived >= 1
    rest = tmp_db.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE archived_at IS NULL"
    ).fetchone()[0]
    assert rest < 10


def test_archive_low_value_skips_recent(tmp_db):
    """Recent low-score rows must not be archived (min_age_days guard)."""
    for i in range(5):
        _seed(tmp_db, memory_id=f"r{i}", score=0.1)  # created_at = now
    archived = me.archive_low_value_memories(tmp_db, score_threshold=0.5, min_age_days=30)
    assert archived == 0


def test_expire_old_memories(tmp_db):
    past = "2020-01-01T00:00:00+00:00"
    _seed(tmp_db, memory_id="x1", valid_to=past)
    _seed(tmp_db, memory_id="x2")  # no valid_to
    n = me.expire_old_memories(tmp_db)
    assert n == 1
    row = tmp_db.conn.execute(
        "SELECT archived_at, archive_reason FROM memories WHERE id = 'x1'"
    ).fetchone()
    assert row["archived_at"] is not None
    assert "valid_to" in row["archive_reason"]


# ---------------------------------------------------------------------------
# create_memory_version
# ---------------------------------------------------------------------------


def test_create_memory_version_archives_old(tmp_db, monkeypatch):
    _seed(tmp_db, memory_id="old", content="v1", angle_rad=0.0)
    monkeypatch.setattr(emb, "embed_text", lambda t: _unit_vec(0.0))
    new_id = me.create_memory_version(tmp_db, "old", "v2 content")
    assert new_id != "old"
    old_row = tmp_db.conn.execute(
        "SELECT valid_to, superseded_by FROM memories WHERE id='old'"
    ).fetchone()
    assert old_row["valid_to"] is not None
    assert old_row["superseded_by"] == new_id


def test_create_memory_version_raises_when_target_missing(tmp_db):
    with pytest.raises(ValueError, match="memory not found"):
        me.create_memory_version(tmp_db, "does-not-exist", "x")


# ---------------------------------------------------------------------------
# Event queue
# ---------------------------------------------------------------------------


def test_enqueue_and_process_event_queue_handles_unknown_type(tmp_db):
    qid = me.enqueue_event(tmp_db, "no_such_event", {"x": 1})
    assert isinstance(qid, int)
    out = me.process_event_queue(tmp_db)
    # Unknown event types are silently no-op and marked done.
    assert out["processed"] >= 1
    row = tmp_db.conn.execute(
        "SELECT status FROM event_queue WHERE id = ?", (qid,)
    ).fetchone()
    assert row["status"] == "done"


def test_process_event_queue_marks_failed_on_bad_payload(tmp_db):
    """A malformed payload triggers the JSON-decode failure branch."""
    tmp_db.conn.execute(
        "INSERT INTO event_queue (event_type, payload_json, status, created_at) "
        "VALUES ('chat_message', '{{not-json', 'pending', '2026-04-27T00:00:00+00:00')"
    )
    tmp_db.conn.commit()
    out = me.process_event_queue(tmp_db)
    assert out["failed"] >= 1


# ---------------------------------------------------------------------------
# Retrieval-mode resolver
# ---------------------------------------------------------------------------


def test_resolve_retrieval_mode_unknown_falls_back(monkeypatch):
    monkeypatch.delenv("MEMOIRS_RETRIEVAL_MODE", raising=False)
    assert me._resolve_retrieval_mode("not-a-mode") == "hybrid"
    assert me._resolve_retrieval_mode(None) == "hybrid"
    assert me._resolve_retrieval_mode("dense") == "dense"


def test_resolve_retrieval_mode_env_var(monkeypatch):
    monkeypatch.setenv("MEMOIRS_RETRIEVAL_MODE", "bm25")
    assert me._resolve_retrieval_mode(None) == "bm25"


# ---------------------------------------------------------------------------
# Conflict detection and helpers
# ---------------------------------------------------------------------------


def test_detect_conflicting_finds_polarity_pairs():
    a = {"id": "a", "type": "fact", "content": "I do not like X", "similarity": 0.7}
    b = {"id": "b", "type": "fact", "content": "I like X always",  "similarity": 0.7}
    pairs = me._detect_conflicting([a, b])
    assert len(pairs) == 1


def test_detect_conflicting_skips_different_type():
    a = {"id": "a", "type": "fact", "content": "x", "similarity": 0.9}
    b = {"id": "b", "type": "preference", "content": "x", "similarity": 0.9}
    assert me._detect_conflicting([a, b]) == []


def test_resolve_conflicts_uses_score_fallback(monkeypatch):
    """With the curator absent, lower-scored memory in the pair gets dropped."""
    from memoirs.engine import curator as _curator
    monkeypatch.setattr(_curator, "_have_curator", lambda: False)
    a = {"id": "a", "type": "fact", "content": "x", "score": 0.9}
    b = {"id": "b", "type": "fact", "content": "y", "score": 0.2}
    out = me._resolve_conflicts([a, b], [(a, b)])
    ids = {m["id"] for m in out}
    assert "a" in ids and "b" not in ids


def test_compress_context_respects_max_chars():
    mems = [
        {"type": "fact", "content": "a" * 100},
        {"type": "fact", "content": "b" * 100},
        {"type": "fact", "content": "c" * 100},
    ]
    out = me._compress_context(mems, max_chars=150)
    # First fits, second would exceed → break.
    assert len(out) == 1


def test_summary_for_truncates_long_content():
    mem = {"content": "x" * 500}
    out = me._summary_for(mem, max_chars=50)
    assert len(out) <= 50
    assert out.endswith("…")


def test_summary_for_short_returns_as_is():
    assert me._summary_for({"content": "short"}, max_chars=50) == "short"


# ---------------------------------------------------------------------------
# assemble_context end-to-end (light)
# ---------------------------------------------------------------------------


def test_assemble_context_dense_mode(tmp_db, monkeypatch):
    """Drains the streaming generator via the wrapper. dense mode skips hybrid."""
    _seed(tmp_db, memory_id="m1", content="alpha", angle_rad=0.0)
    monkeypatch.setattr(emb, "embed_text", lambda t: _unit_vec(0.0))
    out = me.assemble_context(
        tmp_db, "alpha", top_k=5, max_lines=5, retrieval_mode="dense",
    )
    assert "context" in out
    assert "memories" in out
    assert out["live"] is True


def test_assemble_context_time_travel_no_usage_increment(tmp_db, monkeypatch):
    """as_of != None → live=False → no usage_count side effect."""
    _seed(tmp_db, memory_id="m1", content="alpha", angle_rad=0.0)
    tmp_db.conn.execute(
        "UPDATE memories SET valid_from = '2026-01-01T00:00:00+00:00', "
        "usage_count = 0 WHERE id='m1'"
    )
    tmp_db.conn.commit()
    monkeypatch.setattr(emb, "embed_text", lambda t: _unit_vec(0.0))
    out = me.assemble_context(
        tmp_db, "alpha", top_k=5, retrieval_mode="dense",
        as_of="2026-06-01T00:00:00+00:00",
    )
    assert out["live"] is False
    after = tmp_db.conn.execute(
        "SELECT usage_count FROM memories WHERE id='m1'"
    ).fetchone()[0]
    assert after == 0


# ---------------------------------------------------------------------------
# Ebbinghaus recency (P1-7) + parser
# ---------------------------------------------------------------------------


def test_parse_iso_handles_z_suffix():
    out = me._parse_iso("2026-04-27T12:00:00Z")
    assert out is not None
    assert out.tzinfo is not None


def test_parse_iso_returns_none_for_garbage():
    assert me._parse_iso("not-a-date") is None
    assert me._parse_iso(None) is None
    assert me._parse_iso("") is None


def test_parse_iso_assumes_utc_for_naive():
    out = me._parse_iso("2026-04-27T12:00:00")
    assert out is not None
    assert out.tzinfo is not None  # injected UTC


def test_ebbinghaus_recency_none_returns_one():
    """No timestamp → treated as perfectly fresh."""
    assert me.ebbinghaus_recency(None, strength=1.0) == 1.0


def test_ebbinghaus_recency_unparseable_returns_half():
    assert me.ebbinghaus_recency("garbage", strength=1.0) == 0.5


def test_ebbinghaus_recency_decays_over_time():
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    past = (now - timedelta(days=2)).isoformat(timespec="seconds")
    fresh = me.ebbinghaus_recency(now.isoformat(timespec="seconds"), strength=1.0, now=now)
    decayed = me.ebbinghaus_recency(past, strength=1.0, now=now)
    assert fresh > decayed
    # Strength dampens decay.
    decayed_strong = me.ebbinghaus_recency(past, strength=10.0, now=now)
    assert decayed_strong > decayed


def test_ebbinghaus_recency_clamps_to_floor():
    """Very old memory still scores at least 0.01 (so importance can surface it)."""
    score = me.ebbinghaus_recency("1970-01-01T00:00:00+00:00", strength=1.0)
    assert score >= 0.01


def test_record_access_no_op_on_missing_table(tmp_db):
    """If migration 004 hasn't run / row is missing, this never raises."""
    # The fixture runs all migrations, so just call on a non-existent id.
    me.record_access(tmp_db, "non-existent-id")


# ---------------------------------------------------------------------------
# apply_decision ADD path — direct, without monkeypatching _maybe_link_memory
# (it returns early when zettelkasten is disabled)
# ---------------------------------------------------------------------------


def test_apply_decision_add_inserts_memory(tmp_db, monkeypatch):
    """ADD path persists the memory + computes a score."""
    monkeypatch.setattr(emb, "embed_text", lambda t: _unit_vec(0.0))
    # Disable zettelkasten via env so _maybe_link_memory takes the early-return.
    monkeypatch.setenv("MEMOIRS_ZETTELKASTEN", "0")
    cand = Candidate(type="fact", content="brand new memory line", importance=4, confidence=0.8)
    decision = me.Decision("ADD", reason="new")
    result = me.apply_decision(tmp_db, cand, decision)
    assert "memory_id" in result
    row = tmp_db.conn.execute(
        "SELECT type, content, score FROM memories WHERE id = ?", (result["memory_id"],)
    ).fetchone()
    assert row["type"] == "fact"
    assert row["content"] == "brand new memory line"
    assert row["score"] > 0


# ---------------------------------------------------------------------------
# consolidate_candidate / consolidate_pending
# ---------------------------------------------------------------------------


def test_consolidate_pending_processes_candidates(tmp_db, monkeypatch):
    """Insert a fake candidate and verify it gets promoted via the consolidate path."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tmp_db.conn.execute(
        "INSERT INTO memory_candidates (id, conversation_id, type, content, importance, "
        "confidence, status, entities, source_message_ids, extractor, created_at, updated_at) "
        "VALUES ('c1', NULL, 'fact', 'consolidate me please', 3, 0.7, 'pending', '[]', '[]', 'heuristic', ?, ?)",
        (now, now),
    )
    tmp_db.conn.commit()
    monkeypatch.setattr(emb, "embed_text", lambda t: _unit_vec(0.0))
    monkeypatch.setattr(emb, "find_semantic_duplicates", lambda *a, **k: [])
    monkeypatch.setenv("MEMOIRS_ZETTELKASTEN", "0")
    out = me.consolidate_pending(tmp_db, limit=5)
    assert out["processed"] >= 1
    assert "ADD" in out["by_action"]
    # Candidate row marked accepted with a promoted_memory_id pointer.
    row = tmp_db.conn.execute(
        "SELECT status, promoted_memory_id FROM memory_candidates WHERE id='c1'"
    ).fetchone()
    assert row["status"] == "accepted"
    assert row["promoted_memory_id"] is not None
