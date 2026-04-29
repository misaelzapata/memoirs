"""Tests for engine/hybrid_retrieval.

Covers:
  - FTS5 schema + triggers (INSERT/UPDATE/DELETE sync)
  - bm25_search prefers exact lexical matches
  - dense_search prefers semantic matches without lexical overlap (mocked)
  - rrf_fuse combines rankings correctly (formula + ordering)
  - hybrid_search latency target on a 100-memory corpus
  - retrieval_mode wiring through assemble_context
  - rebuild_fts_index backfill
  - graceful degradation when sqlite-vec is missing
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from memoirs.core.ids import content_hash, stable_id, utc_now
from memoirs.engine import hybrid_retrieval as hr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed(db, memory_id: str, content: str, *, mtype: str = "fact",
          archived: bool = False) -> str:
    now = utc_now()
    db.conn.execute(
        """
        INSERT INTO memories (
            id, type, content, content_hash, importance, confidence, score,
            usage_count, user_signal, valid_from, metadata_json,
            created_at, updated_at, archived_at
        ) VALUES (?, ?, ?, ?, 3, 0.7, 0.5, 0, 0, ?, '{}', ?, ?, ?)
        """,
        (memory_id, mtype, content, content_hash(content + memory_id),
         now, now, now, now if archived else None),
    )
    db.conn.commit()
    return memory_id


def _make_fake_dense(rankings: dict[str, list[tuple[str, float]]]):
    """Returns a fake dense_search that consults the per-query map.

    Maps `query` (lowercased) → list[(memory_id, similarity)]. Useful when
    sentence-transformers is unavailable in the test environment or we
    want deterministic semantic ordering.
    """
    def _fake(db, query: str, top_k: int = 20, *, as_of=None):
        return rankings.get(query.lower(), [])[:top_k]
    return _fake


# ---------------------------------------------------------------------------
# Schema + triggers
# ---------------------------------------------------------------------------


def test_ensure_fts_schema_creates_table(tmp_db):
    hr.ensure_fts_schema(tmp_db.conn)
    row = tmp_db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'"
    ).fetchone()
    assert row is not None
    # All three triggers exist.
    triggers = {
        r[0] for r in tmp_db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' "
            "AND name LIKE 'memories_fts_%'"
        ).fetchall()
    }
    assert {"memories_fts_ai", "memories_fts_ad", "memories_fts_au"} <= triggers


def test_triggers_keep_fts_in_sync_on_insert(tmp_db):
    hr.ensure_fts_schema(tmp_db.conn)
    _seed(tmp_db, "m1", "the quick brown fox jumps over the lazy dog")
    rows = tmp_db.conn.execute(
        "SELECT memory_id FROM memories_fts WHERE memories_fts MATCH ?",
        ('"brown" "fox"',),
    ).fetchall()
    assert any(r[0] == "m1" for r in rows)


def test_triggers_handle_update_and_delete(tmp_db):
    hr.ensure_fts_schema(tmp_db.conn)
    _seed(tmp_db, "m1", "redis cluster failover behavior")
    # UPDATE: content change must be reflected in the index.
    tmp_db.conn.execute(
        "UPDATE memories SET content = ?, updated_at = ? WHERE id = 'm1'",
        ("postgres replica promotion", utc_now()),
    )
    tmp_db.conn.commit()
    redis_hits = tmp_db.conn.execute(
        "SELECT memory_id FROM memories_fts WHERE memories_fts MATCH 'redis'"
    ).fetchall()
    pg_hits = tmp_db.conn.execute(
        "SELECT memory_id FROM memories_fts WHERE memories_fts MATCH 'postgres'"
    ).fetchall()
    assert redis_hits == []
    assert any(r[0] == "m1" for r in pg_hits)
    # DELETE: row goes away from the index.
    tmp_db.conn.execute("DELETE FROM memories WHERE id = 'm1'")
    tmp_db.conn.commit()
    after = tmp_db.conn.execute(
        "SELECT memory_id FROM memories_fts WHERE memory_id = 'm1'"
    ).fetchall()
    assert after == []


def test_archive_removes_from_fts(tmp_db):
    """Archiving a memory must purge it from the FTS index.

    The lexical search should never surface a memory that lifecycle has
    soft-deleted. The trigger fires on UPDATE OF archived_at.
    """
    hr.ensure_fts_schema(tmp_db.conn)
    _seed(tmp_db, "m1", "kubernetes ingress nginx")
    tmp_db.conn.execute(
        "UPDATE memories SET archived_at = ?, updated_at = ? WHERE id = 'm1'",
        (utc_now(), utc_now()),
    )
    tmp_db.conn.commit()
    hits = tmp_db.conn.execute(
        "SELECT memory_id FROM memories_fts WHERE memories_fts MATCH 'kubernetes'"
    ).fetchall()
    assert hits == []


# ---------------------------------------------------------------------------
# Backfill / rebuild
# ---------------------------------------------------------------------------


def test_rebuild_fts_index_backfills_existing_rows(tmp_db):
    # Insert rows BEFORE the FTS schema exists — simulates a legacy DB that
    # never went through migration 003. We also drop any auto-created
    # memories_fts rows to be sure rebuild does the work.
    _seed(tmp_db, "m1", "alpha bravo charlie")
    _seed(tmp_db, "m2", "delta echo foxtrot")
    # Ensure the table exists, then nuke contents and rebuild.
    hr.ensure_fts_schema(tmp_db.conn)
    tmp_db.conn.execute("DELETE FROM memories_fts")
    tmp_db.conn.commit()
    n = hr.rebuild_fts_index(tmp_db.conn)
    assert n == 2
    matches = {
        r[0] for r in tmp_db.conn.execute(
            "SELECT memory_id FROM memories_fts WHERE memories_fts MATCH 'bravo OR echo'"
        ).fetchall()
    }
    assert matches == {"m1", "m2"}


def test_rebuild_fts_excludes_archived(tmp_db):
    hr.ensure_fts_schema(tmp_db.conn)
    _seed(tmp_db, "m1", "active row")
    _seed(tmp_db, "m2", "archived row", archived=True)
    tmp_db.conn.execute("DELETE FROM memories_fts")
    n = hr.rebuild_fts_index(tmp_db.conn)
    # Only the non-archived row should be reindexed.
    assert n == 1
    rows = tmp_db.conn.execute(
        "SELECT memory_id FROM memories_fts"
    ).fetchall()
    assert {r[0] for r in rows} == {"m1"}


# ---------------------------------------------------------------------------
# BM25 ranking
# ---------------------------------------------------------------------------


def test_bm25_prefers_exact_lexical_match(tmp_db):
    """Query for a rare token surfaces the doc that contains it, even when
    surrounded by docs that share many common stopwords."""
    hr.ensure_fts_schema(tmp_db.conn)
    _seed(tmp_db, "m1", "the database engine handles transactions and rollback")
    _seed(tmp_db, "m2", "kubernetes ingress nginx routing rules")
    _seed(tmp_db, "m3", "the quick brown fox jumps over the lazy dog")
    results = hr.bm25_search(tmp_db.conn, "kubernetes ingress", top_k=5)
    assert results, "bm25 returned no results"
    top_id = results[0][0]
    assert top_id == "m2"
    # m1 / m3 don't contain the query terms, so they shouldn't appear.
    assert "m1" not in {r[0] for r in results}


def test_bm25_handles_special_characters_safely(tmp_db):
    """A query with FTS5 reserved chars must not blow up; tokens are quoted."""
    hr.ensure_fts_schema(tmp_db.conn)
    _seed(tmp_db, "m1", "C++ template metaprogramming patterns")
    # Should not raise; should still find m1 (token "C" or "C++" depending on tokenizer).
    results = hr.bm25_search(tmp_db.conn, "C++ template (advanced)", top_k=5)
    ids = {r[0] for r in results}
    assert "m1" in ids


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def test_rrf_fuse_basic_formula():
    """Items appearing in both lists with low ranks dominate the fused order."""
    a = [("x", 10.0), ("y", 5.0), ("z", 1.0)]
    b = [("y", 0.9), ("x", 0.8), ("w", 0.7)]
    fused = hr.rrf_fuse([a, b], k=60)
    ids = [m for m, _ in fused]
    # x: 1/(60+1) + 1/(60+2) ≈ 0.03279
    # y: 1/(60+2) + 1/(60+1) ≈ 0.03279 (tied with x)
    # z: 1/(60+3)             ≈ 0.01587
    # w: 1/(60+3)             ≈ 0.01587
    assert ids[0] in {"x", "y"}
    assert ids[1] in {"x", "y"}
    assert set(ids[:2]) == {"x", "y"}
    assert set(ids[2:]) == {"w", "z"}


def test_rrf_fuse_missing_from_one_ranking():
    """An item in only one list still scores, but less than items in both."""
    a = [("only_a", 100.0)]
    b = [("only_b", 0.99), ("only_a", 0.0)]
    fused = dict(hr.rrf_fuse([a, b], k=60))
    # only_a: rank 1 in a + rank 2 in b = 1/61 + 1/62
    # only_b: rank 1 in b              = 1/61
    assert fused["only_a"] > fused["only_b"]


def test_rrf_fuse_empty_inputs():
    assert hr.rrf_fuse([]) == []
    assert hr.rrf_fuse([[], []]) == []


# ---------------------------------------------------------------------------
# Hybrid search — recall@10 microbenchmark
# ---------------------------------------------------------------------------


def _build_corpus(tmp_db, n: int = 100) -> list[str]:
    """Seed N memories. Returns the inserted IDs (m_000 … m_099)."""
    hr.ensure_fts_schema(tmp_db.conn)
    topics = [
        "python asyncio event loop",
        "rust ownership and borrow checker",
        "redis pub/sub patterns at scale",
        "postgres index types b-tree gin",
        "kubernetes pod scheduling affinity",
        "docker layer caching strategies",
        "graphql subscriptions websockets",
        "react hooks useEffect cleanup",
        "tailwind utility-first css",
        "typescript generics constraints",
    ]
    ids: list[str] = []
    for i in range(n):
        topic = topics[i % len(topics)]
        content = f"{topic} — note #{i} with some unique terms like xy{i:03d}"
        mid = f"m_{i:03d}"
        _seed(tmp_db, mid, content)
        ids.append(mid)
    return ids


def test_hybrid_search_latency_under_100ms_on_100_corpus(tmp_db):
    _build_corpus(tmp_db, n=100)
    # Stub dense_search so we don't depend on sentence-transformers loading.
    fake_dense = _make_fake_dense({
        "kubernetes scheduling": [(f"m_{i:03d}", 0.9 - i * 0.01) for i in (4, 14, 24)],
    })
    with patch.object(hr, "dense_search", side_effect=fake_dense):
        # Warm up (FTS5 caches, prepared statement, etc.)
        hr.hybrid_search(tmp_db, "kubernetes scheduling", top_k=10)
        t0 = time.perf_counter()
        results = hr.hybrid_search(tmp_db, "kubernetes scheduling", top_k=10)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert results, "hybrid_search returned nothing"
    assert elapsed_ms < 100.0, f"hybrid_search took {elapsed_ms:.1f} ms on 100-corpus"


def test_hybrid_beats_dense_on_lexical_query(tmp_db):
    """Microbenchmark: a query whose discriminating term is a rare token
    embedded in a noisy semantic neighborhood. Dense alone (mocked to
    confuse the unique tokens with its near neighbors) misses the
    correct answer at top-10; hybrid surfaces it via the BM25 channel.
    """
    _build_corpus(tmp_db, n=100)
    target_id = "m_037"
    # Inject the unique discriminator into m_037 only — BM25 will love it.
    tmp_db.conn.execute(
        "UPDATE memories SET content = ?, updated_at = ? WHERE id = ?",
        ("zorblax frobnicator distinctive token unique payload",
         utc_now(), target_id),
    )
    tmp_db.conn.commit()

    # Dense returns "near neighbors" that don't include the target —
    # simulating an embedding model that doesn't know the rare term.
    fake_dense_ranking = [
        (f"m_{i:03d}", 0.9 - i * 0.01) for i in range(0, 30) if f"m_{i:03d}" != target_id
    ]
    fake_dense = _make_fake_dense({"zorblax frobnicator": fake_dense_ranking})

    with patch.object(hr, "dense_search", side_effect=fake_dense):
        # Dense-only baseline: target NOT in top-10
        dense_only = hr.dense_search(tmp_db, "zorblax frobnicator", top_k=10)
        dense_top_ids = {mid for mid, _ in dense_only}
        assert target_id not in dense_top_ids

        # Hybrid: BM25 places target at rank 1, RRF carries it into top-10
        hybrid = hr.hybrid_search(tmp_db, "zorblax frobnicator", top_k=10)
        hybrid_ids = {h["id"] for h in hybrid}
        assert target_id in hybrid_ids, (
            f"hybrid lost the BM25 winner. got: {[h['id'] for h in hybrid]}"
        )
        # And it should rank well — within top-3
        ranked_ids = [h["id"] for h in hybrid]
        assert ranked_ids.index(target_id) < 3


def test_hybrid_search_handles_missing_dense_gracefully(tmp_db):
    """If dense_search raises (e.g. sqlite-vec missing), hybrid degrades to BM25."""
    _build_corpus(tmp_db, n=20)

    def _explode(*args, **kwargs):
        raise RuntimeError("simulated sqlite-vec missing")

    with patch.object(hr, "dense_search", side_effect=_explode):
        results = hr.hybrid_search(tmp_db, "kubernetes pod", top_k=5)
    # BM25 alone still produces results.
    assert results
    for r in results:
        assert r["bm25_rank"] is not None
        assert r["dense_rank"] is None


# ---------------------------------------------------------------------------
# assemble_context wiring
# ---------------------------------------------------------------------------


def test_assemble_context_default_mode_is_hybrid(tmp_db, monkeypatch):
    """Default `assemble_context` (no env, no arg) routes through hybrid_search."""
    from memoirs.engine import memory_engine as me

    monkeypatch.delenv("MEMOIRS_RETRIEVAL_MODE", raising=False)
    _build_corpus(tmp_db, n=10)

    fake_dense = _make_fake_dense({"python asyncio": [("m_000", 0.95)]})
    called = {"hybrid": 0, "dense": 0}

    real_hybrid = hr.hybrid_search

    def _spy_hybrid(*a, **kw):
        called["hybrid"] += 1
        return real_hybrid(*a, **kw)

    real_dense = hr.dense_search

    def _spy_dense(*a, **kw):
        called["dense"] += 1
        return fake_dense(*a, **kw)

    with patch.object(hr, "hybrid_search", side_effect=_spy_hybrid), \
         patch.object(hr, "dense_search", side_effect=_spy_dense):
        out = me.assemble_context(tmp_db, "python asyncio", top_k=5, max_lines=5)
    assert called["hybrid"] >= 1
    assert isinstance(out, dict)


def test_assemble_context_env_override_to_bm25(tmp_db, monkeypatch):
    from memoirs.engine import memory_engine as me

    monkeypatch.setenv("MEMOIRS_RETRIEVAL_MODE", "bm25")
    _build_corpus(tmp_db, n=10)
    out = me.assemble_context(tmp_db, "kubernetes pod", top_k=5, max_lines=5)
    # BM25-only path doesn't touch dense, but the result still has lines.
    assert "context" in out


def test_resolve_retrieval_mode_falls_back_on_unknown(tmp_db, monkeypatch):
    from memoirs.engine.memory_engine import _resolve_retrieval_mode
    monkeypatch.setenv("MEMOIRS_RETRIEVAL_MODE", "moonbeam")
    assert _resolve_retrieval_mode(None) == "hybrid"
    monkeypatch.delenv("MEMOIRS_RETRIEVAL_MODE")
    assert _resolve_retrieval_mode(None) == "hybrid"
    assert _resolve_retrieval_mode("dense") == "dense"
    assert _resolve_retrieval_mode("bm25") == "bm25"
