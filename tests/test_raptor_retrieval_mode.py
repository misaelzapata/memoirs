"""Tests for the RAPTOR retrieval mode integration in memory_engine.

Covers `_resolve_retrieval_mode` accepting `'raptor'` / `'hybrid_raptor'`,
the `_retrieve_candidates` dispatch into the new branches, and the
end-to-end `assemble_context` wrapper. Trees are built from synthetic
unit-normalized embeddings so the suite stays deterministic and fast.
"""
from __future__ import annotations

import math
import time

import pytest

from memoirs.config import EMBEDDING_DIM
from memoirs.engine import embeddings as emb
from memoirs.engine import memory_engine as me
from memoirs.engine import mmr as _mmr
from memoirs.engine import raptor as rp
from memoirs.engine import reranker as _rk


# ---------------------------------------------------------------------------
# Helpers (mirroring tests/test_raptor.py patterns)
# ---------------------------------------------------------------------------


def _unit_vec(angle_rad: float, *, dim: int = EMBEDDING_DIM) -> list[float]:
    vec = [0.0] * dim
    vec[0] = math.cos(angle_rad)
    vec[1] = math.sin(angle_rad)
    return vec


def _seed_memory(
    db,
    *,
    memory_id: str,
    content: str,
    angle_rad: float,
    mem_type: str = "fact",
) -> None:
    now = "2026-04-27T00:00:00+00:00"
    db.conn.execute(
        "INSERT INTO memories (id, type, content, content_hash, importance, "
        "confidence, score, usage_count, user_signal, valid_from, "
        "metadata_json, created_at, updated_at) "
        "VALUES (?, ?, ?, 'h_'||?, 3, 0.5, 0.5, 0, 0, ?, '{}', ?, ?)",
        (memory_id, mem_type, content, memory_id, now, now, now),
    )
    emb._require_vec(db)
    vec = _unit_vec(angle_rad)
    blob = emb._pack(vec)
    db.conn.execute(
        "INSERT INTO memory_embeddings (memory_id, dim, embedding, model, "
        "created_at) VALUES (?, ?, ?, 'test', ?)",
        (memory_id, EMBEDDING_DIM, blob, now),
    )
    db.conn.execute("DELETE FROM vec_memories WHERE memory_id = ?", (memory_id,))
    db.conn.execute(
        "INSERT INTO vec_memories(memory_id, embedding) VALUES (?, ?)",
        (memory_id, blob),
    )
    db.conn.commit()


def _seed_three_clusters(db, *, per_cluster: int = 10) -> dict[str, list[str]]:
    """3 well-separated clusters at angles 0°, 90°, 180°."""
    out: dict[str, list[str]] = {"c1": [], "c2": [], "c3": []}
    base_angles = {"c1": 0.0, "c2": math.pi / 2, "c3": math.pi}
    for label, base in base_angles.items():
        for i in range(per_cluster):
            jitter = (i - per_cluster / 2.0) * 0.005
            mid = f"mem_{label}_{i:03d}"
            content = f"{label} content keyword{label}{i}"
            _seed_memory(db, memory_id=mid, content=content, angle_rad=base + jitter)
            out[label].append(mid)
    return out


@pytest.fixture(autouse=True)
def _stub_query_embedding(monkeypatch):
    """Replace embed_text / embed_text_cached with a deterministic embedder.

    Maps cluster keywords to specific angles so the engine's RAPTOR branch
    can score memories against the query without real sentence-transformers.
    """

    def _fake(text: str) -> list[float]:
        t = (text or "").lower()
        if "c1" in t:
            return _unit_vec(0.0)
        if "c2" in t:
            return _unit_vec(math.pi / 2)
        if "c3" in t:
            return _unit_vec(math.pi)
        return _unit_vec(math.pi / 4)

    monkeypatch.setattr(emb, "embed_text", _fake)
    monkeypatch.setattr(emb, "embed_text_cached", _fake)
    yield


@pytest.fixture(autouse=True)
def _disable_pipeline_extras(monkeypatch):
    """Turn off HyDE, reranker, MMR, and Gemma so retrieval stays deterministic
    and the test doesn't accidentally load a multi-GB curator model.
    """
    from memoirs.engine import hyde as _hy
    from memoirs.engine import curator as _curator_mod
    monkeypatch.delenv(_hy.ENV_HYDE, raising=False)
    monkeypatch.setenv(_rk.ENV_BACKEND, "none")
    monkeypatch.setenv(_mmr.ENV_MMR, "off")
    # Force the conflict-resolution + curator paths off so assemble_context
    # never tries to load llama.cpp during retrieval-mode tests.
    monkeypatch.setattr(_curator_mod, "_have_curator", lambda: False)
    _rk.reset_reranker_singleton()
    yield
    _rk.reset_reranker_singleton()


# ---------------------------------------------------------------------------
# _resolve_retrieval_mode — accepts the new modes
# ---------------------------------------------------------------------------


def test_resolve_retrieval_mode_accepts_raptor(monkeypatch):
    monkeypatch.delenv("MEMOIRS_RETRIEVAL_MODE", raising=False)
    assert me._resolve_retrieval_mode("raptor") == "raptor"
    assert me._resolve_retrieval_mode("hybrid_raptor") == "hybrid_raptor"
    # Case-insensitive (mirrors how the env var path lowercases).
    assert me._resolve_retrieval_mode("RAPTOR") == "raptor"
    assert me._resolve_retrieval_mode("Hybrid_Raptor") == "hybrid_raptor"
    # Env var path.
    monkeypatch.setenv("MEMOIRS_RETRIEVAL_MODE", "raptor")
    assert me._resolve_retrieval_mode(None) == "raptor"
    monkeypatch.setenv("MEMOIRS_RETRIEVAL_MODE", "hybrid_raptor")
    assert me._resolve_retrieval_mode(None) == "hybrid_raptor"
    # Garbage still falls back to hybrid (existing contract).
    assert me._resolve_retrieval_mode("nonsense_mode") == "hybrid"


# ---------------------------------------------------------------------------
# Empty-tree degradation
# ---------------------------------------------------------------------------


def test_raptor_mode_returns_empty_when_tree_not_built(tmp_db, caplog):
    """No `memoirs raptor build` was ever run → mode='raptor' is a no-op, no raise."""
    rp.ensure_schema(tmp_db)
    _seed_memory(tmp_db, memory_id="m1", content="c1 hello", angle_rad=0.0)
    # Sanity: schema present but no nodes.
    assert tmp_db.conn.execute(
        "SELECT COUNT(*) AS c FROM summary_nodes"
    ).fetchone()["c"] == 0

    with caplog.at_level("WARNING"):
        out = me._retrieve_candidates(
            tmp_db, "c1 hello", top_k=10, as_of=None, mode="raptor",
        )
    assert out == []
    # Should have warned about the missing tree.
    assert any("no RAPTOR tree" in r.getMessage() for r in caplog.records)


def test_hybrid_raptor_falls_back_to_hybrid_when_tree_empty(tmp_db):
    """When no tree exists, hybrid_raptor must still surface hybrid hits."""
    _seed_memory(tmp_db, memory_id="m1", content="c1 alpha keyword", angle_rad=0.0)
    _seed_memory(tmp_db, memory_id="m2", content="c2 beta keyword", angle_rad=math.pi / 2)

    out = me._retrieve_candidates(
        tmp_db, "c1 alpha keyword", top_k=5, as_of=None, mode="hybrid_raptor",
    )
    # No RAPTOR tree → degrades to pure hybrid → still returns matches.
    assert len(out) >= 1
    ids = {r["id"] for r in out}
    assert "m1" in ids


# ---------------------------------------------------------------------------
# Built tree → retrieval works
# ---------------------------------------------------------------------------


def test_raptor_mode_returns_relevant_nodes_with_tree(tmp_db):
    """Build a tree, query close to cluster c1 → c1 leaves dominate."""
    rp.ensure_schema(tmp_db)
    seeded = _seed_three_clusters(tmp_db, per_cluster=10)
    rp.build_raptor_tree(tmp_db, k_per_cluster=10)

    out = me._retrieve_candidates(
        tmp_db, "c1 query", top_k=10, as_of=None, mode="raptor",
    )
    assert len(out) > 0
    c1_ids = set(seeded["c1"])
    hits = sum(1 for r in out if r["id"] in c1_ids)
    # Most of the top-10 should be c1 leaves (cluster is on-axis with the query).
    assert hits >= 5, f"only {hits}/10 c1 hits in {[r['id'] for r in out]}"


def test_hybrid_raptor_blends_hybrid_and_raptor(tmp_db):
    """hybrid_raptor returns rows from BOTH legs (RRF fusion)."""
    rp.ensure_schema(tmp_db)
    seeded = _seed_three_clusters(tmp_db, per_cluster=10)
    rp.build_raptor_tree(tmp_db, k_per_cluster=10)

    out = me._retrieve_candidates(
        tmp_db, "c1 keywordc10", top_k=10, as_of=None, mode="hybrid_raptor",
    )
    assert len(out) > 0
    # hybrid_raptor must surface c1 leaves (both BM25 token overlap on
    # 'keywordc10' and RAPTOR cosine alignment hit cluster c1).
    c1_ids = set(seeded["c1"])
    assert any(r["id"] in c1_ids for r in out)


# ---------------------------------------------------------------------------
# End-to-end: assemble_context wrapper
# ---------------------------------------------------------------------------


def test_assemble_context_with_raptor_mode_end_to_end(tmp_db):
    rp.ensure_schema(tmp_db)
    _seed_three_clusters(tmp_db, per_cluster=10)
    rp.build_raptor_tree(tmp_db, k_per_cluster=10)

    payload = me.assemble_context(
        tmp_db, "c1 query", top_k=10, max_lines=5, retrieval_mode="raptor",
    )
    assert isinstance(payload, dict)
    assert "memories" in payload
    assert len(payload["memories"]) <= 5
    # All ids should come from our seeded set.
    seeded_ids = {f"mem_c{c}_{i:03d}" for c in (1, 2, 3) for i in range(10)}
    returned_ids = {m["id"] for m in payload["memories"]}
    assert returned_ids.issubset(seeded_ids)


def test_assemble_context_raptor_mode_no_tree_does_not_raise(tmp_db):
    """assemble_context with mode='raptor' but no tree returns empty payload."""
    _seed_memory(tmp_db, memory_id="solo", content="c1 solo memory", angle_rad=0.0)
    payload = me.assemble_context(
        tmp_db, "c1 solo memory", top_k=10, max_lines=5, retrieval_mode="raptor",
    )
    assert isinstance(payload, dict)
    assert payload.get("memories", []) == []


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_hybrid_raptor_under_100ms_for_100_node_tree(tmp_db):
    """100 leaf memories + their tree → hybrid_raptor retrieval under 100ms.

    The retrieval call covers: BM25 + (skipped dense for trivial query) +
    RAPTOR scoring + RRF fusion + hydration. A budget of 100ms is plenty
    on commodity CI hardware once the tree is already built.
    """
    rp.ensure_schema(tmp_db)
    # Seed 100 memories across 4 clusters (25 each) at distinct angles.
    angles = [0.0, math.pi / 3, 2 * math.pi / 3, math.pi]
    for ci, base in enumerate(angles):
        for i in range(25):
            mid = f"mem_p{ci}_{i:03d}"
            jitter = (i - 12.5) * 0.002
            _seed_memory(
                tmp_db,
                memory_id=mid,
                content=f"perfcluster{ci} keyword{ci}_{i}",
                angle_rad=base + jitter,
            )
    rp.build_raptor_tree(tmp_db, k_per_cluster=25)
    # Warm any caches (FTS, vec_memories, …).
    me._retrieve_candidates(
        tmp_db, "perfcluster0 query", top_k=10, as_of=None, mode="hybrid_raptor",
    )
    start = time.perf_counter()
    out = me._retrieve_candidates(
        tmp_db, "perfcluster0 query", top_k=10, as_of=None, mode="hybrid_raptor",
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert len(out) > 0
    assert elapsed_ms < 100.0, f"hybrid_raptor took {elapsed_ms:.1f}ms (>100ms)"
