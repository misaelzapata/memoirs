"""Tests for engine/graph_retrieval.py — PPR multi-hop retrieval (P1-2).

We avoid the embeddings stack entirely: PPR only needs the SQL graph tables
(``entities``, ``memory_entities``, ``relationships``, ``memory_links``).
That keeps these tests deterministic and fast (<1s for the perf check).
"""
from __future__ import annotations

import time

import pytest

from memoirs.engine import graph_retrieval as gr


# ---------------------------------------------------------------------------
# Helpers — minimal seeders that bypass embeddings/extraction.
# ---------------------------------------------------------------------------

NOW = "2026-04-27T00:00:00+00:00"


def _seed_memory(db, memory_id: str, content: str, mem_type: str = "fact") -> None:
    db.conn.execute(
        "INSERT INTO memories (id, type, content, content_hash, importance, "
        "confidence, score, usage_count, user_signal, valid_from, metadata_json, "
        "created_at, updated_at) "
        "VALUES (?, ?, ?, 'h_'||?, 3, 0.5, 0.5, 0, 0, ?, '{}', ?, ?)",
        (memory_id, mem_type, content, memory_id, NOW, NOW, NOW),
    )


def _seed_entity(db, entity_id: str, name: str) -> None:
    db.conn.execute(
        "INSERT INTO entities (id, name, normalized_name, type, metadata_json, "
        "created_at, updated_at) VALUES (?, ?, ?, 'concept', '{}', ?, ?)",
        (entity_id, name, name.lower(), NOW, NOW),
    )


def _attach_entity(db, memory_id: str, entity_id: str) -> None:
    db.conn.execute(
        "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
        (memory_id, entity_id),
    )


def _seed_relationship(db, src: str, tgt: str, rel: str = "related_to", confidence: float = 1.0) -> None:
    db.conn.execute(
        "INSERT OR IGNORE INTO relationships (id, source_entity_id, target_entity_id, "
        "relation, confidence, metadata_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, '{}', ?)",
        (f"rel_{src}_{tgt}_{rel}", src, tgt, rel, confidence, NOW),
    )


def _seed_memory_link(db, src: str, tgt: str, sim: float, reason: str = "semantic") -> None:
    db.conn.execute(
        "INSERT OR IGNORE INTO memory_links (source_memory_id, target_memory_id, "
        "similarity, reason, created_at) VALUES (?, ?, ?, ?, ?)",
        (src, tgt, sim, reason, NOW),
    )


@pytest.fixture(autouse=True)
def _clear_graph_cache():
    """Reset the process-local graph cache between tests."""
    gr.invalidate_graph_cache()
    yield
    gr.invalidate_graph_cache()


# ---------------------------------------------------------------------------
# 1. Pure-PPR sanity check on a manual 5-node graph
# ---------------------------------------------------------------------------


def test_personalized_pagerank_decays_with_distance():
    """Build the graph A-B-C-D-E (chain). With seed=A, the score should be
    monotonically non-increasing as we walk away from A.

    A is the seed → highest score. B (1 hop) > C (2 hops) > D (3) > E (4).
    """
    raw = {
        "A": {"B": 1.0},
        "B": {"A": 1.0, "C": 1.0},
        "C": {"B": 1.0, "D": 1.0},
        "D": {"C": 1.0, "E": 1.0},
        "E": {"D": 1.0},
    }
    norm = gr._row_normalize(raw)
    view = gr.GraphView(adjacency=norm, raw_adjacency=raw, memory_count=0, entity_count=0)

    ranks = gr.personalized_pagerank(view, {"A": 1.0}, alpha=0.5, max_iter=100, tol=1e-8)

    # Seed has top score; further-away nodes have strictly less.
    assert ranks["A"] > ranks["B"] > ranks["C"], ranks
    assert ranks["C"] > ranks["D"] > ranks["E"], ranks
    # Sanity: total mass is roughly conserved (close to 1.0 with no danglers).
    assert 0.95 < sum(ranks.values()) < 1.05


def test_personalized_pagerank_distinguishes_one_vs_two_hop_neighbors():
    """Star-shaped graph: A connects to B,C; D is only connected through C.

    With seed=A, B and C must outrank D (which sits at 2 hops).
    """
    raw = {
        "A": {"B": 1.0, "C": 1.0},
        "B": {"A": 1.0},
        "C": {"A": 1.0, "D": 1.0},
        "D": {"C": 1.0},
    }
    norm = gr._row_normalize(raw)
    view = gr.GraphView(adjacency=norm, raw_adjacency=raw, memory_count=0, entity_count=0)
    ranks = gr.personalized_pagerank(view, {"A": 1.0}, alpha=0.5)
    assert ranks["A"] > ranks["B"]
    assert ranks["A"] > ranks["C"]
    assert ranks["B"] > ranks["D"]
    assert ranks["C"] > ranks["D"]


# ---------------------------------------------------------------------------
# 2. Seed extraction
# ---------------------------------------------------------------------------


def test_extract_seed_entities_matches_substring(tmp_db):
    _seed_entity(tmp_db, "ent_memoirs", "memoirs")
    _seed_entity(tmp_db, "ent_pagerank", "PageRank")
    _seed_entity(tmp_db, "ent_unrelated", "Banana")
    tmp_db.conn.commit()

    seeds = gr.extract_seed_entities(tmp_db, "How does memoirs implement PageRank?")
    assert "ent_memoirs" in seeds
    assert "ent_pagerank" in seeds
    assert "ent_unrelated" not in seeds


def test_extract_seed_entities_empty_query(tmp_db):
    _seed_entity(tmp_db, "ent_x", "anything")
    tmp_db.conn.commit()
    assert gr.extract_seed_entities(tmp_db, "") == []
    assert gr.extract_seed_entities(tmp_db, "   \n  ") == []


# ---------------------------------------------------------------------------
# 3. graph_search end-to-end
# ---------------------------------------------------------------------------


def test_graph_search_returns_top_k_memories(tmp_db):
    """Three memories, three entities. Memory M1 directly links to entity E_PPR;
    M2 links to E_GRAPH which is a peer of E_PPR via a relationship; M3 is on
    an island (entity E_OTHER, no path to E_PPR).

    Query = "PageRank"  → seed = E_PPR.
      - M1 should be top (1 hop: M1↔E_PPR).
      - M2 should follow (M2↔E_GRAPH↔E_PPR, 2 hops).
      - M3 should be absent or last.
    """
    _seed_entity(tmp_db, "E_PPR", "PageRank")
    _seed_entity(tmp_db, "E_GRAPH", "graph")
    _seed_entity(tmp_db, "E_OTHER", "weather")
    _seed_memory(tmp_db, "m1", "PageRank is implemented")
    _seed_memory(tmp_db, "m2", "graph traversal explained")
    _seed_memory(tmp_db, "m3", "weather report from yesterday")
    _attach_entity(tmp_db, "m1", "E_PPR")
    _attach_entity(tmp_db, "m2", "E_GRAPH")
    _attach_entity(tmp_db, "m3", "E_OTHER")
    _seed_relationship(tmp_db, "E_GRAPH", "E_PPR", "applies_to")
    _seed_relationship(tmp_db, "E_PPR", "E_GRAPH", "applies_to")
    tmp_db.conn.commit()

    results = gr.graph_search(tmp_db, "PageRank", top_k=10)
    assert results, "expected at least one memory"
    ids_ranked = [mid for mid, _ in results]
    assert ids_ranked[0] == "m1", f"m1 should rank first, got {ids_ranked}"
    assert "m2" in ids_ranked
    if "m3" in ids_ranked:
        assert ids_ranked.index("m3") > ids_ranked.index("m2")
    # All scores are positive and descending.
    scores = [s for _, s in results]
    assert all(s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_graph_search_empty_query_returns_empty_list(tmp_db):
    _seed_entity(tmp_db, "E1", "PageRank")
    _seed_memory(tmp_db, "m1", "PageRank stuff")
    _attach_entity(tmp_db, "m1", "E1")
    tmp_db.conn.commit()
    assert gr.graph_search(tmp_db, "") == []
    # And a query with no matching entities also yields empty.
    assert gr.graph_search(tmp_db, "completely-unrelated-xyz") == []


# ---------------------------------------------------------------------------
# 4. hybrid_graph_search blends both
# ---------------------------------------------------------------------------


def test_hybrid_graph_search_improves_recall_over_either_alone(tmp_db, monkeypatch):
    """Hand-crafted scenario: hybrid (BM25+dense) finds a memory the graph
    missed, the graph finds a memory hybrid missed. RRF fusion should expose
    BOTH in the top-K — strictly higher recall than either alone.
    """
    # Build a tiny graph where the seed entity has one direct memory.
    _seed_entity(tmp_db, "E1", "memoirs")
    _seed_memory(tmp_db, "graph_only", "internal note about memoirs internals")
    _seed_memory(tmp_db, "hybrid_only", "another note about memoirs")
    _attach_entity(tmp_db, "graph_only", "E1")
    # Note: hybrid_only has NO entity attachment → graph won't reach it.
    tmp_db.conn.commit()

    # Mock hybrid_search to return only `hybrid_only` — emulates the
    # surface-form match that graph would miss.
    def fake_hybrid_search(db, query, *, top_k, as_of=None, **_):
        return [{"id": "hybrid_only", "score": 0.9, "bm25_rank": 1,
                 "dense_rank": None, "bm25_score": 0.9, "dense_score": None}]

    from memoirs.engine import hybrid_retrieval as hr
    monkeypatch.setattr(hr, "hybrid_search", fake_hybrid_search)

    fused = gr.hybrid_graph_search(tmp_db, "memoirs", top_k=10)
    fused_ids = {mid for mid, _ in fused}

    # Recall@10:
    graph_ids = {mid for mid, _ in gr.graph_search(tmp_db, "memoirs", top_k=10)}
    hybrid_ids = {"hybrid_only"}
    expected = graph_ids | hybrid_ids

    assert fused_ids >= expected, f"fusion missed something: {fused_ids} vs {expected}"
    assert len(fused_ids) > len(graph_ids)
    assert len(fused_ids) > len(hybrid_ids)
    assert "graph_only" in fused_ids
    assert "hybrid_only" in fused_ids


# ---------------------------------------------------------------------------
# 5. Cache behavior
# ---------------------------------------------------------------------------


def test_build_graph_caches_between_calls(tmp_db, monkeypatch):
    """Second call with same DB + same memory count must NOT re-query."""
    _seed_entity(tmp_db, "E1", "alpha")
    _seed_memory(tmp_db, "m1", "alpha")
    _attach_entity(tmp_db, "m1", "E1")
    tmp_db.conn.commit()

    # First call: cold — should populate the cache.
    v1 = gr.build_graph(tmp_db)
    # Spy on the uncached builder; if the cache works, it must NOT be called.
    calls = {"n": 0}
    real = gr._build_graph_uncached

    def spy(db):
        calls["n"] += 1
        return real(db)

    monkeypatch.setattr(gr, "_build_graph_uncached", spy)

    v2 = gr.build_graph(tmp_db)
    assert calls["n"] == 0, "cache miss: _build_graph_uncached was called"
    # And the returned object is literally the cached one.
    assert v2 is v1


def test_build_graph_invalidates_when_memory_count_changes(tmp_db):
    """Adding a new active memory bumps the count → cache invalidates."""
    _seed_entity(tmp_db, "E1", "alpha")
    _seed_memory(tmp_db, "m1", "alpha")
    _attach_entity(tmp_db, "m1", "E1")
    tmp_db.conn.commit()

    v1 = gr.build_graph(tmp_db)
    assert v1.memory_count == 1

    _seed_memory(tmp_db, "m2", "beta")
    _seed_entity(tmp_db, "E2", "beta")
    _attach_entity(tmp_db, "m2", "E2")
    tmp_db.conn.commit()

    v2 = gr.build_graph(tmp_db)
    assert v2.memory_count == 2
    assert v2 is not v1


# ---------------------------------------------------------------------------
# 6. Performance: 1000 memories + 200 entities under 100ms (warm cache)
# ---------------------------------------------------------------------------


def test_graph_search_perf_warm_cache(tmp_db):
    """Build a 1000-memory + 200-entity graph and time graph_search after the
    cache has been warmed. Target: <100ms median per query.
    """
    NMEM, NENT = 1000, 200

    with tmp_db.conn:
        for i in range(NENT):
            _seed_entity(tmp_db, f"E{i}", f"entity{i}")
        for i in range(NMEM):
            _seed_memory(tmp_db, f"m{i}", f"memory text {i}")
            # Each memory touches ~3 entities (round-robin).
            for k in range(3):
                _attach_entity(tmp_db, f"m{i}", f"E{(i + k) % NENT}")
        # A few random relationships for entity↔entity edges.
        for i in range(0, NENT, 2):
            _seed_relationship(tmp_db, f"E{i}", f"E{(i + 1) % NENT}")
        # Some memory_links to make it spicier.
        for i in range(0, NMEM, 50):
            _seed_memory_link(tmp_db, f"m{i}", f"m{(i + 1) % NMEM}", 0.7)

    # Warm cache.
    t0 = time.perf_counter()
    gr.build_graph(tmp_db)
    cold_build_ms = (time.perf_counter() - t0) * 1000.0
    # Force the cache lookup path on subsequent build_graph calls.

    # Run several queries and take the median.
    samples_ms = []
    for q in ("entity1", "entity50", "entity99", "entity150", "entity199"):
        t0 = time.perf_counter()
        results = gr.graph_search(tmp_db, q, top_k=10)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
        assert isinstance(results, list)

    samples_ms.sort()
    median_ms = samples_ms[len(samples_ms) // 2]
    # Print so the harness shows the actual numbers.
    print(
        f"\n[graph perf] cold_build={cold_build_ms:.1f}ms  "
        f"warm_query_ms={samples_ms} median={median_ms:.1f}ms  "
        f"nodes={gr.build_graph(tmp_db).num_nodes}"
    )
    # Generous budget: 100ms median for the warm path.
    assert median_ms < 100.0, f"graph_search median={median_ms:.1f}ms > 100ms"


# ---------------------------------------------------------------------------
# 7. Multi-hop scoring sanity (real DB, not bare graph)
# ---------------------------------------------------------------------------


def test_graph_search_multi_hop_via_memory_link(tmp_db):
    """Two memories: M1 carries the seed entity; M2 is reachable only via a
    memory_link from M1 (no shared entity). PPR should still surface M2 — the
    Zettelkasten edges feed back into the random walk.
    """
    _seed_entity(tmp_db, "E_PPR", "PageRank")
    _seed_memory(tmp_db, "m1", "PageRank notes")
    _seed_memory(tmp_db, "m2", "graph theory notes")
    _attach_entity(tmp_db, "m1", "E_PPR")
    _seed_memory_link(tmp_db, "m1", "m2", 0.8)
    _seed_memory_link(tmp_db, "m2", "m1", 0.8)
    tmp_db.conn.commit()

    results = gr.graph_search(tmp_db, "PageRank", top_k=5)
    ids_ranked = [mid for mid, _ in results]
    assert "m1" in ids_ranked
    # m2 must appear thanks to the memory_link bridge.
    assert "m2" in ids_ranked, f"multi-hop reach failed: {ids_ranked}"
    # And m1 still beats m2 (closer to the seed).
    assert ids_ranked.index("m1") < ids_ranked.index("m2")


# ---------------------------------------------------------------------------
# 8. Mode dispatch through memory_engine
# ---------------------------------------------------------------------------


def test_resolve_retrieval_mode_accepts_graph_modes():
    from memoirs.engine import memory_engine as me
    assert me._resolve_retrieval_mode("graph") == "graph"
    assert me._resolve_retrieval_mode("hybrid_graph") == "hybrid_graph"
    assert me._resolve_retrieval_mode("HYBRID_GRAPH") == "hybrid_graph"
    # Unknown modes still fall back to hybrid (existing behavior preserved).
    assert me._resolve_retrieval_mode("does-not-exist") == "hybrid"


def test_retrieve_candidates_routes_to_graph(tmp_db, monkeypatch):
    """``_retrieve_candidates(mode='graph')`` must call into graph_retrieval,
    not the hybrid path. We verify by spy.
    """
    _seed_entity(tmp_db, "E_PPR", "PageRank")
    _seed_memory(tmp_db, "m1", "PageRank stuff")
    _attach_entity(tmp_db, "m1", "E_PPR")
    tmp_db.conn.commit()

    from memoirs.engine import memory_engine as me

    called = {"graph": 0, "hybrid_graph": 0}

    def fake_graph_search(db, query, *, top_k, alpha=0.5):
        called["graph"] += 1
        return [("m1", 0.9)]

    def fake_hybrid_graph_search(db, query, *, top_k, as_of=None):
        called["hybrid_graph"] += 1
        return [("m1", 0.9)]

    from memoirs.engine import graph_retrieval as gr2
    monkeypatch.setattr(gr2, "graph_search", fake_graph_search)
    monkeypatch.setattr(gr2, "hybrid_graph_search", fake_hybrid_graph_search)

    r1 = me._retrieve_candidates(tmp_db, "PageRank", top_k=5, as_of=None, mode="graph")
    r2 = me._retrieve_candidates(tmp_db, "PageRank", top_k=5, as_of=None, mode="hybrid_graph")

    assert called["graph"] == 1
    assert called["hybrid_graph"] == 1
    assert r1 and r1[0]["id"] == "m1"
    assert r2 and r2[0]["id"] == "m1"
    # Hydrated rows expose the PPR/RRF score in the `similarity` slot.
    assert r1[0]["similarity"] > 0
    assert r2[0]["similarity"] > 0
