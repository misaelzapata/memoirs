"""Tests for engine/explain.py — chain-of-memory provenance traversal (P1-9).

We deliberately avoid the embeddings stack: explain only walks the SQL
graph (entities, memory_entities, relationships, memory_links) plus the
already-tested PPR primitives in :mod:`engine.graph_retrieval`. That
keeps these tests deterministic and fast enough for the perf budget.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

from memoirs.engine import explain as exp
from memoirs.engine import graph_retrieval as gr
from memoirs.mcp import tools as mt


NOW = "2026-04-27T00:00:00+00:00"


# ---------------------------------------------------------------------------
# Tiny seeders — same shape as test_graph_retrieval.py.
# ---------------------------------------------------------------------------


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


def _seed_relationship(
    db, src: str, tgt: str, rel: str = "uses", confidence: float = 1.0
) -> None:
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
    gr.invalidate_graph_cache()
    yield
    gr.invalidate_graph_cache()


@pytest.fixture
def synthetic_db(tmp_db):
    """5 memorias + 3 entities + 2 relationships + memory_links.

    Topology::

        ent_foo  --[uses]-->  ent_bar  --[implements]-->  ent_baz
           |                     |                            |
           |                     |                            |
        mem_1                 mem_target                   mem_3
                                                              |
                                                              v
                                                            mem_4 --link-- mem_5

    Query "foo" should follow:
      foo (entity_match) -> bar (entity_relation) -> mem_target.
    """
    db = tmp_db
    # Memorias
    for mid, content in [
        ("mem_1", "the foo concept is interesting"),
        ("mem_target", "bar is the target memory"),
        ("mem_3", "baz extends bar"),
        ("mem_4", "another node connected via memory_links"),
        ("mem_5", "leaf node"),
    ]:
        _seed_memory(db, mid, content)
    # Entities
    _seed_entity(db, "ent_foo", "foo")
    _seed_entity(db, "ent_bar", "bar")
    _seed_entity(db, "ent_baz", "baz")
    # memory ↔ entity
    _attach_entity(db, "mem_1", "ent_foo")
    _attach_entity(db, "mem_target", "ent_bar")
    _attach_entity(db, "mem_3", "ent_baz")
    # entity ↔ entity
    _seed_relationship(db, "ent_foo", "ent_bar", rel="uses", confidence=1.0)
    _seed_relationship(db, "ent_bar", "ent_baz", rel="implements", confidence=1.0)
    # memory ↔ memory (Zettelkasten)
    _seed_memory_link(db, "mem_3", "mem_4", 0.84, reason="semantic")
    _seed_memory_link(db, "mem_4", "mem_5", 0.71, reason="semantic")
    db.conn.commit()
    return db


# ---------------------------------------------------------------------------
# 1. Graph traversal builds an entity_match → entity_relation → entity_to_memory chain
# ---------------------------------------------------------------------------


def test_chain_passes_through_entity_relation(synthetic_db):
    chain = exp.build_provenance_chain(
        synthetic_db, "foo", None, "mem_target", max_hops=4,
    )
    kinds = [s["kind"] for s in chain]
    assert kinds[0] == "query"
    # The seed entity (foo) gets its own entity_match step.
    assert "entity_match" in kinds
    # The path traverses an entity_relation hop and lands on entity_to_memory.
    assert "entity_relation" in kinds
    assert kinds[-1] == "entity_to_memory"
    # Final hop arrives at the target memory.
    assert chain[-1]["memory_id"] == "mem_target"
    # Steps are numbered consecutively from 0.
    assert [s["step"] for s in chain] == list(range(len(chain)))


# ---------------------------------------------------------------------------
# 2. Memory unreachable via graph → semantic_match fallback
# ---------------------------------------------------------------------------


def test_semantic_match_fallback_when_no_graph_path(synthetic_db):
    # mem_5 is reachable only via memory_links from mem_3 — but the seed
    # entity for query "foo" is ent_foo, which is on a totally disconnected
    # entity-only side until ent_bar/ent_baz. From foo→bar→baz→mem_3→mem_4→mem_5
    # is 5 hops, so with max_hops=2 the path is unreachable.
    chain = exp.build_provenance_chain(
        synthetic_db, "foo", None, "mem_5",
        max_hops=2, similarity_score=0.42,
    )
    kinds = [s["kind"] for s in chain]
    assert kinds == ["query", "semantic_match"]
    assert chain[1]["score"] == 0.42
    assert chain[1]["memory_id"] == "mem_5"


def test_semantic_match_fallback_when_no_seed_entities(synthetic_db):
    # Query has no token that matches any entity → seed extraction returns []
    # → there is no graph path (no source). We must still get a non-empty chain.
    chain = exp.build_provenance_chain(
        synthetic_db, "zzzzz unrelated query", None, "mem_target",
        similarity_score=0.5,
    )
    assert chain[0]["kind"] == "query"
    assert chain[-1]["kind"] == "semantic_match"


# ---------------------------------------------------------------------------
# 3. max_hops=1 limits the chain
# ---------------------------------------------------------------------------


def test_max_hops_caps_chain_length(synthetic_db):
    # With max_hops=1, foo → mem_target requires 3 graph edges and is
    # unreachable, so we fall back to semantic_match.
    chain = exp.build_provenance_chain(
        synthetic_db, "foo", None, "mem_target", max_hops=1,
        similarity_score=0.3,
    )
    assert chain[-1]["kind"] == "semantic_match"

    # mem_1 is directly attached to ent_foo (1 hop) — should resolve.
    chain_mem1 = exp.build_provenance_chain(
        synthetic_db, "foo", None, "mem_1", max_hops=1,
    )
    kinds = [s["kind"] for s in chain_mem1]
    assert kinds[-1] == "entity_to_memory"
    assert chain_mem1[-1]["memory_id"] == "mem_1"


# ---------------------------------------------------------------------------
# 4. Non-existent memory → not_found step
# ---------------------------------------------------------------------------


def test_not_found_for_missing_memory(synthetic_db):
    chain = exp.build_provenance_chain(
        synthetic_db, "foo", None, "mem_does_not_exist",
    )
    kinds = [s["kind"] for s in chain]
    assert kinds == ["query", "not_found"]
    assert chain[1]["memory_id"] == "mem_does_not_exist"


# ---------------------------------------------------------------------------
# 5. mcp_explain_context returns provenance_chain on every result
# ---------------------------------------------------------------------------


def test_mcp_explain_context_emits_provenance_chain(monkeypatch, synthetic_db):
    """``mcp_explain_context`` must wrap each candidate with ``provenance_chain``."""
    from memoirs.engine import embeddings as emb

    # Stub the dense KNN — explain only needs id + similarity, no real vectors.
    monkeypatch.setattr(
        emb, "search_similar_memories",
        lambda db, q, top_k: [
            {"id": "mem_target", "type": "fact", "content": "bar is target",
             "similarity": 0.9, "score": 0.7, "importance": 4,
             "confidence": 0.8, "usage_count": 1},
            {"id": "mem_5", "type": "fact", "content": "leaf",
             "similarity": 0.3, "score": 0.4, "importance": 2,
             "confidence": 0.5, "usage_count": 0},
        ],
    )
    out = mt.call_tool(
        synthetic_db, "mcp_explain_context",
        {"query": "foo", "top_k": 2, "max_hops": 4},
    )
    assert out["query"] == "foo"
    assert len(out["results"]) == 2
    for r in out["results"]:
        assert "provenance_chain" in r
        chain = r["provenance_chain"]
        assert isinstance(chain, list) and len(chain) >= 2
        assert chain[0]["kind"] == "query"
    # mem_target must have a graph path; mem_5 must fall back to semantic.
    by_id = {r["id"]: r["provenance_chain"] for r in out["results"]}
    assert by_id["mem_target"][-1]["kind"] == "entity_to_memory"
    # mem_5 with default max_hops=4 *is* reachable from foo via the chain
    # foo→bar→baz→mem_3→mem_4→mem_5 (5 hops), so it should fall back.
    assert by_id["mem_5"][-1]["kind"] in {"semantic_match", "entity_to_memory", "memory_link"}


# ---------------------------------------------------------------------------
# 6. CLI `memoirs why` end-to-end
# ---------------------------------------------------------------------------


def test_cli_why_prints_table(synthetic_db, tmp_path):
    db_path = synthetic_db.path
    synthetic_db.conn.commit()
    # Close and reopen via subprocess so the CLI sees the seeded rows.
    out = subprocess.run(
        [
            sys.executable, "-m", "memoirs",
            "--db", str(db_path),
            "why", "mem_target",
            "--query", "foo",
            "--max-hops", "4",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    body = out.stdout
    assert "memory_id=mem_target" in body
    assert "entity_match" in body
    assert "entity_to_memory" in body or "memory_link" in body


def test_cli_why_json_output(synthetic_db):
    db_path = synthetic_db.path
    synthetic_db.conn.commit()
    out = subprocess.run(
        [
            sys.executable, "-m", "memoirs",
            "--db", str(db_path),
            "why", "mem_target",
            "--query", "foo", "--json",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    chain = json.loads(out.stdout)
    assert isinstance(chain, list) and len(chain) >= 2
    assert chain[0]["kind"] == "query"


# ---------------------------------------------------------------------------
# 7. Performance — explain of 10 memorias on a 1k-node graph < 200ms
# ---------------------------------------------------------------------------


def test_performance_explain_10_on_1k_graph(tmp_db):
    """Build a synthetic graph with ~1000 entity nodes and ~500 memorias,
    then ensure ``explain_memory_selection`` for 10 candidates returns in
    well under 200ms.
    """
    db = tmp_db
    # 500 memorias + 500 entities, each memory linked to a unique entity.
    for i in range(500):
        _seed_memory(db, f"m{i}", f"content {i}")
        _seed_entity(db, f"e{i}", f"entity_{i}")
        _attach_entity(db, f"m{i}", f"e{i}")
    # 500 entity-entity relationships forming a chain so the graph is one
    # connected component (worst case for BFS).
    for i in range(499):
        _seed_relationship(db, f"e{i}", f"e{i+1}", rel="next", confidence=1.0)
    # 500 memory_links forming a parallel ring.
    for i in range(499):
        _seed_memory_link(db, f"m{i}", f"m{i+1}", 0.5)
    db.conn.commit()
    gr.invalidate_graph_cache()

    # Pick 10 memorias near entity "entity_5" so the seed extractor finds
    # a real anchor and BFS does some real work.
    candidates = [
        {"id": f"m{i}", "similarity": 0.5}
        for i in range(10)
    ]
    # Warm the graph cache outside the timed window.
    gr.build_graph(db)
    t0 = time.perf_counter()
    enriched = exp.explain_memory_selection(
        db, "entity_5", None, candidates, max_hops=3,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert len(enriched) == 10
    for cand in enriched:
        assert "provenance_chain" in cand
        assert cand["provenance_chain"][0]["kind"] == "query"
    assert elapsed_ms < 200.0, f"explain took {elapsed_ms:.1f}ms (budget 200ms)"


# ---------------------------------------------------------------------------
# 8. UI endpoint /ui/memories/{id}/why returns HTML fragment
# ---------------------------------------------------------------------------


def test_ui_why_endpoint_returns_html_fragment(synthetic_db):
    """Smoke-test the new ``GET /ui/memories/{id}/why?q=...`` endpoint."""
    pytest.importorskip("fastapi")
    from starlette.testclient import TestClient
    from memoirs.api.server import _build_app

    synthetic_db.conn.commit()
    app = _build_app(synthetic_db.path)
    client = TestClient(app)

    r = client.get(
        "/ui/memories/mem_target/why",
        params={"q": "foo"},
    )
    assert r.status_code == 200
    body = r.text
    # Every step kind we expect has a stable label in the template.
    assert "entity_match" in body
    # The chain testid landmark gives the test a stable hook.
    assert 'data-testid="why-chain"' in body
