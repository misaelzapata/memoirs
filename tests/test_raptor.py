"""Tests for engine/raptor.py — RAPTOR-style hierarchical summary tree (P1-6).

We bypass sentence-transformers + Gemma by injecting hand-crafted
unit-normalized vectors and using the heuristic / mock LLM path.
"""
from __future__ import annotations

import math
import sqlite3
import time
from pathlib import Path

import pytest

from memoirs import migrations
from memoirs.config import EMBEDDING_DIM
from memoirs.db import MemoirsDB
from memoirs.engine import embeddings as emb
from memoirs.engine import raptor as rp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(angle_rad: float, *, dim: int = EMBEDDING_DIM) -> list[float]:
    """384-dim unit vector in the (e0, e1) plane.

    Cosine similarity between two such vectors equals ``cos(Δangle)``.
    """
    vec = [0.0] * dim
    vec[0] = math.cos(angle_rad)
    vec[1] = math.sin(angle_rad)
    return vec


def _seed_memory(
    db: MemoirsDB,
    *,
    memory_id: str,
    content: str,
    angle_rad: float,
    mem_type: str = "fact",
) -> None:
    now = "2026-04-27T00:00:00+00:00"
    db.conn.execute(
        "INSERT INTO memories (id, type, content, content_hash, importance, "
        "confidence, score, usage_count, user_signal, valid_from, metadata_json, "
        "created_at, updated_at) "
        "VALUES (?, ?, ?, 'h_'||?, 3, 0.5, 0.5, 0, 0, ?, '{}', ?, ?)",
        (memory_id, mem_type, content, memory_id, now, now, now),
    )
    emb._require_vec(db)
    vec = _unit_vec(angle_rad)
    blob = emb._pack(vec)
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


def _seed_three_clusters(db: MemoirsDB, *, per_cluster: int = 10) -> dict[str, list[str]]:
    """Seed 3 well-separated clusters of `per_cluster` memories each."""
    out: dict[str, list[str]] = {"c1": [], "c2": [], "c3": []}
    base_angles = {"c1": 0.0, "c2": math.pi / 2, "c3": math.pi}  # 0°, 90°, 180°
    for label, base in base_angles.items():
        for i in range(per_cluster):
            jitter = (i - per_cluster / 2.0) * 0.005  # tight cluster
            mid = f"mem_{label}_{i:03d}"
            content = f"{label} content keyword{label}{i}"
            _seed_memory(db, memory_id=mid, content=content, angle_rad=base + jitter)
            out[label].append(mid)
    return out


# Replace embed_text in raptor module to use synthetic vectors during tests:
# raptor.summarize_cluster -> _persist_summary_node -> emb.embed_text(summary).
# We patch ``emb.embed_text`` to return the cluster centroid look-alike — a
# vector along the centroid's angle.


@pytest.fixture(autouse=True)
def _stub_embed_text(monkeypatch):
    """Replace embed_text with a deterministic synthetic embedder.

    Maps known cluster keywords to specific angles so summaries land near
    their cluster's leaves.
    """

    def _fake(text: str) -> list[float]:
        t = (text or "").lower()
        if "c1" in t:
            return _unit_vec(0.0)
        if "c2" in t:
            return _unit_vec(math.pi / 2)
        if "c3" in t:
            return _unit_vec(math.pi)
        # default: middle vector
        return _unit_vec(math.pi / 4)

    monkeypatch.setattr(emb, "embed_text", _fake)
    monkeypatch.setattr(emb, "embed_text_cached", _fake)
    yield


# ---------------------------------------------------------------------------
# Migration round-trip
# ---------------------------------------------------------------------------


def test_migration_006_round_trip(tmp_path):
    """Migration 006 up/down must be reversible without data loss."""
    db_path = tmp_path / "m6.sqlite"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        migrations.run_pending_migrations(conn)
        # Tables exist
        names = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "summary_nodes" in names
        assert "summary_node_members" in names
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

        # Roll back to before 006: tables dropped
        target = max(0, 5)
        # Find migration 006's predecessor:
        all_versions = [m.version for m in migrations.discover_migrations()]
        pre_006 = max([v for v in all_versions if v < 6], default=0)
        migrations.migrate_to(conn, pre_006)
        names_after = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "summary_nodes" not in names_after
        assert "summary_node_members" not in names_after

        # Re-apply: succeeds again (idempotent up)
        applied = migrations.run_pending_migrations(conn)
        assert 6 in applied
        names_back = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "summary_nodes" in names_back
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# cluster_memories
# ---------------------------------------------------------------------------


def test_cluster_memories_finds_three_clusters(tmp_db):
    """30 memories in 3 well-separated clusters → ~3 cluster groupings."""
    rp.ensure_schema(tmp_db)
    seeded = _seed_three_clusters(tmp_db, per_cluster=10)
    clusters = rp.cluster_memories(
        tmp_db, level=0, k_per_cluster=10
    )
    # K-means with k = n // k_per_cluster = 30 // 10 = 3 → exactly 3 clusters.
    # Greedy fallback may produce 3 too thanks to the tight separation.
    assert 2 <= len(clusters) <= 5
    # Each cluster should map predominantly to a single label.
    label_purity: list[float] = []
    for c in clusters:
        counter: dict[str, int] = {}
        for mem in c.members:
            for lbl, ids in seeded.items():
                if mem.node_id in ids:
                    counter[lbl] = counter.get(lbl, 0) + 1
                    break
        if counter:
            top = max(counter.values())
            label_purity.append(top / len(c.members))
    # Almost-perfect purity: each cluster contains members of mostly one label
    avg_purity = sum(label_purity) / len(label_purity)
    assert avg_purity >= 0.8, f"average purity={avg_purity:.2f}"


# ---------------------------------------------------------------------------
# summarize_cluster
# ---------------------------------------------------------------------------


def test_summarize_cluster_with_mock_llm():
    """When llm.create_completion is provided, its output is used verbatim."""

    class _MockLLM:
        # mimic the llama-cpp-python interface enough for raptor._gemma_summarize.
        def tokenize(self, data, *, add_bos=False, special=False):
            # ~1 token per char to keep budgets simple
            return [0] * len(data or b"")

        def detokenize(self, tokens):  # pragma: no cover -- unused in summary path
            return b""

        @property
        def n_ctx(self):
            return 4096

        def create_completion(self, prompt, **kwargs):
            return {"choices": [{"text": "MOCKED CLUSTER SUMMARY"}]}

    llm = _MockLLM()
    members = [
        rp.ClusterMember(node_id=f"m{i}", kind="memory",
                         embedding=_unit_vec(0.0), content=f"alpha {i}")
        for i in range(3)
    ]
    cluster = rp.Cluster(members=members, centroid=_unit_vec(0.0))
    summary = rp.summarize_cluster(cluster, llm=llm)
    assert "MOCKED" in summary


def test_summarize_cluster_heuristic_no_llm():
    """Without an LLM, summary is deterministic and includes member content."""
    members = [
        rp.ClusterMember(node_id=f"m{i}", kind="memory",
                         embedding=_unit_vec(0.0),
                         content=f"sqlite local memory engine {i}")
        for i in range(4)
    ]
    cluster = rp.Cluster(members=members, centroid=_unit_vec(0.0))
    summary1 = rp.summarize_cluster(cluster)
    summary2 = rp.summarize_cluster(cluster)
    assert summary1 == summary2  # determinism
    assert summary1 != ""
    # Top TF-IDF terms should make it into the [...] prefix.
    assert "sqlite" in summary1.lower() or "memory" in summary1.lower()


# ---------------------------------------------------------------------------
# build_raptor_tree
# ---------------------------------------------------------------------------


def test_build_raptor_tree_creates_multiple_levels(tmp_db):
    rp.ensure_schema(tmp_db)
    _seed_three_clusters(tmp_db, per_cluster=10)
    tree = rp.build_raptor_tree(
        tmp_db,
        scope_kind="global",
        scope_id=None,
        max_levels=4,
        k_per_cluster=10,
    )
    assert tree.leaf_count == 30
    assert tree.root_id is not None
    # At least one extra level beyond the leaves.
    assert len(tree.levels) >= 2
    levels_only = [lvl for lvl, _ in tree.levels]
    assert 0 in levels_only
    assert max(levels_only) >= 1


def test_build_raptor_tree_idempotent(tmp_db):
    """Running build twice without --rebuild does not duplicate nodes."""
    rp.ensure_schema(tmp_db)
    _seed_three_clusters(tmp_db, per_cluster=10)
    rp.build_raptor_tree(tmp_db, k_per_cluster=10)
    count_after_first = tmp_db.conn.execute(
        "SELECT COUNT(*) AS c FROM summary_nodes"
    ).fetchone()["c"]
    rp.build_raptor_tree(tmp_db, k_per_cluster=10)
    count_after_second = tmp_db.conn.execute(
        "SELECT COUNT(*) AS c FROM summary_nodes"
    ).fetchone()["c"]
    assert count_after_first == count_after_second
    # Now with rebuild=True nodes are wiped + recreated, so total should match
    # the first build (deterministic ids → same total).
    rp.build_raptor_tree(tmp_db, k_per_cluster=10, rebuild=True)
    count_after_rebuild = tmp_db.conn.execute(
        "SELECT COUNT(*) AS c FROM summary_nodes"
    ).fetchone()["c"]
    assert count_after_rebuild == count_after_first


# ---------------------------------------------------------------------------
# retrieve_raptor (tree descent)
# ---------------------------------------------------------------------------


def test_retrieve_raptor_returns_cluster_leaves(tmp_db):
    """Query close to cluster c1 → its 10 leaves dominate top-K."""
    rp.ensure_schema(tmp_db)
    seeded = _seed_three_clusters(tmp_db, per_cluster=10)
    rp.build_raptor_tree(tmp_db, k_per_cluster=10)
    # query angle 0 → cluster c1
    qvec = _unit_vec(0.0)
    results = rp.retrieve_raptor(
        tmp_db, qvec, top_k=10, prefer_high_level=False
    )
    assert len(results) == 10
    c1_ids = set(seeded["c1"])
    hits = sum(1 for (mid, _, _, _) in results if mid in c1_ids)
    # Pure RAPTOR descent: most of the top-10 should be c1 members.
    assert hits >= 8


def test_retrieve_raptor_path_includes_summary(tmp_db):
    """Each leaf result carries the chain of ancestor summary node ids."""
    rp.ensure_schema(tmp_db)
    _seed_three_clusters(tmp_db, per_cluster=10)
    rp.build_raptor_tree(tmp_db, k_per_cluster=10)
    qvec = _unit_vec(math.pi / 2)  # cluster c2
    results = rp.retrieve_raptor(tmp_db, qvec, top_k=5)
    # At least one path should include a summary id (sum_*) for the matched
    # cluster.
    paths = [path for (_, _, _, path) in results]
    assert any(any(p.startswith("sum_") for p in path) for path in paths)


# ---------------------------------------------------------------------------
# delete_subtree
# ---------------------------------------------------------------------------


def test_delete_subtree_cleans_rows(tmp_db):
    rp.ensure_schema(tmp_db)
    _seed_three_clusters(tmp_db, per_cluster=10)
    tree = rp.build_raptor_tree(tmp_db, k_per_cluster=10)
    assert tree.root_id is not None
    pre = tmp_db.conn.execute(
        "SELECT COUNT(*) AS c FROM summary_nodes"
    ).fetchone()["c"]
    deleted = rp.delete_subtree(tmp_db, tree.root_id)
    assert deleted >= 1
    post = tmp_db.conn.execute(
        "SELECT COUNT(*) AS c FROM summary_nodes"
    ).fetchone()["c"]
    assert post < pre
    # When the root anchored a chain ending in level-1 nodes, the subtree
    # rooted there should be wiped.
    rest_members = tmp_db.conn.execute(
        "SELECT COUNT(*) AS c FROM summary_node_members WHERE node_id = ?",
        (tree.root_id,),
    ).fetchone()["c"]
    assert rest_members == 0


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_build_raptor_tree_100_memories_under_5s(tmp_db):
    """100 synthetic memories build in <5s with mock-LLM (pure heuristic)."""
    rp.ensure_schema(tmp_db)
    # 4 clusters of 25 memories each.
    for c_idx, base in enumerate([0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]):
        for i in range(25):
            jitter = (i - 12.5) * 0.005
            _seed_memory(
                tmp_db,
                memory_id=f"perf_c{c_idx}_{i:03d}",
                content=f"cluster {c_idx} content {i}",
                angle_rad=base + jitter,
            )
    t0 = time.perf_counter()
    tree = rp.build_raptor_tree(tmp_db, k_per_cluster=10, max_levels=4)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"build took {elapsed:.2f}s (>5s budget)"
    assert tree.leaf_count == 100
    assert tree.root_id is not None


# ---------------------------------------------------------------------------
# Scope filtering
# ---------------------------------------------------------------------------


def test_build_raptor_tree_handles_empty_scope(tmp_db):
    """Building over an empty scope yields a SummaryTree with no root."""
    rp.ensure_schema(tmp_db)
    tree = rp.build_raptor_tree(tmp_db, scope_kind="global")
    assert tree.leaf_count == 0
    assert tree.root_id is None


def test_raptor_search_falls_back_when_no_embeddings(tmp_db, monkeypatch):
    """When embed_text raises EmbeddingsUnavailable, raptor_search returns []."""

    def _raise(*a, **k):
        raise emb.EmbeddingsUnavailable("test")

    monkeypatch.setattr(emb, "embed_text_cached", _raise)
    rp.ensure_schema(tmp_db)
    out = rp.raptor_search(tmp_db, "anything", top_k=5)
    assert out == []
