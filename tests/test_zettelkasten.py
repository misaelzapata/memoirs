"""Tests for engine/zettelkasten.py — A-MEM Zettelkasten linking (P1-3).

We bypass sentence-transformers by injecting hand-crafted unit-normalized
vectors directly into ``memory_embeddings`` + ``vec_memories``. This keeps
the suite fast (<2s) and deterministic, while still exercising the real
sqlite-vec ANN code path.
"""
from __future__ import annotations

import math
import os
import time

import pytest

from memoirs.config import EMBEDDING_DIM
from memoirs.engine import embeddings as emb
from memoirs.engine import zettelkasten as zk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(angle_rad: float) -> list[float]:
    """384-dim unit vector that lives in the (e0, e1) plane.

    Cosine similarity between two such vectors equals ``cos(Δangle)`` —
    perfect for crafting pairs at known similarities for testing.
    """
    vec = [0.0] * EMBEDDING_DIM
    vec[0] = math.cos(angle_rad)
    vec[1] = math.sin(angle_rad)
    return vec


def _seed_memory(db, *, memory_id: str, content: str, mem_type: str = "fact",
                 angle_rad: float | None = None) -> None:
    """Insert a memory + (optional) pre-baked embedding."""
    now = "2026-04-27T00:00:00+00:00"
    db.conn.execute(
        "INSERT INTO memories (id, type, content, content_hash, importance, "
        "confidence, score, usage_count, user_signal, valid_from, metadata_json, "
        "created_at, updated_at) "
        "VALUES (?, ?, ?, 'h_'||?, 3, 0.5, 0.5, 0, 0, ?, '{}', ?, ?)",
        (memory_id, mem_type, content, memory_id, now, now, now),
    )
    if angle_rad is not None:
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


def _seed_entity(db, *, entity_id: str, name: str) -> None:
    now = "2026-04-27T00:00:00+00:00"
    db.conn.execute(
        "INSERT INTO entities (id, name, normalized_name, type, metadata_json, "
        "created_at, updated_at) VALUES (?, ?, ?, 'concept', '{}', ?, ?)",
        (entity_id, name, name.lower(), now, now),
    )
    db.conn.commit()


def _attach_entity(db, memory_id: str, entity_id: str) -> None:
    db.conn.execute(
        "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
        (memory_id, entity_id),
    )
    db.conn.commit()


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------


def test_ensure_schema_creates_table(tmp_db):
    zk.ensure_schema(tmp_db.conn)
    rows = tmp_db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_links'"
    ).fetchall()
    assert len(rows) == 1


def test_ensure_schema_idempotent(tmp_db):
    zk.ensure_schema(tmp_db.conn)
    zk.ensure_schema(tmp_db.conn)  # second call must not raise
    rows = tmp_db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND name IN ('idx_memory_links_source','idx_memory_links_target')"
    ).fetchall()
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# link_memory: bidirectional + threshold
# ---------------------------------------------------------------------------


def test_link_memory_creates_bidirectional(tmp_db):
    """A→B link must also write B→A so neighbor lookups work both ways."""
    zk.ensure_schema(tmp_db.conn)
    _seed_memory(tmp_db, memory_id="m_a", content="alpha", angle_rad=0.0)
    _seed_memory(tmp_db, memory_id="m_b", content="beta",  angle_rad=0.1)  # cos≈0.995
    _seed_memory(tmp_db, memory_id="m_c", content="gamma", angle_rad=0.2)  # cos≈0.980

    links = zk.link_memory(tmp_db, "m_a", top_k=5, threshold=0.5)
    assert len(links) == 2

    forward = tmp_db.conn.execute(
        "SELECT target_memory_id FROM memory_links WHERE source_memory_id='m_a' AND reason='semantic'"
    ).fetchall()
    backward = tmp_db.conn.execute(
        "SELECT source_memory_id FROM memory_links WHERE target_memory_id='m_a' AND reason='semantic'"
    ).fetchall()
    forward_ids = {r["target_memory_id"] for r in forward}
    backward_ids = {r["source_memory_id"] for r in backward}
    assert forward_ids == {"m_b", "m_c"}
    assert backward_ids == {"m_b", "m_c"}


def test_link_memory_threshold_filters(tmp_db):
    """Under mode='absolute', vectors below the cosine threshold must NOT be linked."""
    zk.ensure_schema(tmp_db.conn)
    _seed_memory(tmp_db, memory_id="m_a", content="alpha", angle_rad=0.0)
    _seed_memory(tmp_db, memory_id="m_b", content="beta",  angle_rad=0.05)   # cos≈0.999 (very close)
    _seed_memory(tmp_db, memory_id="m_far", content="far", angle_rad=math.pi / 2)  # cos=0 (orthogonal)

    # The fixed-threshold gate is mode='absolute'. The new default 'topk'
    # intentionally ignores threshold (see test_zettelkasten_threshold.py).
    links = zk.link_memory(tmp_db, "m_a", top_k=5, threshold=0.9, mode="absolute")
    # m_far at cos=0 must be excluded; only m_b is similar enough.
    target_ids = {l.target_memory_id for l in links}
    assert "m_b" in target_ids
    assert "m_far" not in target_ids


def test_link_memory_idempotent(tmp_db):
    """Re-linking twice must not create duplicate rows (UNIQUE constraint)."""
    zk.ensure_schema(tmp_db.conn)
    _seed_memory(tmp_db, memory_id="m_a", content="alpha", angle_rad=0.0)
    _seed_memory(tmp_db, memory_id="m_b", content="beta",  angle_rad=0.1)

    zk.link_memory(tmp_db, "m_a", top_k=5, threshold=0.5)
    first_count = tmp_db.conn.execute(
        "SELECT COUNT(*) AS c FROM memory_links WHERE reason='semantic'"
    ).fetchone()["c"]

    zk.link_memory(tmp_db, "m_a", top_k=5, threshold=0.5)
    second_count = tmp_db.conn.execute(
        "SELECT COUNT(*) AS c FROM memory_links WHERE reason='semantic'"
    ).fetchone()["c"]
    assert first_count == second_count == 2  # m_a→m_b, m_b→m_a


def test_link_memory_skips_archived(tmp_db):
    """Archived memorias must not appear as link targets."""
    zk.ensure_schema(tmp_db.conn)
    _seed_memory(tmp_db, memory_id="m_a", content="alpha", angle_rad=0.0)
    _seed_memory(tmp_db, memory_id="m_b", content="beta",  angle_rad=0.05)
    _seed_memory(tmp_db, memory_id="m_dead", content="dead", angle_rad=0.06)
    tmp_db.conn.execute(
        "UPDATE memories SET archived_at='2026-04-27T00:00:00+00:00' WHERE id='m_dead'"
    )
    tmp_db.conn.commit()

    links = zk.link_memory(tmp_db, "m_a", top_k=5, threshold=0.5)
    target_ids = {l.target_memory_id for l in links}
    assert "m_dead" not in target_ids


# ---------------------------------------------------------------------------
# link_by_shared_entities
# ---------------------------------------------------------------------------


def test_link_by_shared_entities_creates_link(tmp_db):
    """Memorias sharing an entity get a `shared_entity` link with non-zero similarity."""
    zk.ensure_schema(tmp_db.conn)
    _seed_memory(tmp_db, memory_id="m_a", content="alpha")
    _seed_memory(tmp_db, memory_id="m_b", content="beta")
    _seed_entity(tmp_db, entity_id="e_x", name="X")
    _attach_entity(tmp_db, "m_a", "e_x")
    _attach_entity(tmp_db, "m_b", "e_x")

    links = zk.link_by_shared_entities(tmp_db, "m_a")
    assert len(links) == 1
    assert links[0].target_memory_id == "m_b"
    assert links[0].reason == "shared_entity"
    assert links[0].similarity > 0  # Jaccard = 1/1 = 1.0 here


def test_link_by_shared_entities_no_match(tmp_db):
    """Memorias with disjoint entity sets must produce no links."""
    zk.ensure_schema(tmp_db.conn)
    _seed_memory(tmp_db, memory_id="m_a", content="alpha")
    _seed_memory(tmp_db, memory_id="m_b", content="beta")
    _seed_entity(tmp_db, entity_id="e_x", name="X")
    _seed_entity(tmp_db, entity_id="e_y", name="Y")
    _attach_entity(tmp_db, "m_a", "e_x")
    _attach_entity(tmp_db, "m_b", "e_y")

    links = zk.link_by_shared_entities(tmp_db, "m_a")
    assert links == []


# ---------------------------------------------------------------------------
# get_neighbors traversal
# ---------------------------------------------------------------------------


def test_get_neighbors_depth_1(tmp_db):
    """Direct neighbors of a memory at depth=1."""
    zk.ensure_schema(tmp_db.conn)
    _seed_memory(tmp_db, memory_id="m_a", content="alpha")
    _seed_memory(tmp_db, memory_id="m_b", content="beta")
    # Manually wire one bidirectional link.
    tmp_db.conn.execute(
        "INSERT INTO memory_links (source_memory_id, target_memory_id, similarity, reason) "
        "VALUES ('m_a','m_b',0.9,'semantic'),('m_b','m_a',0.9,'semantic')"
    )
    tmp_db.conn.commit()

    nbrs = zk.get_neighbors(tmp_db, "m_a", max_depth=1, min_similarity=0.5)
    assert len(nbrs) == 1
    assert nbrs[0].memory_id == "m_b"
    assert nbrs[0].depth == 1


def test_get_neighbors_depth_2_traverses_transitively(tmp_db):
    """A→B→C must surface C at depth=2 even when A→C isn't an edge."""
    zk.ensure_schema(tmp_db.conn)
    _seed_memory(tmp_db, memory_id="m_a", content="alpha")
    _seed_memory(tmp_db, memory_id="m_b", content="beta")
    _seed_memory(tmp_db, memory_id="m_c", content="gamma")
    tmp_db.conn.executemany(
        "INSERT INTO memory_links (source_memory_id, target_memory_id, similarity, reason) "
        "VALUES (?, ?, ?, 'semantic')",
        [
            ("m_a", "m_b", 0.9),
            ("m_b", "m_a", 0.9),
            ("m_b", "m_c", 0.8),
            ("m_c", "m_b", 0.8),
        ],
    )
    tmp_db.conn.commit()

    nbrs = zk.get_neighbors(tmp_db, "m_a", max_depth=2, min_similarity=0.5)
    by_id = {n.memory_id: n for n in nbrs}
    assert "m_b" in by_id and by_id["m_b"].depth == 1
    assert "m_c" in by_id and by_id["m_c"].depth == 2


def test_get_neighbors_min_similarity_prunes(tmp_db):
    """Edges below the threshold must be excluded from traversal."""
    zk.ensure_schema(tmp_db.conn)
    _seed_memory(tmp_db, memory_id="m_a", content="alpha")
    _seed_memory(tmp_db, memory_id="m_b", content="beta")
    tmp_db.conn.execute(
        "INSERT INTO memory_links (source_memory_id, target_memory_id, similarity, reason) "
        "VALUES ('m_a','m_b',0.3,'semantic'),('m_b','m_a',0.3,'semantic')"
    )
    tmp_db.conn.commit()

    nbrs = zk.get_neighbors(tmp_db, "m_a", max_depth=1, min_similarity=0.5)
    assert nbrs == []


# ---------------------------------------------------------------------------
# Integration / performance
# ---------------------------------------------------------------------------


def test_link_memory_no_embedding_returns_empty(tmp_db):
    """Memory without a stored embedding should fall back gracefully.

    The fallback path tries to encode via sentence-transformers; if that's
    not loadable in the test environment it returns []. Either way it must
    not raise.
    """
    zk.ensure_schema(tmp_db.conn)
    _seed_memory(tmp_db, memory_id="m_lonely", content="orphan", angle_rad=None)
    # No embedding row exists. Without an embedder loaded, link_memory must
    # not crash; with one loaded it would compute on-the-fly.
    try:
        links = zk.link_memory(tmp_db, "m_lonely", top_k=5, threshold=0.5)
    except Exception as e:
        pytest.fail(f"link_memory raised on missing embedding: {e}")
    # Either way, with no other memorias around, no links.
    assert isinstance(links, list)


def test_recompute_links_processes_all(tmp_db):
    """Backfill walks the whole corpus and produces neighbor links."""
    zk.ensure_schema(tmp_db.conn)
    # 5 memorias arranged on a tight arc → all near each other
    for i in range(5):
        _seed_memory(
            tmp_db, memory_id=f"m_{i}", content=f"item-{i}", angle_rad=i * 0.05,
        )
    result = zk.recompute_links(
        tmp_db, batch_size=2, top_k=3, threshold=0.5,
        include_shared_entities=False,
    )
    assert result["processed"] == 5
    # Every memory should have at least one neighbor in this dense cluster.
    for i in range(5):
        rows = tmp_db.conn.execute(
            "SELECT COUNT(*) AS c FROM memory_links WHERE source_memory_id=?",
            (f"m_{i}",),
        ).fetchone()
        assert rows["c"] >= 1


def test_link_memory_perf_under_50ms_for_1000(tmp_db):
    """Acceptance: link_memory must stay under ~50ms with 1k memorias.

    This is a smoke test, not a strict bound — CI noise can push it above
    50ms. We check < 250ms to leave headroom while still catching O(n²)
    regressions.
    """
    zk.ensure_schema(tmp_db.conn)
    # Seed 1k memorias spread around a circle so vec0 has a real workload.
    for i in range(1000):
        angle = (i / 1000) * 2 * math.pi
        _seed_memory(tmp_db, memory_id=f"p_{i:04d}", content=f"x{i}", angle_rad=angle)

    target = "p_0500"
    start = time.perf_counter()
    zk.link_memory(tmp_db, target, top_k=5, threshold=0.0)
    elapsed_ms = (time.perf_counter() - start) * 1000
    # If running on a slow CI machine, allow up to 250ms.
    assert elapsed_ms < 250, f"link_memory took {elapsed_ms:.1f}ms (expected <250ms)"


# ---------------------------------------------------------------------------
# Env-var toggle
# ---------------------------------------------------------------------------


def test_zettelkasten_env_off_disables(monkeypatch):
    monkeypatch.setenv("MEMOIRS_ZETTELKASTEN", "off")
    assert zk._is_enabled() is False
    monkeypatch.setenv("MEMOIRS_ZETTELKASTEN", "on")
    assert zk._is_enabled() is True
    monkeypatch.delenv("MEMOIRS_ZETTELKASTEN", raising=False)
    assert zk._is_enabled() is True  # default on
