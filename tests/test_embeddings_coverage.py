"""Targeted coverage tests for memoirs/engine/embeddings.py.

These tests focus on branches, error paths, and edge cases that the broader
suite skips. Real sentence-transformers calls are avoided by injecting hand-
crafted unit-normalized vectors directly into the DB; we only exercise the
sentence-transformers loader via monkeypatching.
"""
from __future__ import annotations

import builtins
import math

import pytest

from memoirs.config import EMBEDDING_DIM
from memoirs.engine import embeddings as emb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(angle_rad: float) -> list[float]:
    v = [0.0] * EMBEDDING_DIM
    v[0] = math.cos(angle_rad)
    v[1] = math.sin(angle_rad)
    return v


def _seed_memory(db, *, memory_id: str, content: str = "x", mem_type: str = "fact",
                 angle_rad: float | None = None, dim_override: int | None = None) -> None:
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
        dim = dim_override if dim_override is not None else EMBEDDING_DIM
        db.conn.execute(
            "INSERT INTO memory_embeddings (memory_id, dim, embedding, model, created_at) "
            "VALUES (?, ?, ?, 'test', ?)",
            (memory_id, dim, blob, now),
        )
        # Mirror to vec_memories so ANN search can find this memory.
        if dim == EMBEDDING_DIM:
            db.conn.execute("DELETE FROM vec_memories WHERE memory_id = ?", (memory_id,))
            db.conn.execute(
                "INSERT INTO vec_memories(memory_id, embedding) VALUES (?, ?)",
                (memory_id, blob),
            )
    db.conn.commit()


# ---------------------------------------------------------------------------
# Pack / unpack — the BLOB roundtrip
# ---------------------------------------------------------------------------


def test_pack_unpack_roundtrip():
    original = [0.1, -0.2, 0.3, 0.4, 0.5]
    blob = emb._pack(original)
    out = emb._unpack(blob, dim=len(original))
    assert len(out) == len(original)
    for a, b in zip(original, out):
        assert abs(a - b) < 1e-6


def test_unpack_rejects_wrong_size():
    """The size check is a defensive branch when blob length doesn't match dim*4."""
    blob = emb._pack([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="embedding blob size"):
        emb._unpack(blob, dim=10)  # mismatched dim


# ---------------------------------------------------------------------------
# _require_embedder — error path when sentence-transformers is missing
# ---------------------------------------------------------------------------


def test_require_embedder_raises_when_missing(monkeypatch):
    """When sentence_transformers isn't importable, EmbeddingsUnavailable bubbles up."""
    # Reset singleton so we re-trigger the import path.
    monkeypatch.setattr(emb, "_MODEL_SINGLETON", None)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(emb.EmbeddingsUnavailable):
        emb._require_embedder()


def test_require_embedder_caches_singleton(monkeypatch):
    """Second call must return the cached model, not reload."""
    sentinel = object()
    monkeypatch.setattr(emb, "_MODEL_SINGLETON", sentinel)
    assert emb._require_embedder() is sentinel


# ---------------------------------------------------------------------------
# _require_vec — idempotency + the missing-package branch
# ---------------------------------------------------------------------------


def test_require_vec_idempotent(tmp_db):
    emb._require_vec(tmp_db)
    # Second call short-circuits via the _vec_loaded flag — no exception.
    emb._require_vec(tmp_db)
    assert getattr(tmp_db, "_vec_loaded", False) is True


def test_require_vec_raises_when_missing(monkeypatch, tmp_db):
    """When sqlite_vec is not importable, the helper raises EmbeddingsUnavailable."""
    # Force the import branch to run (otherwise short-circuited).
    if hasattr(tmp_db, "_vec_loaded"):
        tmp_db._vec_loaded = False
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sqlite_vec":
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(emb.EmbeddingsUnavailable):
        emb._require_vec(tmp_db)


# ---------------------------------------------------------------------------
# sync_vec_index — empty + dim-mismatch branches
# ---------------------------------------------------------------------------


def test_sync_vec_index_empty_db(tmp_db):
    """No rows => 0 inserts."""
    emb._require_vec(tmp_db)
    assert emb.sync_vec_index(tmp_db) == 0


def test_sync_vec_index_skips_dim_mismatch(tmp_db):
    """Rows with wrong dim should be ignored, not crash the sync."""
    emb._require_vec(tmp_db)
    # Seed with the correct vec0 row first then drop it from vec_memories so
    # sync_vec_index re-runs on an "orphan" row with mismatched dim.
    _seed_memory(tmp_db, memory_id="m1", angle_rad=0.0, dim_override=EMBEDDING_DIM + 1)
    tmp_db.conn.execute("DELETE FROM vec_memories WHERE memory_id = 'm1'")
    tmp_db.conn.commit()
    inserted = emb.sync_vec_index(tmp_db)
    assert inserted == 0
    # vec_memories still empty because dim didn't match
    n = tmp_db.conn.execute("SELECT COUNT(*) FROM vec_memories").fetchone()[0]
    assert n == 0


def test_sync_vec_index_inserts_orphans(tmp_db):
    """Rows in memory_embeddings missing from vec_memories should be mirrored."""
    emb._require_vec(tmp_db)
    _seed_memory(tmp_db, memory_id="m1", angle_rad=0.0)
    # Wipe vec_memories so the sync has work to do.
    tmp_db.conn.execute("DELETE FROM vec_memories WHERE memory_id = 'm1'")
    tmp_db.conn.commit()
    inserted = emb.sync_vec_index(tmp_db)
    assert inserted == 1
    n = tmp_db.conn.execute("SELECT COUNT(*) FROM vec_memories").fetchone()[0]
    assert n == 1


# ---------------------------------------------------------------------------
# search_similar_memories — both temporal branches + find_semantic_duplicates
# ---------------------------------------------------------------------------


def test_search_similar_memories_live_excludes_archived(tmp_db, monkeypatch):
    """Live mode (as_of=None) must skip archived rows."""
    emb._require_vec(tmp_db)
    _seed_memory(tmp_db, memory_id="m_active", content="alpha", angle_rad=0.0)
    _seed_memory(tmp_db, memory_id="m_archived", content="beta", angle_rad=0.0)
    tmp_db.conn.execute(
        "UPDATE memories SET archived_at = '2026-01-01T00:00:00+00:00' WHERE id = 'm_archived'"
    )
    tmp_db.conn.commit()
    # Stub embedder to avoid loading sentence-transformers.
    monkeypatch.setattr(emb, "embed_text", lambda text: _unit_vec(0.0))
    rows = emb.search_similar_memories(tmp_db, "anything", top_k=5)
    ids = {r["id"] for r in rows}
    assert "m_active" in ids
    assert "m_archived" not in ids


def test_search_similar_memories_as_of_includes_archived_if_then_valid(tmp_db, monkeypatch):
    """Time-travel: archived AFTER `as_of` should still surface."""
    emb._require_vec(tmp_db)
    _seed_memory(tmp_db, memory_id="m_old", content="alpha", angle_rad=0.0)
    # archived in 2027, query as of mid-2026 (after valid_from default)
    tmp_db.conn.execute(
        "UPDATE memories SET archived_at = '2027-06-01T00:00:00+00:00', "
        "valid_from = '2026-01-01T00:00:00+00:00' WHERE id = 'm_old'"
    )
    tmp_db.conn.commit()
    monkeypatch.setattr(emb, "embed_text", lambda text: _unit_vec(0.0))
    rows = emb.search_similar_memories(
        tmp_db, "anything", top_k=5, as_of="2027-01-01T00:00:00+00:00",
    )
    ids = {r["id"] for r in rows}
    assert "m_old" in ids
    # similarity field is added
    assert all("similarity" in r for r in rows)


def test_find_semantic_duplicates_threshold(tmp_db, monkeypatch):
    """Above threshold returns hits; raising the bar drops them all."""
    emb._require_vec(tmp_db)
    _seed_memory(tmp_db, memory_id="m_a", content="alpha", angle_rad=0.0)
    monkeypatch.setattr(emb, "embed_text", lambda text: _unit_vec(0.0))
    # cosine sim ~1.0 against itself; threshold 0.5 => match
    matches_low = emb.find_semantic_duplicates(tmp_db, "alpha", threshold=0.5)
    assert len(matches_low) >= 1
    # threshold 1.5 (impossible) => zero matches, no crash
    matches_high = emb.find_semantic_duplicates(tmp_db, "alpha", threshold=1.5)
    assert matches_high == []
