"""Integration tests for the 3-stage retrieval pipeline (P2-2 + P2-3 + P2-4).

These tests exercise ``assemble_context_stream`` end-to-end using a real
SQLite + sqlite-vec DB, but with hand-crafted unit-normalized vectors so
the suite stays deterministic and fast.
"""
from __future__ import annotations

import math

import pytest

from memoirs.config import EMBEDDING_DIM
from memoirs.engine import embeddings as emb
from memoirs.engine import hyde
from memoirs.engine import memory_engine as me
from memoirs.engine import mmr as _mmr
from memoirs.engine import reranker as _rk


def _unit_vec(angle_rad: float) -> list[float]:
    vec = [0.0] * EMBEDDING_DIM
    vec[0] = math.cos(angle_rad)
    vec[1] = math.sin(angle_rad)
    return vec


def _seed_memory(db, *, memory_id, content, mem_type="fact", angle_rad=None):
    now = "2026-04-27T00:00:00+00:00"
    db.conn.execute(
        "INSERT INTO memories (id, type, content, content_hash, importance, "
        "confidence, score, usage_count, user_signal, valid_from, "
        "metadata_json, created_at, updated_at) "
        "VALUES (?, ?, ?, 'h_'||?, 3, 0.5, 0.5, 0, 0, ?, '{}', ?, ?)",
        (memory_id, mem_type, content, memory_id, now, now, now),
    )
    if angle_rad is not None:
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


@pytest.fixture(autouse=True)
def _reset_singletons():
    _rk.reset_reranker_singleton()
    yield
    _rk.reset_reranker_singleton()


def _all_flags_off(monkeypatch):
    monkeypatch.delenv(hyde.ENV_HYDE, raising=False)
    monkeypatch.setenv(_rk.ENV_BACKEND, "none")
    monkeypatch.setenv(_mmr.ENV_MMR, "off")


# ---------------------------------------------------------------------------
# Regression: all flags off → pipeline behaves exactly like before
# ---------------------------------------------------------------------------


def test_pipeline_all_flags_off_is_no_regression(tmp_db, monkeypatch):
    _all_flags_off(monkeypatch)
    # Seed 3 memories with distinct embeddings.
    for i, angle in enumerate([0.0, math.pi / 2, math.pi]):
        _seed_memory(
            tmp_db,
            memory_id=f"mem_a{i}",
            content=f"alpha document {i}",
            angle_rad=angle,
        )

    # Force dense mode so retrieval is deterministic w.r.t. the angle setup.
    out = me.assemble_context(
        tmp_db, "anything", top_k=10, max_lines=5, retrieval_mode="dense"
    )
    assert isinstance(out, dict)
    assert "memories" in out
    assert len(out["memories"]) <= 5
    # Sanity: all returned ids are from our seeded set.
    ids = {m["id"] for m in out["memories"]}
    assert ids.issubset({"mem_a0", "mem_a1", "mem_a2"})


def test_assemble_context_stream_yields_meta_first(tmp_db, monkeypatch):
    _all_flags_off(monkeypatch)
    _seed_memory(tmp_db, memory_id="m1", content="hello world",
                 angle_rad=0.0)
    events = list(
        me.assemble_context_stream(
            tmp_db, "hello", top_k=5, max_lines=3, retrieval_mode="dense"
        )
    )
    tags = [e[0] for e in events]
    assert tags[0] == "meta"
    assert tags[-1] == "context"


# ---------------------------------------------------------------------------
# MMR reduces redundancy when on
# ---------------------------------------------------------------------------


def _avg_pairwise_cosine(vectors):
    n = len(vectors)
    if n < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _mmr._cosine(vectors[i], vectors[j])
            pairs += 1
    return total / pairs if pairs else 0.0


def test_mmr_reduces_redundancy(tmp_db, monkeypatch):
    """Seed many near-duplicate memories at angle 0 + a few orthogonal ones.
    With MMR off the top-K should be all near-duplicates; with MMR on the
    output should include at least one orthogonal memory and have lower
    average pairwise cosine.
    """
    monkeypatch.delenv(hyde.ENV_HYDE, raising=False)
    monkeypatch.setenv(_rk.ENV_BACKEND, "none")
    monkeypatch.setenv(_mmr.ENV_LAMBDA, "0.5")  # balanced

    # 6 near-duplicates clustered around angle 0.
    for i in range(6):
        _seed_memory(
            tmp_db,
            memory_id=f"dup_{i}",
            content=f"duplicate cluster {i}",
            angle_rad=0.01 * i,  # tiny angular spread
        )
    # 4 orthogonal docs at angle pi/2.
    for i in range(4):
        _seed_memory(
            tmp_db,
            memory_id=f"orth_{i}",
            content=f"orthogonal doc {i}",
            angle_rad=math.pi / 2 + 0.01 * i,
        )

    def _vectors_for(memories):
        out = []
        for m in memories:
            row = tmp_db.conn.execute(
                "SELECT embedding, dim FROM memory_embeddings WHERE memory_id = ?",
                (m["id"],),
            ).fetchone()
            out.append(emb._unpack(bytes(row["embedding"]), int(row["dim"])))
        return out

    # MMR off
    monkeypatch.setenv(_mmr.ENV_MMR, "off")
    out_off = me.assemble_context(
        tmp_db, "duplicate", top_k=10, max_lines=5, retrieval_mode="dense"
    )
    div_off = _avg_pairwise_cosine(_vectors_for(out_off["memories"]))

    # MMR on
    monkeypatch.setenv(_mmr.ENV_MMR, "on")
    out_on = me.assemble_context(
        tmp_db, "duplicate", top_k=10, max_lines=5, retrieval_mode="dense"
    )
    div_on = _avg_pairwise_cosine(_vectors_for(out_on["memories"]))

    # Diversity = 1 - avg pairwise cosine. We want it to GO UP when MMR is on.
    assert (1.0 - div_on) > (1.0 - div_off), (
        f"MMR did not improve diversity: off={1.0 - div_off:.3f} "
        f"on={1.0 - div_on:.3f}"
    )


# ---------------------------------------------------------------------------
# HyDE: query passed to _retrieve_candidates is the expanded one
# ---------------------------------------------------------------------------


def test_hyde_pipeline_passes_expanded_query(tmp_db, monkeypatch):
    monkeypatch.setenv(hyde.ENV_HYDE, "on")
    monkeypatch.setenv(_rk.ENV_BACKEND, "none")
    monkeypatch.setenv(_mmr.ENV_MMR, "off")

    # Stub HyDE expansion to a known string.
    def _fake_expand(query, backend="auto"):
        return hyde.ExpandedQuery(
            original=query,
            hypothetical_doc="HYPOTHETICAL_ANSWER_TOKEN",
            keywords=["kw1", "kw2"],
            combined=f"{query}. HYPOTHETICAL_ANSWER_TOKEN. kw1 kw2",
            backend="gemma",
        )

    monkeypatch.setattr(hyde, "expand_query", _fake_expand)

    # Spy on _retrieve_candidates to capture the query string actually used.
    captured: dict = {}
    real_retrieve = me._retrieve_candidates

    def _spy(db, q, *, top_k, as_of, mode, **kwargs):
        captured["query"] = q
        return real_retrieve(db, q, top_k=top_k, as_of=as_of, mode=mode)

    monkeypatch.setattr(me, "_retrieve_candidates", _spy)

    _seed_memory(tmp_db, memory_id="m1", content="anything", angle_rad=0.0)

    me.assemble_context(
        tmp_db, "user query", top_k=5, max_lines=3, retrieval_mode="dense"
    )
    assert "HYPOTHETICAL_ANSWER_TOKEN" in captured["query"]
    assert "user query" in captured["query"]


def test_hyde_off_passes_original_query_unchanged(tmp_db, monkeypatch):
    monkeypatch.delenv(hyde.ENV_HYDE, raising=False)
    monkeypatch.setenv(_rk.ENV_BACKEND, "none")
    monkeypatch.setenv(_mmr.ENV_MMR, "off")

    captured: dict = {}
    real_retrieve = me._retrieve_candidates

    def _spy(db, q, *, top_k, as_of, mode, **kwargs):
        captured["query"] = q
        return real_retrieve(db, q, top_k=top_k, as_of=as_of, mode=mode)

    monkeypatch.setattr(me, "_retrieve_candidates", _spy)

    _seed_memory(tmp_db, memory_id="m1", content="anything", angle_rad=0.0)
    me.assemble_context(
        tmp_db, "untouched query", top_k=5, max_lines=3, retrieval_mode="dense"
    )
    assert captured["query"] == "untouched query"


# ---------------------------------------------------------------------------
# Reranker: bge stub reorders the candidates
# ---------------------------------------------------------------------------


def test_pipeline_reranker_reorders_candidates(tmp_db, monkeypatch):
    monkeypatch.delenv(hyde.ENV_HYDE, raising=False)
    monkeypatch.setenv(_mmr.ENV_MMR, "off")
    monkeypatch.setenv(_rk.ENV_BACKEND, "bge")

    # 3 memories with distinct embeddings — dense retrieval orders them by
    # similarity to the query embedding. We don't care about the exact
    # ordering; we just want to prove the reranker can flip it.
    _seed_memory(tmp_db, memory_id="a", content="content A", angle_rad=0.0)
    _seed_memory(tmp_db, memory_id="b", content="content B",
                 angle_rad=math.pi / 6)
    _seed_memory(tmp_db, memory_id="c", content="content C",
                 angle_rad=math.pi / 3)

    # Stub the BGE model: assigns increasing scores in input order, so
    # whichever id was retrieved LAST (lowest dense similarity) becomes the
    # rerank winner.
    class _StubModel:
        def predict(self, pairs):
            # Use a wide enough range to dominate the similarity*0.6 term.
            return [float(i * 10) for i in range(len(pairs))]

    monkeypatch.setattr(_rk.BGERerankerLocal, "is_available", lambda self: True)
    monkeypatch.setattr(_rk.BGERerankerLocal, "_ensure_loaded", lambda self: _StubModel())

    _rk.reset_reranker_singleton()

    # Capture the dense order BEFORE the reranker runs.
    real_retrieve = me._retrieve_candidates
    pre_order: list[str] = []

    def _spy(db, q, *, top_k, as_of, mode, **kwargs):
        cands = real_retrieve(db, q, top_k=top_k, as_of=as_of, mode=mode)
        pre_order.extend(c["id"] for c in cands)
        return cands

    monkeypatch.setattr(me, "_retrieve_candidates", _spy)

    out = me.assemble_context(
        tmp_db, "content", top_k=10, max_lines=3, retrieval_mode="dense"
    )
    post_ids = [m["id"] for m in out["memories"]]
    # The reranker assigned the highest score to whichever candidate appeared
    # LAST in the dense output — verify that one is now first.
    assert pre_order[-1] == post_ids[0], (
        f"reranker did not flip ordering: pre={pre_order} post={post_ids}"
    )
    # And it is no longer the original top.
    assert pre_order[0] != post_ids[0]
