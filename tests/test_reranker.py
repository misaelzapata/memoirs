"""Tests for engine/reranker.py — P2-2 cross-encoder reranking.

We never invoke the real BGE model; CrossEncoder is monkey-patched with a
deterministic stub so the suite stays fast and offline.
"""
from __future__ import annotations

import pytest

from memoirs.engine import reranker as rk


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Clear the process-wide reranker singleton between tests."""
    rk.reset_reranker_singleton()
    yield
    rk.reset_reranker_singleton()


def _cands(*items):
    """Build candidate dicts. ``items`` = list of (id, content, score) tuples."""
    return [
        {"id": cid, "content": content, "score": float(score), "similarity": 0.0}
        for cid, content, score in items
    ]


# ---------------------------------------------------------------------------
# NoopReranker — preserves order and scores
# ---------------------------------------------------------------------------


def test_noop_reranker_preserves_order_and_scores():
    cands = _cands(("a", "alpha", 0.9), ("b", "beta", 0.5), ("c", "gamma", 0.1))
    pairs = rk.NoopReranker().rerank("anything", cands)
    assert pairs == [("a", 0.9), ("b", 0.5), ("c", 0.1)]


def test_get_reranker_default_is_noop(monkeypatch):
    monkeypatch.delenv(rk.ENV_BACKEND, raising=False)
    rk.reset_reranker_singleton()
    inst = rk.get_reranker()
    assert isinstance(inst, rk.NoopReranker)


def test_get_reranker_explicit_none_is_noop(monkeypatch):
    monkeypatch.setenv(rk.ENV_BACKEND, "none")
    rk.reset_reranker_singleton()
    assert isinstance(rk.get_reranker(), rk.NoopReranker)


# ---------------------------------------------------------------------------
# Reorder via stub: invert scores → reorder confirmed
# ---------------------------------------------------------------------------


class _InvertingReranker(rk.Reranker):
    """Test double: returns 1 - score so the order flips."""

    name = "invert"

    def rerank(self, query, candidates):
        out = []
        for c in candidates:
            cid = c.get("id") if isinstance(c, dict) else getattr(c, "id", None)
            score = float(c.get("score", 0.0)) if isinstance(c, dict) else float(getattr(c, "score", 0.0))
            out.append((cid, 1.0 - score))
        return out


def test_apply_rerank_reorders_with_inverting_stub():
    cands = _cands(("a", "x", 0.9), ("b", "y", 0.5), ("c", "z", 0.1))
    out = rk.apply_rerank("q", cands, reranker=_InvertingReranker())
    # Original ranking: a > b > c. Inverted should be c > b > a.
    assert [m["id"] for m in out] == ["c", "b", "a"]
    # rerank_score must be set
    assert all("rerank_score" in m for m in out)
    # The first candidate should have the highest rerank_score.
    assert out[0]["rerank_score"] >= out[-1]["rerank_score"]


# ---------------------------------------------------------------------------
# top_n caps how many candidates the cross-encoder sees
# ---------------------------------------------------------------------------


class _RecordingReranker(rk.Reranker):
    """Test double: records how many candidates it received."""

    name = "recording"

    def __init__(self):
        self.seen_counts: list[int] = []

    def rerank(self, query, candidates):
        self.seen_counts.append(len(candidates))
        # Identity — preserves scores so we don't accidentally reorder.
        out = []
        for c in candidates:
            cid = c["id"] if isinstance(c, dict) else c.id
            out.append((cid, float(c.get("score", 0.0))))
        return out


def test_apply_rerank_top_n_limits_input(monkeypatch):
    monkeypatch.setenv(rk.ENV_TOP_N, "10")
    cands = _cands(*[(f"id{i}", f"c{i}", 1.0 - i / 100.0) for i in range(25)])
    rec = _RecordingReranker()
    out = rk.apply_rerank("q", cands, reranker=rec)
    assert rec.seen_counts == [10]
    # All 25 still present in output.
    assert len(out) == 25
    # Tail (id10..id24) is preserved verbatim after the reranked head.
    assert [m["id"] for m in out[10:]] == [f"id{i}" for i in range(10, 25)]


def test_apply_rerank_explicit_top_n_override():
    cands = _cands(*[(f"id{i}", f"c{i}", 1.0 - i / 100.0) for i in range(20)])
    rec = _RecordingReranker()
    rk.apply_rerank("q", cands, reranker=rec, top_n=5)
    assert rec.seen_counts == [5]


# ---------------------------------------------------------------------------
# bge backend not available → fallback to Noop with warning
# ---------------------------------------------------------------------------


def test_bge_unavailable_falls_back_to_noop(monkeypatch, caplog):
    monkeypatch.setenv(rk.ENV_BACKEND, "bge")
    rk.reset_reranker_singleton()

    # Force is_available to return False to simulate missing dependency,
    # without disturbing the actual sentence-transformers install.
    real_is_avail = rk.BGERerankerLocal.is_available
    monkeypatch.setattr(
        rk.BGERerankerLocal, "is_available", lambda self: False
    )

    with caplog.at_level("WARNING", logger="memoirs.reranker"):
        inst = rk.get_reranker()
    assert isinstance(inst, rk.NoopReranker)
    assert any("falling back" in r.getMessage().lower() for r in caplog.records)

    # Restore for hygiene (autouse fixture also resets the singleton).
    monkeypatch.setattr(rk.BGERerankerLocal, "is_available", real_is_avail)


def test_bge_loaded_reranker_uses_predict(monkeypatch):
    """Stub the cross-encoder so we don't touch the network/model."""
    monkeypatch.setenv(rk.ENV_BACKEND, "bge")
    rk.reset_reranker_singleton()

    class _StubModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def predict(self, pairs):
            # Reverse: longer content → higher score.
            return [float(len(b)) for _a, b in pairs]

    # Pretend dep is installed and patch the actual loader.
    monkeypatch.setattr(rk.BGERerankerLocal, "is_available", lambda self: True)
    monkeypatch.setattr(
        rk.BGERerankerLocal, "_ensure_loaded", lambda self: _StubModel()
    )

    cands = _cands(("a", "short", 0.9), ("b", "much longer text", 0.1))
    inst = rk.get_reranker()
    assert isinstance(inst, rk.BGERerankerLocal)
    pairs = inst.rerank("q", cands)
    # Long text wins despite original score.
    pairs_by_id = dict(pairs)
    assert pairs_by_id["b"] > pairs_by_id["a"]


def test_unknown_backend_falls_back_to_none(monkeypatch, caplog):
    monkeypatch.setenv(rk.ENV_BACKEND, "rocketscience")
    rk.reset_reranker_singleton()
    with caplog.at_level("WARNING", logger="memoirs.reranker"):
        inst = rk.get_reranker()
    assert isinstance(inst, rk.NoopReranker)


def test_apply_rerank_short_circuits_on_noop():
    """Noop must not allocate or reorder — the input list is returned as-is."""
    cands = _cands(("a", "x", 0.5), ("b", "y", 0.4))
    out = rk.apply_rerank("q", cands, reranker=rk.NoopReranker())
    # Same object identity (fast path).
    assert out is cands


def test_apply_rerank_handles_empty_list():
    assert rk.apply_rerank("q", [], reranker=_InvertingReranker()) == []


def test_invalid_top_n_uses_default(monkeypatch, caplog):
    monkeypatch.setenv(rk.ENV_TOP_N, "not-a-number")
    cands = _cands(*[(f"id{i}", f"c{i}", 1.0 - i / 100.0) for i in range(60)])
    rec = _RecordingReranker()
    with caplog.at_level("WARNING", logger="memoirs.reranker"):
        rk.apply_rerank("q", cands, reranker=rec)
    # Default is 50.
    assert rec.seen_counts == [50]
