"""Tests for engine/mmr.py — P2-4 MMR diversification."""
from __future__ import annotations

import math

import pytest

from memoirs.engine import mmr


def _vec(x: float, y: float, dim: int = 8) -> list[float]:
    """Build a low-dim vector pinned to the (e0, e1) plane.

    Cosine between two such vectors equals ``cos(Δθ)`` after normalization,
    perfect for crafting "identical" / "orthogonal" pairs.
    """
    v = [0.0] * dim
    v[0] = x
    v[1] = y
    return v


# ---------------------------------------------------------------------------
# Activation gate
# ---------------------------------------------------------------------------


def test_is_enabled_default_on(monkeypatch):
    monkeypatch.delenv(mmr.ENV_MMR, raising=False)
    assert mmr.is_enabled() is True


def test_is_enabled_off(monkeypatch):
    monkeypatch.setenv(mmr.ENV_MMR, "off")
    assert mmr.is_enabled() is False


def test_get_lambda_default(monkeypatch):
    monkeypatch.delenv(mmr.ENV_LAMBDA, raising=False)
    assert mmr.get_lambda() == pytest.approx(0.7)


def test_get_lambda_invalid_uses_default(monkeypatch, caplog):
    monkeypatch.setenv(mmr.ENV_LAMBDA, "not-a-float")
    with caplog.at_level("WARNING", logger="memoirs.mmr"):
        assert mmr.get_lambda() == pytest.approx(mmr.DEFAULT_LAMBDA)


def test_get_lambda_clamps_to_unit_range(monkeypatch):
    monkeypatch.setenv(mmr.ENV_LAMBDA, "1.5")
    assert mmr.get_lambda() == 1.0
    monkeypatch.setenv(mmr.ENV_LAMBDA, "-0.3")
    assert mmr.get_lambda() == 0.0


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


def test_mmr_with_5_identical_diversifies():
    """5 candidates with identical embeddings — MMR should still pick all of
    them (k=3) but with the redundancy term punishing duplicates 2..N.

    What we really care about is: after the FIRST high-relevance pick,
    subsequent picks have rel - 1·redundancy = rel - 1 (with λ=0.5), so the
    selected order is determined by the original score order.
    """
    same = _vec(1.0, 0.0)
    candidates = [
        {"id": f"id{i}", "score": 1.0 - i * 0.1, "embedding": same}
        for i in range(5)
    ]
    out = mmr.mmr_select(candidates, k=3, lambda_=0.5,
                          embedding_lookup=lambda _id: None)
    assert len(out) == 3
    # Highest-score first (id0).
    assert out[0]["id"] == "id0"
    # All chosen ids must be unique.
    ids = [m["id"] for m in out]
    assert len(set(ids)) == 3


def test_mmr_with_orthogonal_candidates_preserves_relevance_order():
    """Orthogonal embeddings → redundancy term = 0 always → output equals
    input ordered by score, regardless of λ.
    """
    candidates = [
        {"id": "a", "score": 0.9, "embedding": _vec(1.0, 0.0)},
        {"id": "b", "score": 0.7, "embedding": _vec(0.0, 1.0)},
        # Different dims (e2, e3) so still orthogonal to first two.
        {"id": "c", "score": 0.5, "embedding": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        {"id": "d", "score": 0.3, "embedding": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]},
    ]
    out = mmr.mmr_select(candidates, k=3, lambda_=0.7,
                          embedding_lookup=lambda _id: None)
    assert [m["id"] for m in out] == ["a", "b", "c"]


def test_mmr_lambda_one_equals_top_k_by_score():
    candidates = [
        {"id": "a", "score": 0.9, "embedding": _vec(1.0, 0.0)},
        {"id": "b", "score": 0.7, "embedding": _vec(1.0, 0.0)},  # duplicate
        {"id": "c", "score": 0.5, "embedding": _vec(0.0, 1.0)},
    ]
    out = mmr.mmr_select(candidates, k=2, lambda_=1.0,
                          embedding_lookup=lambda _id: None)
    # Pure relevance — duplicate b wins over c despite redundancy.
    assert [m["id"] for m in out] == ["a", "b"]


def test_mmr_lambda_zero_maximizes_diversity():
    """λ=0 → ignore relevance, maximize negative max-similarity to chosen set.

    With 1 high-relevance + N near-duplicates + 1 unique outlier, λ=0 must
    pick the outlier as the second item.
    """
    candidates = [
        {"id": "a", "score": 0.9, "embedding": _vec(1.0, 0.0)},
        {"id": "b", "score": 0.85, "embedding": _vec(1.0, 0.0)},  # near-dup of a
        {"id": "c", "score": 0.10, "embedding": _vec(0.0, 1.0)},  # outlier
    ]
    out = mmr.mmr_select(candidates, k=2, lambda_=0.0,
                          embedding_lookup=lambda _id: None)
    ids = [m["id"] for m in out]
    # First pick: λ=0 means pure-diversity score = 0 for everything before
    # any pick → tie-break by original order → 'a'. Second pick: c is
    # orthogonal, b is duplicate.
    assert "c" in ids
    assert "b" not in ids


def test_mmr_k_greater_than_candidates_returns_all():
    candidates = [
        {"id": "a", "score": 0.9, "embedding": _vec(1.0, 0.0)},
        {"id": "b", "score": 0.5, "embedding": _vec(0.0, 1.0)},
    ]
    out = mmr.mmr_select(candidates, k=10, lambda_=0.7,
                          embedding_lookup=lambda _id: None)
    # Same objects, original order, no expensive loop.
    assert out == candidates


def test_mmr_k_zero_returns_empty():
    candidates = [{"id": "a", "score": 0.9, "embedding": _vec(1.0, 0.0)}]
    assert mmr.mmr_select(candidates, k=0) == []


def test_mmr_empty_input():
    assert mmr.mmr_select([], k=3) == []


def test_mmr_lookup_returns_none_does_not_crash():
    """Some candidate's embedding can't be loaded — MMR must still return
    a valid selection without raising.
    """
    candidates = [
        {"id": "a", "score": 0.9},  # no embedding inline
        {"id": "b", "score": 0.5, "embedding": _vec(0.0, 1.0)},
        {"id": "c", "score": 0.3, "embedding": _vec(1.0, 0.0)},
    ]

    def lookup(memory_id):
        if memory_id == "a":
            return None  # simulated DB miss
        return None  # everyone else already has inline

    out = mmr.mmr_select(candidates, k=3, lambda_=0.7, embedding_lookup=lookup)
    assert {m["id"] for m in out} == {"a", "b", "c"}
    # First pick is still the highest-score (relevance dominates).
    assert out[0]["id"] == "a"


def test_mmr_uses_inline_embedding_over_lookup():
    """Inline ``embedding`` must take precedence over ``embedding_lookup``."""
    inline = _vec(1.0, 0.0)
    candidates = [
        {"id": "a", "score": 0.9, "embedding": inline},
        {"id": "b", "score": 0.5, "embedding": _vec(0.0, 1.0)},
    ]

    calls: list[str] = []

    def lookup(mid):
        calls.append(mid)
        return _vec(0.5, 0.5)  # would change the result if used

    out = mmr.mmr_select(candidates, k=2, lambda_=0.5, embedding_lookup=lookup)
    # Lookup MUST NOT be called because both candidates have inline vectors.
    assert calls == []
    assert {m["id"] for m in out} == {"a", "b"}


def test_mmr_with_dataclass_candidate():
    """Candidates can be Candidate dataclasses, not just dicts."""
    candidates = [
        mmr.Candidate(memory_id="a", score=0.9, embedding=_vec(1.0, 0.0)),
        mmr.Candidate(memory_id="b", score=0.5, embedding=_vec(0.0, 1.0)),
    ]
    out = mmr.mmr_select(candidates, k=2, lambda_=0.5)
    assert [c.memory_id for c in out] == ["a", "b"]


def test_cosine_helper():
    assert mmr._cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert mmr._cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-9)
    # Anti-parallel.
    assert mmr._cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    # Zero-norm fallback.
    assert mmr._cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    # None-safe.
    assert mmr._cosine(None, [1.0]) == 0.0
