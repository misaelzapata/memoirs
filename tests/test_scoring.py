"""Tests for engine/memory_engine scoring formula."""
from datetime import datetime, timezone, timedelta

from memoirs.engine.memory_engine import (
    _normalize_importance,
    _normalize_usage,
    _recency_score,
    calculate_memory_score,
)


def test_normalize_importance_bounds():
    assert _normalize_importance(1) == 0.0
    assert _normalize_importance(5) == 1.0
    assert 0 < _normalize_importance(3) < 1


def test_normalize_importance_clamps():
    assert _normalize_importance(0) == 0.0
    assert _normalize_importance(10) == 1.0


def test_normalize_usage_zero():
    assert _normalize_usage(0) == 0.0


def test_normalize_usage_caps_at_one():
    assert _normalize_usage(1000) == _normalize_usage(1000)
    assert _normalize_usage(1000) <= 1.0


def test_recency_score_recent_high():
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Allow a sliver of decay for the microseconds between iso conversion + check
    assert _recency_score(None, now) >= 0.99


def test_recency_score_old_decays():
    old = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat(timespec="seconds")
    assert _recency_score(None, old) < 0.1  # >> 4 half-lives


def test_recency_score_handles_bad_input():
    assert 0 <= _recency_score(None, "not-a-date") <= 1


def test_calculate_memory_score_zero_memory():
    m = {"importance": 1, "confidence": 0.0, "usage_count": 0,
         "user_signal": 0.0, "created_at": "2020-01-01T00:00:00+00:00"}
    score = calculate_memory_score(m)
    assert 0 <= score < 0.1


def test_calculate_memory_score_max_memory():
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    m = {"importance": 5, "confidence": 1.0, "usage_count": 100,
         "user_signal": 1.0, "last_used_at": now, "created_at": now}
    score = calculate_memory_score(m)
    assert 0.9 <= score <= 1.0


def test_calculate_memory_score_weights_sum_correctly():
    """If all factors are 1.0, score should equal sum of weights = 1.0"""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # importance=5 (-> 1.0), confidence=1.0, recency=1.0 (just created),
    # usage cap, user_signal=1.0
    m = {"importance": 5, "confidence": 1.0, "usage_count": 10000,
         "user_signal": 1.0, "last_used_at": now, "created_at": now}
    assert calculate_memory_score(m) == 1.0 or calculate_memory_score(m) > 0.99
