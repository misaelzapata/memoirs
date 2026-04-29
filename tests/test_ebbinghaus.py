"""Tests for the Ebbinghaus forgetting-curve recency scorer (P1-7).

Covers:
* `ebbinghaus_recency` shape: 1.0 fresh, ~e^-1 after 24h with S=1, slower
  decay with S=2, clamp to [0.01, 1.0].
* `record_access` mutates ``last_accessed_at`` and grows ``strength`` by 1.5
  per call, capped at 64.0.
* Memory whose ``last_accessed_at`` is NULL falls back to ``created_at``.
* ROI sanity check: fresh memory scores >2× the same memory accessed 7d ago.
* End-to-end hook: ``assemble_context`` updates ``last_accessed_at`` on
  every returned memory.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from memoirs.core.ids import utc_now
from memoirs.engine.memory_engine import (
    _STRENGTH_GROWTH,
    _STRENGTH_MAX,
    assemble_context,
    calculate_memory_score,
    ebbinghaus_recency,
    record_access,
)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Pure function: ebbinghaus_recency
# ---------------------------------------------------------------------------


def test_ebbinghaus_recency_just_accessed_is_one():
    """Memory accessed at `now` should score ≈ 1.0 (no decay).

    `_iso` truncates to seconds, so reusing the rounded timestamp as both
    the access marker and `now` keeps Δt at exactly 0.
    """
    now_iso = _iso(datetime.now(timezone.utc))
    now = datetime.fromisoformat(now_iso)
    score = ebbinghaus_recency(now_iso, strength=1.0, now=now)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_ebbinghaus_recency_24h_strength_1_is_e_minus_1():
    """Δt=24h, S=1 → exp(-1) ≈ 0.3679."""
    now = datetime.now(timezone.utc)
    last = now - timedelta(hours=24)
    score = ebbinghaus_recency(_iso(last), strength=1.0, now=now)
    assert score == pytest.approx(math.exp(-1), abs=1e-3)


def test_ebbinghaus_recency_24h_strength_2_decays_slower():
    """Δt=24h, S=2 → exp(-0.5) ≈ 0.6065 (consolidation slows decay)."""
    now = datetime.now(timezone.utc)
    last = now - timedelta(hours=24)
    score = ebbinghaus_recency(_iso(last), strength=2.0, now=now)
    assert score == pytest.approx(math.exp(-0.5), abs=1e-3)
    # Sanity: stronger memory > weaker memory at the same Δt.
    weak = ebbinghaus_recency(_iso(last), strength=1.0, now=now)
    assert score > weak


def test_ebbinghaus_recency_clamps_floor_at_001():
    """Ancient memory → clamp to 0.01, never below."""
    now = datetime.now(timezone.utc)
    last = now - timedelta(days=10_000)
    score = ebbinghaus_recency(_iso(last), strength=1.0, now=now)
    assert score == pytest.approx(0.01, abs=1e-6)
    assert score >= 0.01


def test_ebbinghaus_recency_handles_garbage_timestamp():
    """Unparseable timestamp falls back to a neutral 0.5 — never crashes."""
    score = ebbinghaus_recency("not-a-date", strength=1.0)
    assert 0.0 <= score <= 1.0


def test_ebbinghaus_recency_none_last_accessed_fresh():
    """``last_accessed_at`` of None is treated as fresh (caller picks the
    fallback timestamp)."""
    assert ebbinghaus_recency(None, strength=1.0) == 1.0


# ---------------------------------------------------------------------------
# DB-bound: record_access
# ---------------------------------------------------------------------------


def _seed_memory(db, *, mid="mem_test_1", strength=1.0, last_accessed=None):
    now_iso = utc_now()
    db.conn.execute(
        """
        INSERT INTO memories (
            id, type, content, content_hash, importance, confidence,
            score, usage_count, user_signal, valid_from, metadata_json,
            created_at, updated_at, strength, last_accessed_at
        ) VALUES (?, 'fact', ?, ?, 3, 0.8, 0, 0, 0, ?, '{}', ?, ?, ?, ?)
        """,
        (
            mid,
            f"content for {mid}",
            f"hash_{mid}",
            now_iso,
            now_iso,
            now_iso,
            strength,
            last_accessed,
        ),
    )
    db.conn.commit()
    return mid


def test_record_access_updates_timestamp_and_strength(tmp_db):
    mid = _seed_memory(tmp_db, strength=1.0, last_accessed=None)
    fixed_now = datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc)
    record_access(tmp_db, mid, now=fixed_now)
    tmp_db.conn.commit()
    row = tmp_db.conn.execute(
        "SELECT last_accessed_at, strength FROM memories WHERE id = ?", (mid,)
    ).fetchone()
    assert row["last_accessed_at"] == _iso(fixed_now)
    assert row["strength"] == pytest.approx(1.5, rel=1e-6)


def test_record_access_strength_capped_at_max(tmp_db):
    """Hammering record_access shouldn't overflow strength past 64.0."""
    mid = _seed_memory(tmp_db, strength=50.0)
    # 50 * 1.5 = 75 > 64 → cap to 64 on the very first call.
    record_access(tmp_db, mid)
    tmp_db.conn.commit()
    s1 = tmp_db.conn.execute(
        "SELECT strength FROM memories WHERE id = ?", (mid,)
    ).fetchone()["strength"]
    assert s1 == pytest.approx(_STRENGTH_MAX, rel=1e-6)
    # Repeated calls stay capped.
    for _ in range(5):
        record_access(tmp_db, mid)
    tmp_db.conn.commit()
    s2 = tmp_db.conn.execute(
        "SELECT strength FROM memories WHERE id = ?", (mid,)
    ).fetchone()["strength"]
    assert s2 == pytest.approx(_STRENGTH_MAX, rel=1e-6)


def test_record_access_growth_constant():
    """Sanity: the growth constant matches MemoryBank (×1.5)."""
    assert _STRENGTH_GROWTH == 1.5


# ---------------------------------------------------------------------------
# Score integration
# ---------------------------------------------------------------------------


def test_calculate_score_uses_created_at_when_last_accessed_missing():
    """Memory with neither last_accessed_at nor last_used_at falls back to
    created_at — should still score reasonably for a freshly-created row.
    """
    now = utc_now()
    m = {
        "importance": 4,
        "confidence": 0.9,
        "usage_count": 0,
        "user_signal": 0.0,
        "created_at": now,
        # no last_accessed_at, no last_used_at, no strength
    }
    score = calculate_memory_score(m)
    # Recency for a row created right now is ≈ 1.0; importance=0.75,
    # confidence=0.9 → score should be well above 0.4.
    assert score > 0.4


def test_score_recent_vs_week_old_more_than_double_recency():
    """ROI sanity: identical memories — one accessed today, one a week ago
    — and the recency contribution alone differs by >2×.

    With S=1: today → recency≈1.0; 7 days ago → exp(-7) ≈ 0.0009 (clamped
    to 0.01). 1.0 / 0.01 = 100×, well above the 2× target.
    """
    now = datetime.now(timezone.utc)
    fresh_iso = _iso(now)
    old_iso = _iso(now - timedelta(days=7))
    fresh = ebbinghaus_recency(fresh_iso, strength=1.0, now=now)
    old = ebbinghaus_recency(old_iso, strength=1.0, now=now)
    assert fresh / old > 2.0
    # And on the composite score:
    base = {
        "importance": 3,
        "confidence": 0.8,
        "usage_count": 1,
        "user_signal": 0.0,
        "strength": 1.0,
    }
    fresh_score = calculate_memory_score({**base, "last_accessed_at": fresh_iso, "created_at": fresh_iso})
    old_score = calculate_memory_score({**base, "last_accessed_at": old_iso, "created_at": old_iso})
    assert fresh_score > old_score
    # The 0.15 weight on recency means the ratio of TOTAL scores is smaller
    # than the recency ratio, but the diff must still be visibly positive.
    assert (fresh_score - old_score) > 0.05


# ---------------------------------------------------------------------------
# End-to-end retrieval hook
# ---------------------------------------------------------------------------


def test_assemble_context_updates_last_accessed_at(tmp_db):
    """After a live retrieval, every returned memory should have
    ``last_accessed_at`` bumped and ``strength`` multiplied.
    """
    # Seed a memory whose last_accessed_at is far in the past.
    old_iso = _iso(datetime(2024, 1, 1, tzinfo=timezone.utc))
    mid = _seed_memory(
        tmp_db,
        mid="mem_e2e_hook",
        strength=1.0,
        last_accessed=old_iso,
    )
    # Update the content to something queryable.
    tmp_db.conn.execute(
        "UPDATE memories SET content = ? WHERE id = ?",
        ("the user prefers minimalist UI design", mid),
    )
    tmp_db.conn.commit()
    # Refresh FTS index so BM25 finds it.
    tmp_db.conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (mid,))
    tmp_db.conn.execute(
        "INSERT INTO memories_fts(memory_id, content) VALUES (?, ?)",
        (mid, "the user prefers minimalist UI design"),
    )
    tmp_db.conn.commit()

    result = assemble_context(tmp_db, "minimalist UI design", top_k=5, max_lines=5)
    returned_ids = {m["id"] for m in result.get("memories", [])}
    assert mid in returned_ids, "seeded memory should be retrieved"

    row = tmp_db.conn.execute(
        "SELECT last_accessed_at, strength FROM memories WHERE id = ?", (mid,)
    ).fetchone()
    assert row["last_accessed_at"] != old_iso, "hook didn't update timestamp"
    assert row["strength"] > 1.0, "hook didn't bump strength"
    assert row["strength"] == pytest.approx(1.5, rel=1e-3)


def test_assemble_context_skips_record_access_on_time_travel(tmp_db):
    """Time-travel queries (`as_of`) must NOT mutate strength/last_accessed."""
    old_iso = _iso(datetime(2024, 1, 1, tzinfo=timezone.utc))
    mid = _seed_memory(
        tmp_db,
        mid="mem_timetravel",
        strength=1.0,
        last_accessed=old_iso,
    )
    tmp_db.conn.execute(
        "UPDATE memories SET content = ? WHERE id = ?",
        ("python is the best language for ML", mid),
    )
    tmp_db.conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (mid,))
    tmp_db.conn.execute(
        "INSERT INTO memories_fts(memory_id, content) VALUES (?, ?)",
        (mid, "python is the best language for ML"),
    )
    tmp_db.conn.commit()

    # Future as_of → memory exists in that snapshot.
    future = _iso(datetime(2099, 1, 1, tzinfo=timezone.utc))
    assemble_context(tmp_db, "python ML", top_k=5, max_lines=5, as_of=future)

    row = tmp_db.conn.execute(
        "SELECT last_accessed_at, strength FROM memories WHERE id = ?", (mid,)
    ).fetchone()
    assert row["last_accessed_at"] == old_iso
    assert row["strength"] == pytest.approx(1.0, rel=1e-6)
