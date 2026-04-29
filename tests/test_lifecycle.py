"""Tests for engine/lifecycle: promote, demote, refresh."""
from datetime import datetime, timezone, timedelta

from memoirs.engine.lifecycle import (
    _older_than_days,
    _within_days,
    demote_unused_memory,
    promote_frequently_used_memory,
    refresh_memory_if_reconfirmed,
)


def _seed_memory(db, *, memory_id, importance=3, usage_count=0, last_used_at=None,
                 created_at=None, content="x"):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    created_at = created_at or now
    db.conn.execute(
        "INSERT INTO memories (id, type, content, content_hash, importance, "
        "confidence, score, usage_count, user_signal, valid_from, "
        "metadata_json, created_at, updated_at, last_used_at) "
        "VALUES (?, 'fact', ?, 'hash_'||?, ?, 0.5, 0, ?, 0, ?, '{}', ?, ?, ?)",
        (memory_id, content, memory_id, importance, usage_count, now, created_at, now, last_used_at),
    )
    db.conn.commit()


def test_within_days_helper():
    now = datetime.now(timezone.utc)
    assert _within_days((now - timedelta(days=3)).isoformat(timespec="seconds"), 7)
    assert not _within_days((now - timedelta(days=10)).isoformat(timespec="seconds"), 7)


def test_older_than_days_helper():
    now = datetime.now(timezone.utc)
    assert _older_than_days((now - timedelta(days=70)).isoformat(timespec="seconds"), 60)
    assert not _older_than_days((now - timedelta(days=30)).isoformat(timespec="seconds"), 60)


def test_promote_when_usage_high_and_recent(tmp_db):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _seed_memory(tmp_db, memory_id="m1", importance=3, usage_count=10, last_used_at=now)
    assert promote_frequently_used_memory(tmp_db, "m1") is True
    imp = tmp_db.conn.execute("SELECT importance FROM memories WHERE id='m1'").fetchone()[0]
    assert imp == 4


def test_promote_skip_when_low_usage(tmp_db):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _seed_memory(tmp_db, memory_id="m2", usage_count=2, last_used_at=now)
    assert promote_frequently_used_memory(tmp_db, "m2") is False


def test_promote_caps_at_5(tmp_db):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _seed_memory(tmp_db, memory_id="m3", importance=5, usage_count=10, last_used_at=now)
    assert promote_frequently_used_memory(tmp_db, "m3") is False


def test_demote_old_unused_drops_importance(tmp_db):
    old = (datetime.now(timezone.utc) - timedelta(days=70)).isoformat(timespec="seconds")
    _seed_memory(tmp_db, memory_id="m4", importance=3, usage_count=0, created_at=old)
    assert demote_unused_memory(tmp_db, "m4") is True
    imp = tmp_db.conn.execute("SELECT importance FROM memories WHERE id='m4'").fetchone()[0]
    assert imp == 2


def test_demote_skip_when_used(tmp_db):
    old = (datetime.now(timezone.utc) - timedelta(days=70)).isoformat(timespec="seconds")
    _seed_memory(tmp_db, memory_id="m5", importance=3, usage_count=5, created_at=old)
    assert demote_unused_memory(tmp_db, "m5") is False


def test_demote_floors_at_1(tmp_db):
    old = (datetime.now(timezone.utc) - timedelta(days=70)).isoformat(timespec="seconds")
    _seed_memory(tmp_db, memory_id="m6", importance=1, usage_count=0, created_at=old)
    assert demote_unused_memory(tmp_db, "m6") is False


def test_refresh_bumps_confidence(tmp_db):
    _seed_memory(tmp_db, memory_id="m7", usage_count=0)
    initial_conf = tmp_db.conn.execute("SELECT confidence FROM memories WHERE id='m7'").fetchone()[0]
    refresh_memory_if_reconfirmed(tmp_db, "m7")
    new_conf = tmp_db.conn.execute("SELECT confidence FROM memories WHERE id='m7'").fetchone()[0]
    assert new_conf > initial_conf
    new_usage = tmp_db.conn.execute("SELECT usage_count FROM memories WHERE id='m7'").fetchone()[0]
    assert new_usage == 1
