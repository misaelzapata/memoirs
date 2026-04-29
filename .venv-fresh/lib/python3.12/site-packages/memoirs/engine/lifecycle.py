"""Layer 5.2 — memory lifecycle.

Functions that age, promote, demote, archive, and merge memorias. The blueprint
chat.md lists 8 lifecycle ops; this module implements them. They run on demand
from `memoirs maintenance`/`memoirs cleanup` and on schedule from the daemon.

Public API:
    auto_merge_near_duplicates(db, threshold, dry_run)
    promote_frequently_used_memory(db, memory_id) / promote_all(db)
    demote_unused_memory(db, memory_id) / demote_all(db)
    refresh_memory_if_reconfirmed(db, memory_id)
    calculate_decay(memory) — public alias of internal _recency_score
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Iterable

from ..config import RECENCY_HALF_LIFE_DAYS
from ..core.ids import utc_now
from ..db import MemoirsDB
from . import embeddings as emb


log = logging.getLogger("memoirs.lifecycle")


# ---------------------------------------------------------------------------
# 5.2.1 — calculate_decay (public alias)
# ---------------------------------------------------------------------------


def calculate_decay(memory: dict) -> float:
    """Public API for the recency decay score (0..1).

    Wraps `engine.memory_engine._recency_score` so external callers don't need
    to import private symbols.
    """
    from .memory_engine import _recency_score
    return _recency_score(memory.get("last_used_at"), memory.get("created_at", utc_now()))


# ---------------------------------------------------------------------------
# 5.2.2 — promote / demote
# ---------------------------------------------------------------------------


def promote_frequently_used_memory(db: MemoirsDB, memory_id: str) -> bool:
    """If memory has been used ≥5 times AND last_used_at < 7 days ago,
    bump importance by 1 (cap 5). Returns True iff promoted."""
    row = db.conn.execute(
        "SELECT importance, usage_count, last_used_at FROM memories "
        "WHERE id = ? AND archived_at IS NULL",
        (memory_id,),
    ).fetchone()
    if not row:
        return False
    if row["usage_count"] < 5:
        return False
    if not _within_days(row["last_used_at"], 7):
        return False
    if row["importance"] >= 5:
        return False
    db.conn.execute(
        "UPDATE memories SET importance = importance + 1, updated_at = ? WHERE id = ?",
        (utc_now(), memory_id),
    )
    db.conn.commit()
    log.info("promoted memory=%s usage=%d → importance %d→%d",
             memory_id[:16], row["usage_count"], row["importance"], row["importance"] + 1)
    return True


def demote_unused_memory(db: MemoirsDB, memory_id: str) -> bool:
    """If memory has 0 uses AND age > 60 days, drop importance by 1 (floor 1)."""
    row = db.conn.execute(
        "SELECT importance, usage_count, created_at FROM memories "
        "WHERE id = ? AND archived_at IS NULL",
        (memory_id,),
    ).fetchone()
    if not row:
        return False
    if row["usage_count"] > 0:
        return False
    if not _older_than_days(row["created_at"], 60):
        return False
    if row["importance"] <= 1:
        return False
    db.conn.execute(
        "UPDATE memories SET importance = importance - 1, updated_at = ? WHERE id = ?",
        (utc_now(), memory_id),
    )
    db.conn.commit()
    log.info("demoted memory=%s unused>60d → importance %d→%d",
             memory_id[:16], row["importance"], row["importance"] - 1)
    return True


def promote_all(db: MemoirsDB) -> int:
    """Walk eligible memorias and promote each. Returns count promoted."""
    rows = db.conn.execute(
        "SELECT id FROM memories WHERE archived_at IS NULL "
        "AND usage_count >= 5 AND importance < 5"
    ).fetchall()
    return sum(1 for r in rows if promote_frequently_used_memory(db, r["id"]))


def demote_all(db: MemoirsDB) -> int:
    """Walk eligible memorias and demote each."""
    rows = db.conn.execute(
        "SELECT id FROM memories WHERE archived_at IS NULL "
        "AND usage_count = 0 AND importance > 1 "
        "AND created_at <= datetime('now', '-60 days')"
    ).fetchall()
    return sum(1 for r in rows if demote_unused_memory(db, r["id"]))


# ---------------------------------------------------------------------------
# 5.2.3 — refresh on reconfirm
# ---------------------------------------------------------------------------


def refresh_memory_if_reconfirmed(db: MemoirsDB, memory_id: str) -> None:
    """Bump confidence and reset last_used_at when an UPDATE reconfirms a memory."""
    db.conn.execute(
        "UPDATE memories SET "
        "  confidence = MIN(1.0, confidence + 0.05), "
        "  last_used_at = ?, "
        "  usage_count = usage_count + 1, "
        "  updated_at = ? "
        "WHERE id = ? AND archived_at IS NULL",
        (utc_now(), utc_now(), memory_id),
    )
    db.conn.commit()


# ---------------------------------------------------------------------------
# 5.5 — auto-merge near-duplicates (the big one)
# ---------------------------------------------------------------------------


def auto_merge_near_duplicates(
    db: MemoirsDB,
    *,
    threshold: float = 0.92,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict:
    """Walk active memorias, find their nearest neighbor, and merge near-dups.

    Same type + similarity ≥ threshold → keep higher-score memoria, archive
    the other. Different type + similarity ≥ threshold → flag both as
    contradictory in metadata for human review (no merge).

    Returns:
        {"merged": int, "contradictions": int, "scanned": int, "dry_run": bool}
    """
    rows = db.conn.execute(
        "SELECT id, type, content, score FROM memories "
        "WHERE archived_at IS NULL "
        "ORDER BY score DESC"
        + (f" LIMIT {int(limit)}" if limit else "")
    ).fetchall()

    merged = 0
    contradictions = 0
    scanned = 0
    seen_archived: set[str] = set()
    now = utc_now()

    for r in rows:
        if r["id"] in seen_archived:
            continue
        scanned += 1
        # Top-2: skip self (top-1)
        try:
            similar = emb.search_similar_memories(db, r["content"] or "", top_k=3)
        except emb.EmbeddingsUnavailable:
            log.warning("auto_merge: embeddings unavailable, skipping")
            return {"merged": 0, "contradictions": 0, "scanned": 0, "dry_run": dry_run,
                    "error": "embeddings_unavailable"}
        # Filter out self + already archived
        candidates_other = [
            s for s in similar
            if s["id"] != r["id"] and s["id"] not in seen_archived
        ]
        if not candidates_other:
            continue
        top = candidates_other[0]
        if top["similarity"] < threshold:
            continue

        if top["type"] == r["type"]:
            # Same type → MERGE: keep higher score, archive the other
            keep, drop = (r, top) if r["score"] >= (top["score"] or 0) else (top, r)
            if not dry_run:
                _merge_into(db, keeper_id=keep["id"], dropped_id=drop["id"], now=now,
                            similarity=top["similarity"])
            seen_archived.add(drop["id"])
            merged += 1
            log.info(
                "auto_merge: %s sim=%.3f keep=%s drop=%s (%s)",
                "DRY-RUN" if dry_run else "MERGE",
                top["similarity"], keep["id"][:16], drop["id"][:16], r["type"],
            )
        else:
            # Different type → flag contradiction, don't merge
            if not dry_run:
                _flag_contradiction(db, a_id=r["id"], b_id=top["id"],
                                    similarity=top["similarity"], now=now)
            contradictions += 1
            log.info(
                "auto_merge: %s sim=%.3f a=%s(%s) b=%s(%s)",
                "DRY-RUN-CONTRADICTION" if dry_run else "CONTRADICTION",
                top["similarity"], r["id"][:16], r["type"], top["id"][:16], top["type"],
            )

    return {"merged": merged, "contradictions": contradictions, "scanned": scanned, "dry_run": dry_run}


def _merge_into(db: MemoirsDB, *, keeper_id: str, dropped_id: str, now: str, similarity: float) -> None:
    """Archive `dropped_id`, transferring its usage and entity links to `keeper_id`."""
    with db.conn:
        # Sum usage_count + max user_signal into keeper
        keeper = db.conn.execute(
            "SELECT usage_count, user_signal FROM memories WHERE id = ?", (keeper_id,)
        ).fetchone()
        dropped = db.conn.execute(
            "SELECT usage_count, user_signal FROM memories WHERE id = ?", (dropped_id,)
        ).fetchone()
        if keeper and dropped:
            db.conn.execute(
                "UPDATE memories SET "
                "  usage_count = ?, "
                "  user_signal = ?, "
                "  updated_at = ? "
                "WHERE id = ?",
                (
                    int(keeper["usage_count"]) + int(dropped["usage_count"]),
                    max(float(keeper["user_signal"] or 0), float(dropped["user_signal"] or 0)),
                    now,
                    keeper_id,
                ),
            )
        # Transfer entity links (de-duped via PK)
        db.conn.execute(
            "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) "
            "SELECT ?, entity_id FROM memory_entities WHERE memory_id = ?",
            (keeper_id, dropped_id),
        )
        # Drop the loser's vector entry to free the ANN index
        db.conn.execute("DELETE FROM vec_memories WHERE memory_id = ?", (dropped_id,))
        db.conn.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (dropped_id,))
        # Archive the loser with a pointer to the winner
        db.conn.execute(
            "UPDATE memories SET "
            "  archived_at = ?, "
            "  archive_reason = ?, "
            "  superseded_by = ?, "
            "  updated_at = ? "
            "WHERE id = ?",
            (now, f"merged into {keeper_id} (sim={similarity:.3f})", keeper_id, now, dropped_id),
        )


def _flag_contradiction(db: MemoirsDB, *, a_id: str, b_id: str, similarity: float, now: str) -> None:
    """Mark two memorias as contradictory candidates in metadata (no archive)."""
    note = f"semantic-similar to {b_id} (sim={similarity:.3f}) but different type"
    note_b = f"semantic-similar to {a_id} (sim={similarity:.3f}) but different type"
    db.conn.execute(
        "UPDATE memories SET metadata_json = json_set(COALESCE(metadata_json,'{}'),"
        "  '$.contradiction_with', ?), updated_at = ? WHERE id = ?",
        (b_id, now, a_id),
    )
    db.conn.execute(
        "UPDATE memories SET metadata_json = json_set(COALESCE(metadata_json,'{}'),"
        "  '$.contradiction_with', ?), updated_at = ? WHERE id = ?",
        (a_id, now, b_id),
    )
    db.conn.commit()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _within_days(iso_ts: str | None, days: int) -> bool:
    if not iso_ts:
        return False
    try:
        ts = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except ValueError:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0
    return 0 <= age <= days


def _older_than_days(iso_ts: str | None, days: int) -> bool:
    if not iso_ts:
        return False
    try:
        ts = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except ValueError:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0
    return age > days
