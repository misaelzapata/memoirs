"""Layer 5 — memory engine.

Implements the core that turns raw candidates into curated memories and
assembles compact context for retrieval. Sub-pieces from chat-blueprint.md:

  5.1 Consolidation        — decide_memory_action / consolidate_candidate
  5.2 Lifecycle            — decay, expire, promote, archive
  5.3 Scoring              — fórmula importance·0.35 + confidence·0.20 + ...
  5.4 Temporal versioning  — valid_from/valid_to, supersede chain
  5.5 Deduplication        — exact (content_hash) + semantic (cosine ≥ 0.92)
  5.6 Reasoning            — assemble_context: select → rank → conflicts → compress
  5.7 Event-driven         — drained from event_queue (db.py)

This module is pure-Python and works zero-deps. It calls into `embeddings`
opportunistically — if real vectors are available, semantic dedup uses them;
otherwise falls back to Jaccard. Same for retrieval.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator

from . import embeddings as emb
from . import lifecycle_decisions as _lcd
from ..core.normalize import canonicalize_for_dedup, should_skip_extraction
from ..db import MemoirsDB, content_hash, stable_id, utc_now
from ..models import DEFAULT_SCOPE, Scope, ScopeFilter
from .curator import Candidate


log = logging.getLogger("memoirs.engine")


# ----------------------------------------------------------------------
# Valid memory types — Layer 5 enforces this set on every persisted row.
# ----------------------------------------------------------------------
#
# The historical canonical set lives in ``gemma.ALLOWED_TYPES`` (used by the
# extractor). This local mirror adds ``tool_call`` (P1-8 GAP) so the new
# ``record_tool_call`` API can validate without changing the extractor's
# contract — Gemma should never emit ``tool_call`` from free text; only the
# explicit tool-call recorder does.
_VALID_MEMORY_TYPES = frozenset(
    {
        "preference",
        "fact",
        "project",
        "task",
        "decision",
        "style",
        "credential_pointer",
        "tool_call",
        "procedural",
    }
)


# ----------------------------------------------------------------------
# 5.3 Scoring — Ebbinghaus forgetting curve (P1-7 / MemoryBank, AAAI'24)
# ----------------------------------------------------------------------

# Strength ceiling: log_1.5(64) ≈ 10 reinforcements, well past the point
# where decay is effectively flat. Keeps `exp` numerically stable even when
# a memory is hammered for years.
_STRENGTH_MAX = 64.0
_STRENGTH_GROWTH = 1.5


def _parse_iso(ts: str | None) -> datetime | None:
    """Lenient ISO-8601 parser. Accepts trailing ``Z`` and naive timestamps
    (assumed UTC). Returns None on garbage input.
    """
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def ebbinghaus_recency(
    last_accessed_at: str | None,
    strength: float,
    *,
    now: datetime | None = None,
) -> float:
    """Ebbinghaus forgetting curve: ``R(t) = exp(-Δt / (S * 24h))``.

    Parameters
    ----------
    last_accessed_at:
        ISO-8601 timestamp of the most recent access. ``None`` is treated as
        "perfectly fresh" — the caller (``calculate_memory_score``) handles
        the fallback to ``created_at`` before calling us.
    strength:
        Consolidation factor S (≥ 1.0). Each access multiplies S by 1.5
        (capped at :data:`_STRENGTH_MAX`). Higher S → slower decay.
    now:
        Reference time, mostly for tests. Defaults to ``datetime.now(UTC)``.

    Returns
    -------
    float
        Decay value clamped to ``[0.01, 1.0]``. The lower clamp prevents a
        memory from contributing literally zero to the composite score, so
        importance + user_signal can still surface ancient-but-pinned facts.
    """
    if last_accessed_at is None:
        return 1.0
    ts = _parse_iso(last_accessed_at)
    if ts is None:
        # Unparseable timestamp → mid value, same as the previous scorer's
        # tolerance behavior. Better than 0.0 (which would archive the row).
        return 0.5
    ref_now = now or datetime.now(timezone.utc)
    if ref_now.tzinfo is None:
        ref_now = ref_now.replace(tzinfo=timezone.utc)
    delta_hours = (ref_now - ts).total_seconds() / 3600.0
    if delta_hours <= 0:
        return 1.0
    s = max(1.0, float(strength or 1.0))
    # S=1, Δt=24h → exp(-1) ≈ 0.368 (one e-fold per day).
    decayed = math.exp(-delta_hours / (s * 24.0))
    return max(0.01, min(1.0, decayed))


def _recency_score(last_used_at: str | None, created_at: str) -> float:
    """Backwards-compatible wrapper used by callers that don't have access
    to the new ``strength`` column (lifecycle helpers, mcp_tools, etc.).

    Internally delegates to :func:`ebbinghaus_recency` with ``S=1.0``,
    falling back to ``created_at`` when ``last_used_at`` is missing — the
    same fallback policy the legacy scorer used.
    """
    ref = last_used_at or created_at
    return ebbinghaus_recency(ref, strength=1.0)


def record_access(
    db: MemoirsDB,
    memory_id: str,
    *,
    now: datetime | None = None,
) -> None:
    """Mark a memory as freshly accessed: bump ``last_accessed_at`` to now
    and multiply ``strength`` by 1.5 (capped at :data:`_STRENGTH_MAX`).

    Idempotent at the row level — repeated calls in quick succession all
    use the same ``now`` if you pass it explicitly. Best-effort: silently
    no-ops on rows that don't exist or DBs that haven't run migration 004
    yet (so older test fixtures don't break).
    """
    ts = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    try:
        db.conn.execute(
            """
            UPDATE memories
            SET last_accessed_at = ?,
                strength = MIN(?, COALESCE(strength, 1.0) * ?)
            WHERE id = ?
            """,
            (ts, _STRENGTH_MAX, _STRENGTH_GROWTH, memory_id),
        )
    except Exception:
        # Migration 004 not applied or the column is missing — this hook
        # is purely advisory, never fail the retrieval path because of it.
        log.debug("record_access skipped for %s (no strength column?)", memory_id[:16])


def _normalize_importance(importance: int) -> float:
    return max(0.0, min(1.0, (importance - 1) / 4.0))  # 1..5 → 0..1


def _normalize_usage(usage_count: int) -> float:
    # diminishing returns
    if usage_count <= 0:
        return 0.0
    return min(1.0, 1.0 - 1.0 / (1.0 + usage_count / 5.0))


def calculate_memory_score(memory: dict) -> float:
    """Score formula from chat.md.

    Recency now uses the Ebbinghaus forgetting curve (P1-7): each access
    bumps the memory's ``strength`` and shifts the decay curve to the right.
    Falls back to ``last_used_at`` / ``created_at`` when ``last_accessed_at``
    is missing, so rows that pre-date migration 004 still score correctly.
    """
    importance = _normalize_importance(int(memory.get("importance", 3)))
    confidence = float(memory.get("confidence", 0.5))
    last_access = (
        memory.get("last_accessed_at")
        or memory.get("last_used_at")
        or memory.get("created_at", utc_now())
    )
    strength = float(memory.get("strength", 1.0) or 1.0)
    recency = ebbinghaus_recency(last_access, strength)
    usage = _normalize_usage(int(memory.get("usage_count", 0)))
    signal = float(memory.get("user_signal", 0.0))
    score = (
        importance * 0.35
        + confidence * 0.20
        + recency * 0.15
        + usage * 0.15
        + signal * 0.15
    )
    return round(score, 4)


def recompute_all_scores(db: MemoirsDB) -> int:
    rows = db.conn.execute(
        "SELECT id, importance, confidence, usage_count, last_used_at, user_signal, created_at FROM memories WHERE archived_at IS NULL"
    ).fetchall()
    n = 0
    with db.conn:
        for r in rows:
            score = calculate_memory_score(dict(r))
            db.conn.execute(
                "UPDATE memories SET score = ?, updated_at = ? WHERE id = ?",
                (score, utc_now(), r["id"]),
            )
            n += 1
    return n


# ----------------------------------------------------------------------
# 5.5 Deduplication
# ----------------------------------------------------------------------


def detect_exact_duplicate(db: MemoirsDB, content: str) -> dict | None:
    """Return an active memory whose content matches ``content`` exactly OR
    after canonicalization (Fix #1.C of GAP audit Fase 5C).

    The canonical form ignores whitespace, case, trailing punctuation, and
    URLs, so two candidates that differ only in formatting collapse to the
    same memoria — and cross-TYPE matches are also returned (the curator
    layer decides what to do with a type mismatch).
    """
    # Fast path: byte-for-byte match via the existing unique index.
    h = content_hash(content)
    row = db.conn.execute(
        "SELECT id, type, content, importance FROM memories "
        "WHERE content_hash = ? AND archived_at IS NULL",
        (h,),
    ).fetchone()
    if row:
        return dict(row)

    # Slow path: canonical comparison. We scan active rows — the
    # `idx_memories_content_hash` partial index already keeps the working
    # set tight, and this code path only triggers when the fast hash misses.
    canon = canonicalize_for_dedup(content)
    if not canon:
        return None
    for r in db.conn.execute(
        "SELECT id, type, content, importance FROM memories "
        "WHERE archived_at IS NULL"
    ).fetchall():
        other = canonicalize_for_dedup(r["content"] or "")
        if other and other == canon:
            return dict(r)
    return None


# Default semantic-dup threshold lowered from 0.92 → 0.85 (Fix #1.C, GAP
# audit Fase 5C). The audit found 221 pairs at sim ≥ 0.85 that the old
# threshold let through as separate memorias.
SEMANTIC_DUP_THRESHOLD = 0.85


def detect_semantic_duplicate(
    db: MemoirsDB, content: str, threshold: float = SEMANTIC_DUP_THRESHOLD
) -> dict | None:
    matches = emb.find_semantic_duplicates(db, content, threshold=threshold)
    return matches[0] if matches else None


# ----------------------------------------------------------------------
# 5.1 Consolidation
# ----------------------------------------------------------------------


@dataclass
class Decision:
    action: str  # ADD | UPDATE | MERGE | IGNORE | CONTRADICTION | EXPIRE | ARCHIVE
    target_memory_id: str | None = None
    reason: str = ""
    # P1-10: cascade actions emitted by lifecycle_decisions.enrich_decision —
    # e.g. an ADD that obsoletes a stale neighbor carries an EXPIRE here.
    secondary_actions: list["Decision"] = field(default_factory=list)
    # Migration 012 — attribution: who/what produced this decision. Filled by
    # _decide_memory_action_base before the persist call so we can stamp
    # ``provenance_json`` on the new memoria. Defaults to "heuristic"; the
    # curator path overrides to "curator" when Gemma/Qwen actually answered.
    actor: str = "heuristic"
    process: str | None = None  # "extract" | "consolidate" | "manual" | "import"


_CURATOR_NEIGHBORS = 5


def _curator_mode() -> str:
    """Read MEMOIRS_CURATOR_ENABLED (legacy: MEMOIRS_GEMMA_CURATOR) ∈
    {on, off, auto}, default auto.
    """
    os_env = __import__("os").environ
    # Late env read so tests can flip the flag per-case. Prefer the new var,
    # fall back to the legacy one for backward compat.
    raw = os_env.get("MEMOIRS_CURATOR_ENABLED")
    if not raw:
        raw = os_env.get("MEMOIRS_GEMMA_CURATOR", "auto")
    val = (raw or "auto").strip().lower()
    if val not in {"on", "off", "auto"}:
        return "auto"
    return val


def _heuristic_decide_memory_action(db: MemoirsDB, candidate: Candidate) -> Decision:
    """Rules-based fallback curator. Kept verbatim from the original
    implementation so that disabling Gemma (`MEMOIRS_GEMMA_CURATOR=off`) is
    behaviorally identical to the pre-P1-11 path.

    Fix #1 (GAP audit Fase 5C):
      - REJECT noise candidates (defense in depth — extractor already drops
        them, but a memoria can also be added programmatically).
      - Cross-type EXACT dup → MERGE at the more-important type (instead of
        the legacy "exact match across types" silent UPDATE).
    """
    if not candidate.content.strip():
        return Decision("IGNORE", reason="empty content")

    skip, reason = should_skip_extraction(candidate.content)
    if skip:
        return Decision("REJECT", reason=f"noise: {reason}")

    exact = detect_exact_duplicate(db, candidate.content)
    if exact:
        if exact.get("type") == candidate.type:
            return Decision(
                "UPDATE",
                target_memory_id=exact["id"],
                reason="exact duplicate → reinforce",
            )
        # Cross-type exact dup: MERGE into the more-important slot. The
        # `apply_decision` MERGE branch uses max(importance) and may
        # re-tag the type via the explicit `_promote_type_to` hint.
        return Decision(
            "MERGE",
            target_memory_id=exact["id"],
            reason=(
                f"exact duplicate cross-type ({exact.get('type')} vs "
                f"{candidate.type}) → merge at higher importance"
            ),
        )

    semantic = detect_semantic_duplicate(db, candidate.content)
    if semantic:
        # Same type → MERGE; different type → flag CONTRADICTION
        if semantic["type"] == candidate.type:
            return Decision("MERGE", target_memory_id=semantic["id"], reason=f"semantic dup (sim={semantic.get('similarity')})")
        return Decision("CONTRADICTION", target_memory_id=semantic["id"], reason="similar content, different type")

    return Decision("ADD", reason="new memory")


def _gather_curator_neighbors(db: MemoirsDB, candidate: Candidate,
                               *, top_k: int = _CURATOR_NEIGHBORS) -> list[dict]:
    """Collect a small ranked list of neighbors (most similar first) to feed
    into `curator_consolidate`. Falls back to an empty list if embeddings
    aren't wired up — Gemma can still ADD vs IGNORE without context.
    """
    try:
        return emb.search_similar_memories(db, candidate.content, top_k=top_k)
    except Exception:
        log.debug("curator: search_similar_memories unavailable, neighbors=[]",
                  exc_info=True)
        return []


def _decide_memory_action_base(db: MemoirsDB, candidate: Candidate) -> Decision:
    """Decide ADD/UPDATE/MERGE/IGNORE/EXPIRE for a new candidate.

    Behaviour controlled by ``MEMOIRS_CURATOR_ENABLED`` env var (legacy:
    ``MEMOIRS_GEMMA_CURATOR``, still honored as fallback):

    - ``off``   → use the legacy heuristic curator only.
    - ``on``    → require Gemma. If Gemma is unavailable / returns garbage,
                  log and fall back to the heuristic anyway (we never block
                  consolidation).
    - ``auto``  → try Gemma first; silently fall back on miss. (Default.)

    The heuristic still runs the cheap exact-duplicate check before any
    Gemma call — exact matches are unambiguous and we save a model
    round-trip on every reinforcement.
    """
    if not candidate.content.strip():
        return Decision("IGNORE", reason="empty content")

    # Fix #1 (GAP audit Fase 5C): defense in depth. Even if the extractor
    # filter (curator_extract → should_skip_extraction) was bypassed, the
    # curator must REJECT obvious noise BEFORE any embedding lookup or
    # Gemma call. This is also reachable when callers add a candidate
    # programmatically (e.g. `mcp_add_memory`).
    skip, reason = should_skip_extraction(candidate.content)
    if skip:
        return Decision("REJECT", reason=f"noise: {reason}")

    mode = _curator_mode()
    # Cheap pre-check: exact dupes never need a model.
    exact = detect_exact_duplicate(db, candidate.content)
    if exact:
        if exact.get("type") == candidate.type:
            return Decision(
                "UPDATE",
                target_memory_id=exact["id"],
                reason="exact duplicate → reinforce",
            )
        # Cross-type EXACT dup: MERGE at higher importance (Fix #1.C).
        return Decision(
            "MERGE",
            target_memory_id=exact["id"],
            reason=(
                f"exact duplicate cross-type ({exact.get('type')} vs "
                f"{candidate.type}) → merge at higher importance"
            ),
        )

    if mode == "off":
        return _heuristic_decide_memory_action(db, candidate)

    # `auto` and `on` both attempt the LLM curator first.
    try:
        from . import curator as _curator  # local import to avoid cycles at module load
    except Exception:  # pragma: no cover — defensive
        return _heuristic_decide_memory_action(db, candidate)

    neighbors = _gather_curator_neighbors(db, candidate)
    cand_dict = {
        "type": candidate.type,
        "content": candidate.content,
        "importance": candidate.importance,
        "confidence": candidate.confidence,
    }

    try:
        result = _curator.curator_consolidate(cand_dict, neighbors)
    except Exception as e:
        log.warning("decide_memory_action: curator_consolidate raised %s", e)
        result = {"action": None, "source": "gemma_parse_error",
                  "reason": str(e)}

    action = (result or {}).get("action")
    source = (result or {}).get("source", "gemma_unavailable")

    if action in {"ADD", "UPDATE", "MERGE", "IGNORE", "EXPIRE"}:
        target_id = result.get("target_id")
        reason = result.get("reason") or f"curator:{action.lower()}"
        # Sanity: actions that touch a target need a target — fall back
        # to heuristic if the model nominated UPDATE/MERGE/EXPIRE without
        # a candidate target AND no neighbors exist.
        if action in {"UPDATE", "MERGE", "EXPIRE"} and not target_id:
            log.info(
                "decide_memory_action: curator %s without target → heuristic fallback",
                action,
            )
            return _heuristic_decide_memory_action(db, candidate)
        log.debug(
            "decide_memory_action: curator=%s target=%s reason=%s",
            action, (target_id or "-")[:16], reason[:80],
        )
        # Reason prefix uses the legacy ``gemma:`` token for backward compat
        # with tests / log scrapers that grep on it.
        return Decision(
            action, target_memory_id=target_id, reason=f"gemma: {reason}",
            actor="curator", process="consolidate",
        )

    # Curator did not produce a usable action.
    if mode == "on":
        log.warning(
            "decide_memory_action: MEMOIRS_CURATOR_ENABLED=on but curator "
            "returned source=%s — falling back to heuristic anyway",
            source,
        )
    else:
        log.debug("decide_memory_action: curator source=%s → heuristic", source)
    return _heuristic_decide_memory_action(db, candidate)


def decide_memory_action(db: MemoirsDB, candidate: Candidate) -> Decision:
    """Public curator entrypoint. Wraps the base curator with EXPIRE/ARCHIVE
    enrichment (P1-10) — primary action stays the same, but cascading
    secondary actions (e.g. expire an obsolete neighbor) are appended.
    """
    decision = _decide_memory_action_base(db, candidate)
    if decision.action in {"IGNORE", "REJECT", "CONTRADICTION", "EXPIRE", "ARCHIVE"}:
        return decision
    try:
        neighbors = _gather_curator_neighbors(db, candidate)
    except Exception:
        neighbors = []
    try:
        return _lcd.enrich_decision(decision, candidate, neighbors)
    except Exception:
        log.exception("enrich_decision failed; using base decision")
        return decision


def _maybe_link_memory(db: MemoirsDB, memory_id: str) -> None:
    """Run A-MEM Zettelkasten linking for a freshly added/updated memory.

    Controlled by ``MEMOIRS_ZETTELKASTEN`` (default ``on``). Failures are
    logged but never propagate — link generation is auxiliary, not critical
    to the consolidation path.
    """
    from . import zettelkasten as _zk  # local: avoid forcing the import for callers that disable it
    if not _zk._is_enabled():
        return
    try:
        _zk.link_memory(db, memory_id)
    except Exception:
        log.exception("zettelkasten link_memory failed for %s", memory_id[:16])


def apply_decision(
    db: MemoirsDB,
    candidate: Candidate,
    decision: Decision,
    *,
    scope: Scope = DEFAULT_SCOPE,
) -> dict:
    """Apply ``decision`` for ``candidate``.

    Scoping (P0-5 / P3-3): when ``scope`` is provided the new ADD row is
    persisted with the caller's ``user_id`` / ``agent_id`` / ``run_id`` /
    ``namespace`` / ``visibility``. Defaults to :data:`DEFAULT_SCOPE`
    (``user_id="local"``, ``visibility="private"``) so existing single-user
    callers get the legacy behaviour with zero changes.

    P1-10: any ``decision.secondary_actions`` (cascading EXPIRE / ARCHIVE
    emitted by ``lifecycle_decisions.enrich_decision``) are applied after
    the primary action and their summaries are returned under
    ``result["secondary_results"]``.
    """
    primary_result = _apply_primary_decision(db, candidate, decision, scope=scope)
    secondaries = list(getattr(decision, "secondary_actions", None) or [])
    if not secondaries:
        return primary_result
    secondary_results: list[dict] = []
    for sec in secondaries:
        try:
            sec_res = _apply_primary_decision(db, candidate, sec, scope=scope)
        except Exception:
            log.exception(
                "secondary action %s failed for target=%s",
                sec.action, (sec.target_memory_id or "-")[:16],
            )
            continue
        secondary_results.append(sec_res)
    if secondary_results:
        primary_result["secondary_results"] = secondary_results
    return primary_result


def _apply_primary_decision(
    db: MemoirsDB,
    candidate: Candidate,
    decision: Decision,
    *,
    scope: Scope = DEFAULT_SCOPE,
) -> dict:
    """Inner worker for :func:`apply_decision` — handles a single decision
    (no cascading). Split out so secondary_actions reuse the exact same
    code paths as primaries.
    """
    now = utc_now()
    result: dict = {"action": decision.action, "reason": decision.reason}

    if decision.action == "IGNORE":
        return result

    # Fix #1.B (GAP audit Fase 5C): REJECT — candidate is noise. No memory
    # row is created or touched. The caller (consolidate_candidate) is
    # responsible for persisting `status='rejected'` + `rejection_reason`
    # on the originating memory_candidates row. Returned reason is forwarded
    # there.
    if decision.action == "REJECT":
        result["rejection_reason"] = decision.reason
        return result

    # ACL write check (Fase 5A): mutating actions (UPDATE / MERGE / EXPIRE /
    # ARCHIVE / CONTRADICTION) must not let a non-owner clobber another
    # user's memory. ADD always creates a new row owned by ``scope``, so it
    # does not need the predicate. Loading the target row also catches
    # dangling target_memory_ids — a missing row falls through to the
    # action-specific branches which already no-op cleanly.
    _MUTATING = {"UPDATE", "MERGE", "EXPIRE", "ARCHIVE", "CONTRADICTION"}
    if decision.action in _MUTATING and decision.target_memory_id:
        from . import acl as _acl
        target_row = db.conn.execute(
            "SELECT id, user_id, visibility FROM memories WHERE id = ?",
            (decision.target_memory_id,),
        ).fetchone()
        if target_row is not None:
            target_dict = dict(target_row)
            if not _acl.can_write(target_dict, scope):
                owner = target_dict.get("user_id") or "local"
                return {
                    "action": "REJECTED",
                    "reason": (
                        f"ACL: cannot write to memory owned by user {owner}"
                    ),
                    "memory_id": decision.target_memory_id,
                }

    if decision.action == "ADD":
        mid = stable_id("mem", candidate.type, candidate.content)
        h = content_hash(candidate.content)
        provenance = json.dumps({
            "actor": decision.actor,
            "process": decision.process or "extract",
            "decision": "ADD",
            "reason": decision.reason,
        }, separators=(",", ":"))
        with db.conn:
            db.conn.execute(
                """
                INSERT INTO memories (
                    id, type, content, content_hash, importance, confidence,
                    score, usage_count, user_signal, valid_from, metadata_json,
                    provenance_json,
                    user_id, agent_id, run_id, namespace, visibility,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0, ?, '{}', ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(content_hash) WHERE archived_at IS NULL DO NOTHING
                """,
                (
                    mid, candidate.type, candidate.content, h,
                    candidate.importance, candidate.confidence, now,
                    provenance,
                    scope.user_id, scope.agent_id, scope.run_id,
                    scope.namespace, scope.visibility,
                    now, now,
                ),
            )
        emb.upsert_memory_embedding(db, mid, candidate.content)
        # score
        row = db.conn.execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone()
        if row:
            score = calculate_memory_score(dict(row))
            db.conn.execute("UPDATE memories SET score = ? WHERE id = ?", (score, mid))
            db.conn.commit()
        result["memory_id"] = mid
        # P1-3: A-MEM Zettelkasten links — best-effort, never blocks the insert.
        _maybe_link_memory(db, mid)
        # P0-4: drop a `memory_promoted` event so downstream consumers (graph
        # indexers, audit, etc.) can react without polling. Best-effort.
        try:
            from . import event_queue as _eq
            _eq.enqueue(db, event_type="memory_promoted", payload={"memory_id": mid, "type": candidate.type})
        except Exception:  # noqa: BLE001
            log.exception("event_queue enqueue failed for memory_promoted %s", mid)
        return result

    if decision.action in {"UPDATE", "MERGE"} and decision.target_memory_id:
        # bump confidence (capped) and importance (max), refresh last_used_at
        # via lifecycle.refresh_memory_if_reconfirmed (Layer 5.2 contract).
        target = db.conn.execute(
            "SELECT type, importance, confidence, usage_count FROM memories WHERE id = ?",
            (decision.target_memory_id,),
        ).fetchone()
        if target:
            new_conf = min(1.0, float(target["confidence"]) + 0.05)
            cand_imp = int(candidate.importance)
            tgt_imp = int(target["importance"])
            new_imp = max(tgt_imp, cand_imp)
            # Fix #1.C (GAP audit Fase 5C): cross-type MERGE promotes the type
            # to whichever side has the higher importance. Ties keep the
            # existing target.type so we don't churn rows on equal-importance
            # collisions.
            new_type = target["type"]
            if (
                decision.action == "MERGE"
                and candidate.type
                and candidate.type != target["type"]
                and cand_imp > tgt_imp
            ):
                new_type = candidate.type
            with db.conn:
                db.conn.execute(
                    """
                    UPDATE memories
                    SET type = ?, confidence = ?, importance = ?,
                        usage_count = usage_count + 1,
                        last_used_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (new_type, new_conf, new_imp, now, now, decision.target_memory_id),
                )
            row = db.conn.execute("SELECT * FROM memories WHERE id = ?", (decision.target_memory_id,)).fetchone()
            if row:
                db.conn.execute(
                    "UPDATE memories SET score = ? WHERE id = ?",
                    (calculate_memory_score(dict(row)), decision.target_memory_id),
                )
                db.conn.commit()
        result["memory_id"] = decision.target_memory_id
        # P1-3: refresh links on UPDATE/MERGE too — neighborhood may shift.
        _maybe_link_memory(db, decision.target_memory_id)
        return result

    if decision.action == "CONTRADICTION" and decision.target_memory_id:
        # Record but don't auto-resolve. Caller (or human review) decides.
        with db.conn:
            db.conn.execute(
                """
                UPDATE memories
                SET metadata_json = json_set(COALESCE(metadata_json,'{}'), '$.contradiction', ?),
                    updated_at = ?
                WHERE id = ?
                """,
                (candidate.content, now, decision.target_memory_id),
            )
        result["memory_id"] = decision.target_memory_id
        return result

    # P1-10: EXPIRE — neighbor was made obsolete by the candidate. Close its
    # validity window (valid_to=now, status=expired) and link the supersedor
    # if the candidate already produced a memory id (from a primary ADD).
    if decision.action == "EXPIRE" and decision.target_memory_id:
        superseded_by = result.get("memory_id") or _resolve_supersedor_id(
            db, candidate
        )
        with db.conn:
            db.conn.execute(
                """
                UPDATE memories
                SET valid_to = COALESCE(valid_to, ?),
                    superseded_by = COALESCE(superseded_by, ?),
                    metadata_json = json_set(
                        COALESCE(metadata_json, '{}'),
                        '$.status', 'expired',
                        '$.expire_reason', ?
                    ),
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    now,
                    superseded_by,
                    decision.reason or "expired by candidate",
                    now,
                    decision.target_memory_id,
                ),
            )
        result["memory_id"] = decision.target_memory_id
        result["expired_id"] = decision.target_memory_id
        return result

    # P1-10: ARCHIVE — neighbor is stale. Soft-delete (archived_at) with the
    # archive_reason carrying the rule that fired, and stash a structured
    # marker in metadata_json so audits can reconstruct *why*.
    if decision.action == "ARCHIVE" and decision.target_memory_id:
        with db.conn:
            db.conn.execute(
                """
                UPDATE memories
                SET archived_at = COALESCE(archived_at, ?),
                    archive_reason = COALESCE(archive_reason, ?),
                    metadata_json = json_set(
                        COALESCE(metadata_json, '{}'),
                        '$.archive_rule', ?,
                        '$.archived_at', ?
                    ),
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    now,
                    decision.reason or "archived (lifecycle decision)",
                    decision.reason or "stale",
                    now,
                    now,
                    decision.target_memory_id,
                ),
            )
        result["memory_id"] = decision.target_memory_id
        result["archived_id"] = decision.target_memory_id
        return result

    return result


def _resolve_supersedor_id(db: MemoirsDB, candidate: Candidate) -> str | None:
    """Best-effort: find the active memory id matching ``candidate.content``.

    Used by the EXPIRE branch when the candidate did not just ADD a row in
    this same call (e.g. an EXPIRE issued as a primary action). Returns
    ``None`` when nothing matches — superseded_by stays NULL, which is fine.
    """
    try:
        row = db.conn.execute(
            "SELECT id FROM memories "
            "WHERE content_hash = ? AND archived_at IS NULL "
            "ORDER BY created_at DESC LIMIT 1",
            (content_hash(candidate.content),),
        ).fetchone()
    except Exception:
        return None
    return row["id"] if row else None


def consolidate_candidate(db: MemoirsDB, candidate_row: dict) -> dict:
    """Take a pending memory_candidates row, decide, apply, mark candidate."""
    cand = Candidate(
        type=candidate_row["type"],
        content=candidate_row["content"],
        importance=int(candidate_row["importance"]),
        confidence=float(candidate_row["confidence"]),
        entities=json.loads(candidate_row["entities"] or "[]"),
        source_message_ids=json.loads(candidate_row["source_message_ids"] or "[]"),
        extractor=candidate_row.get("extractor") or "heuristic",
    )
    decision = decide_memory_action(db, cand)
    result = apply_decision(db, cand, decision)
    new_status = {
        "ADD": "accepted",
        "UPDATE": "merged",
        "MERGE": "merged",
        "IGNORE": "rejected",
        "REJECT": "rejected",
        "CONTRADICTION": "merged",
        "EXPIRE": "rejected",
        "ARCHIVE": "rejected",
    }.get(decision.action, "rejected")
    promoted = result.get("memory_id")
    with db.conn:
        db.conn.execute(
            """
            UPDATE memory_candidates
            SET status = ?, rejection_reason = ?, promoted_memory_id = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                new_status,
                decision.reason if new_status == "rejected" else None,
                promoted,
                utc_now(),
                candidate_row["id"],
            ),
        )
    snippet = candidate_row["content"][:80].replace("\n", " ")
    log.info(
        "consolidate cand=%s type=%s action=%s mem=%s — %s",
        candidate_row["id"][:16],
        candidate_row["type"],
        decision.action,
        (promoted or "-")[:16],
        snippet,
    )
    if decision.reason and decision.action != "ADD":
        log.debug("  reason: %s", decision.reason)
    return {"candidate_id": candidate_row["id"], **result}


def consolidate_pending(db: MemoirsDB, *, limit: int = 100) -> dict:
    rows = db.conn.execute(
        """
        SELECT * FROM memory_candidates
        WHERE status = 'pending'
        ORDER BY importance DESC, confidence DESC, created_at ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    results = [consolidate_candidate(db, dict(r)) for r in rows]
    counts: dict[str, int] = {}
    for r in results:
        counts[r["action"]] = counts.get(r["action"], 0) + 1
    return {"processed": len(results), "by_action": counts}


# ----------------------------------------------------------------------
# 5.2 Lifecycle
# ----------------------------------------------------------------------


def archive_low_value_memories(
    db: MemoirsDB,
    *,
    score_threshold: float | None = None,
    min_age_days: int = 30,
    percentile: float = 0.10,
) -> int:
    """Archive memorias whose score is in the bottom `percentile` AND older
    than `min_age_days`.

    By default uses a dynamic p10 threshold computed from the current corpus —
    avoids killing memorias of types that naturally score lower (e.g.
    `style` tends to have lower importance than `decision`).

    Pass `score_threshold` explicitly to override with a static cutoff.
    """
    cutoff = (datetime.now(timezone.utc).timestamp() - min_age_days * 86400)
    if score_threshold is None:
        # Compute the percentile on the corpus dynamically
        all_scores = [r[0] for r in db.conn.execute(
            "SELECT score FROM memories WHERE archived_at IS NULL ORDER BY score ASC"
        ).fetchall()]
        if not all_scores:
            return 0
        idx = int(percentile * len(all_scores))
        score_threshold = all_scores[idx] if idx < len(all_scores) else all_scores[-1]
        log.info("archive_low_value: dynamic p%d threshold = %.3f (n=%d)",
                 int(percentile * 100), score_threshold, len(all_scores))
    rows = db.conn.execute(
        "SELECT id, score, created_at FROM memories WHERE archived_at IS NULL AND score < ?",
        (score_threshold,),
    ).fetchall()
    n = 0
    now = utc_now()
    with db.conn:
        for r in rows:
            try:
                ts = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")).timestamp()
            except (ValueError, AttributeError):
                continue
            if ts > cutoff:
                continue
            db.conn.execute(
                "UPDATE memories SET archived_at = ?, archive_reason = ? WHERE id = ?",
                (now, f"low score ({r['score']:.3f})", r["id"]),
            )
            n += 1
    return n


def expire_old_memories(db: MemoirsDB) -> int:
    """Mark memories whose valid_to has passed as archived."""
    now = utc_now()
    rows = db.conn.execute(
        "SELECT id FROM memories WHERE archived_at IS NULL AND valid_to IS NOT NULL AND valid_to < ?",
        (now,),
    ).fetchall()
    with db.conn:
        for r in rows:
            db.conn.execute(
                "UPDATE memories SET archived_at = ?, archive_reason = 'valid_to expired' WHERE id = ?",
                (now, r["id"]),
            )
    return len(rows)


def run_daily_maintenance(db: MemoirsDB) -> dict:
    """End-to-end maintenance pass: scores → expiration → low-value archive →
    promote/demote → near-duplicate merge. Each step is idempotent and safe to
    re-run.
    """
    from . import lifecycle as lc

    result = {
        "scores_updated": recompute_all_scores(db),
        "expired": expire_old_memories(db),
        "archived_low_value": archive_low_value_memories(db),
        "promoted": lc.promote_all(db),
        "demoted": lc.demote_all(db),
    }
    # Auto-merge near-duplicates only when embeddings are usable
    try:
        merge_result = lc.auto_merge_near_duplicates(db)
        result["merged_dups"] = merge_result["merged"]
        result["contradictions_flagged"] = merge_result["contradictions"]
    except Exception as e:
        log.warning("auto_merge_near_duplicates failed: %s", e)
        result["merged_dups"] = 0
        result["contradictions_flagged"] = 0
    return result


# ----------------------------------------------------------------------
# 5.4 Temporal versioning
# ----------------------------------------------------------------------


def create_memory_version(db: MemoirsDB, old_memory_id: str, new_content: str) -> str:
    """Append-only update: archive the old memory, insert a new one that supersedes it."""
    now = utc_now()
    old = db.conn.execute("SELECT * FROM memories WHERE id = ?", (old_memory_id,)).fetchone()
    if not old:
        raise ValueError(f"memory not found: {old_memory_id}")
    old = dict(old)
    new_id = stable_id("mem", old["type"], new_content, now)
    h = content_hash(new_content)
    with db.conn:
        db.conn.execute(
            """
            INSERT INTO memories (
                id, type, content, content_hash, importance, confidence,
                score, usage_count, user_signal, valid_from, metadata_json,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?, '{}', ?, ?)
            """,
            (new_id, old["type"], new_content, h, old["importance"], old["confidence"], old["user_signal"], now, now, now),
        )
        db.conn.execute(
            "UPDATE memories SET valid_to = ?, superseded_by = ?, updated_at = ? WHERE id = ?",
            (now, new_id, now, old_memory_id),
        )
    emb.upsert_memory_embedding(db, new_id, new_content)
    return new_id


# ----------------------------------------------------------------------
# 5.6 Reasoning — assemble_context (THE function)
# ----------------------------------------------------------------------


def _detect_conflicting(memories: list[dict]) -> list[tuple[dict, dict]]:
    """Find pairs of memories that look contradictory (same type, similar content).

    Pre-filter (cheap heuristic): same type + jaccard-ish closeness via the
    `similarity` already in each row. The expensive Gemma check happens in
    `_resolve_conflicts` only on these candidates.

    Returns pairs (a, b) where both are flagged as same-topic same-type and
    might be contradictory. The actual contradiction decision comes later.
    """
    conflicts: list[tuple[dict, dict]] = []
    NEGATIONS = (" no ", " not ", " never ", " sin ", " nunca ", " doesn't ", " isn't ", " avoid ")
    for i, a in enumerate(memories):
        for b in memories[i + 1:]:
            if a["type"] != b["type"]:
                continue
            text_a = f" {a['content'].lower()} "
            text_b = f" {b['content'].lower()} "
            a_has_neg = any(n in text_a for n in NEGATIONS)
            b_has_neg = any(n in text_b for n in NEGATIONS)
            opposite_polarity = a_has_neg != b_has_neg
            sim = min(float(a.get("similarity", 0.5)), float(b.get("similarity", 0.5)))
            # Be more inclusive: any same-type pair with sim >= 0.5 is a candidate.
            # Gemma will be the judge in _resolve_conflicts.
            if opposite_polarity and sim >= 0.4:
                conflicts.append((a, b))
            elif sim >= 0.7:
                # Even same polarity, very similar → check with Gemma
                conflicts.append((a, b))
    return conflicts


def _resolve_conflicts(memories: list[dict], conflicts: list[tuple[dict, dict]]) -> list[dict]:
    """Resolve conflicts using the curator LLM when available, score fallback otherwise.

    Perf guard (root-cause of the p95 60s outlier observed in bench):
    retrieval must stay sub-100ms, but each curator call costs 500-2000ms.
    With many conflict pairs that compounded into multi-second latency. Now:
      - default OFF for retrieval; opt-in via ``MEMOIRS_RETRIEVAL_CURATOR=on``
        (legacy ``MEMOIRS_RETRIEVAL_GEMMA`` still honored).
      - capped at ``MEMOIRS_RETRIEVAL_CURATOR_MAX`` calls per query (default 2;
        legacy ``MEMOIRS_RETRIEVAL_GEMMA_MAX`` honored as fallback).
    Deeper conflict reasoning belongs in the sleep cron, not the hot path.
    """
    import os
    drop: set[str] = set()

    flag = os.environ.get("MEMOIRS_RETRIEVAL_CURATOR")
    if flag is None:
        flag = os.environ.get("MEMOIRS_RETRIEVAL_GEMMA", "off")
    use_curator = flag.lower() not in ("off", "0", "false", "no")
    cap_raw = os.environ.get("MEMOIRS_RETRIEVAL_CURATOR_MAX")
    if cap_raw is None:
        cap_raw = os.environ.get("MEMOIRS_RETRIEVAL_GEMMA_MAX", "2")
    try:
        max_calls = max(0, int(cap_raw))
    except ValueError:
        max_calls = 2

    have_curator = False
    _curator = None
    if use_curator and max_calls > 0:
        try:
            from . import curator as _curator  # lazy: don't load model unless needed
            have_curator = _curator._have_curator()
        except Exception:
            have_curator = False

    curator_calls = 0
    for a, b in conflicts:
        if a["id"] in drop or b["id"] in drop:
            continue
        if have_curator and curator_calls < max_calls:
            try:
                verdict = _curator.curator_resolve_conflict(a["content"], b["content"])
            except Exception as e:  # never block retrieval
                log.debug("curator_resolve_conflict raised in retrieval: %s", e)
                verdict = None
            curator_calls += 1
            if verdict and verdict.get("contradictory") and verdict.get("winner") in ("A", "B"):
                loser = b if verdict["winner"] == "A" else a
                drop.add(loser["id"])
                log.info("conflict resolve curator: a=%s b=%s winner=%s drop=%s reason=%s",
                         a["id"][:12], b["id"][:12], verdict["winner"], loser["id"][:12],
                         (verdict.get("reason") or "")[:60])
                continue
            if verdict and not verdict.get("contradictory"):
                continue  # compatible — keep both
        # Fallback: prefer higher score (fast path — always taken in default config)
        loser = a if (a.get("score", 0) < b.get("score", 0)) else b
        drop.add(loser["id"])
    return [m for m in memories if m["id"] not in drop]


def _compress_context(memories: list[dict], max_chars: int = 4000) -> list[str]:
    out: list[str] = []
    used = 0
    for m in memories:
        line = f"[{m['type']}] {m['content']}"
        if used + len(line) > max_chars:
            break
        out.append(line)
        used += len(line) + 1
    return out


def _summary_for(memory: dict, *, max_chars: int = 140) -> str:
    """Short preview of a memory's content for streaming events."""
    content = (memory.get("content") or "").strip().replace("\n", " ")
    if len(content) <= max_chars:
        return content
    return content[: max_chars - 1].rstrip() + "…"


def _resolve_retrieval_mode(explicit: str | None) -> str:
    """Pick the retrieval mode: explicit arg > env var > default 'hybrid'.

    Valid values: 'hybrid' (BM25 + dense fused via RRF), 'dense' (cosine
    only — legacy behavior), 'bm25' (lexical only), 'graph' / 'hybrid_graph'
    (PPR multi-hop), 'raptor' / 'hybrid_raptor' (RAPTOR tree-descent).
    Unknown values fall back to hybrid with a warning.
    """
    import os
    mode = (explicit or os.environ.get("MEMOIRS_RETRIEVAL_MODE") or "hybrid").lower()
    if mode not in {
        "hybrid", "dense", "bm25", "graph", "hybrid_graph",
        "raptor", "hybrid_raptor",
    }:
        log.warning("unknown MEMOIRS_RETRIEVAL_MODE=%r — using hybrid", mode)
        return "hybrid"
    return mode


def _hydrate_scope_columns(db: MemoirsDB, candidates: list[dict]) -> None:
    """Best-effort backfill of scoping columns on rows hydrated by retrieval
    backends that don't SELECT them (e.g. ``hybrid_retrieval.hydrate_memories``,
    ``embeddings.search_similar_memories``). Keeps :class:`ScopeFilter` and
    :func:`acl.can_read` working without forcing every backend to learn
    about the new columns.

    Mutates each dict in-place. Silently no-ops if migration 008 hasn't been
    applied yet (older test fixtures).
    """
    if not candidates:
        return
    missing_ids = [
        c["id"] for c in candidates
        if "user_id" not in c or "visibility" not in c
    ]
    if not missing_ids:
        return
    placeholders = ",".join("?" * len(missing_ids))
    try:
        rows = db.conn.execute(
            f"SELECT id, user_id, agent_id, run_id, namespace, visibility "
            f"FROM memories WHERE id IN ({placeholders})",
            missing_ids,
        ).fetchall()
    except Exception:
        return
    extra = {r["id"]: dict(r) for r in rows}
    for c in candidates:
        if c["id"] in extra:
            for col in ("user_id", "agent_id", "run_id", "namespace", "visibility"):
                c.setdefault(col, extra[c["id"]].get(col))


def _resolve_requester_scope(
    scope: Scope | None,
    scope_filter: ScopeFilter | None,
) -> Scope:
    """Pick the :class:`Scope` to use for ACL ``can_read`` checks.

    Priority (highest first):
      1. Explicit ``scope`` arg, when not the default sentinel.
      2. The first ``user_id`` listed in ``scope_filter.user_ids`` (callers
         that pass ``scope_filter(user_ids={"alice"})`` are effectively
         declaring the requester identity even without setting ``scope``).
      3. ``MEMOIRS_USER_ID`` env, if set.
      4. :data:`DEFAULT_SCOPE` (``user_id='local'``) — the legacy single-user
         path.
    """
    if scope is not None and scope is not DEFAULT_SCOPE:
        return scope
    if scope_filter is not None and scope_filter.user_ids:
        # Pick a deterministic representative — callers that filter on a
        # single user_id (the common case) get exactly that identity.
        uid = sorted(scope_filter.user_ids)[0]
        return Scope(user_id=uid)
    import os
    env_uid = os.environ.get("MEMOIRS_USER_ID")
    if env_uid:
        return Scope(user_id=env_uid)
    return DEFAULT_SCOPE


def _retrieve_candidates(
    db: MemoirsDB,
    query: str,
    *,
    top_k: int,
    as_of: str | None,
    mode: str,
    scope_filter: ScopeFilter | None = None,
    scope: Scope | None = None,
) -> list[dict]:
    """Dispatch to the requested retrieval backend. Always returns rows shaped
    like `embeddings.search_similar_memories` (id, type, content, score,
    similarity, …) so the rest of `assemble_context_stream` doesn't care.

    P-perf: when ``mode == "hybrid"`` and the query is trivial (single
    keyword, only stop-words, …), the dense leg is skipped entirely so we
    avoid the embed call. Explicit ``mode="dense"`` is honored verbatim —
    callers that ask for dense get dense.

    Scoping (P0-5 / P3-3): pass a non-None ``scope_filter`` to drop rows
    that don't satisfy the requested ``user_ids`` / ``agent_ids`` /
    ``namespaces`` / ``visibilities`` constraints. Filtering happens
    post-fetch so every retrieval backend benefits without touching their
    SQL.

    ACL wiring (Fase 5A): after ``scope_filter`` runs, every surviving
    candidate is checked with :func:`acl.can_read` against ``scope`` (or a
    requester derived from ``scope_filter`` / ``MEMOIRS_USER_ID`` / the
    default). Memories the requester cannot read are dropped silently so
    private rows from another tenant never leak into a context payload —
    even when they happened to share a vocabulary with the query. A
    fast-path skips the per-row check entirely when every candidate is
    owned by ``"local"`` (single-user mode), so the legacy hot path is
    untouched.
    """
    if mode == "dense":
        candidates = emb.search_similar_memories(db, query, top_k=top_k, as_of=as_of)
    else:
        from . import hybrid_retrieval as hr
        if mode == "bm25":
            bm25 = hr.bm25_search(db.conn, query, top_k=top_k, as_of=as_of)
            fused = [
                {"id": mid, "score": s, "bm25_rank": i, "dense_rank": None,
                 "bm25_score": s, "dense_score": None}
                for i, (mid, s) in enumerate(bm25, start=1)
            ]
            candidates = hr.hydrate_memories(db, fused, as_of=as_of)
        elif mode in {"graph", "hybrid_graph"}:
            # PPR multi-hop retrieval (HippoRAG 2-inspired). Both modes hydrate
            # via graph_retrieval.hydrate_memories so downstream sees the same
            # row shape (similarity slot carries the PPR/RRF score).
            from . import graph_retrieval as gr
            if mode == "graph":
                pairs = gr.graph_search(db, query, top_k=top_k)
            else:
                pairs = gr.hybrid_graph_search(db, query, top_k=top_k, as_of=as_of)
            candidates = gr.hydrate_memories(db, pairs, as_of=as_of)
        elif mode in {"raptor", "hybrid_raptor"}:
            # RAPTOR tree-descent retrieval (P1-6). When no tree has been built
            # yet, the raptor leg degrades to an empty list (with a warning) —
            # so ``raptor`` returns [] and ``hybrid_raptor`` falls back to the
            # pure hybrid ranking. We never raise on missing trees.
            from . import raptor as rp
            from . import graph_retrieval as gr
            rp.ensure_schema(db)
            tree_row = db.conn.execute(
                "SELECT 1 FROM summary_nodes LIMIT 1"
            ).fetchone()
            tree_built = tree_row is not None
            raptor_pairs: list[tuple[str, float]] = []
            if tree_built:
                try:
                    q_emb = emb.embed_text_cached(query)
                except emb.EmbeddingsUnavailable:
                    log.warning(
                        "retrieval_mode=%s: embeddings unavailable — raptor leg empty",
                        mode,
                    )
                    q_emb = None
                if q_emb is not None:
                    try:
                        raw = rp.retrieve_raptor(db, q_emb, top_k=top_k)
                    except Exception as e:  # pragma: no cover -- defensive
                        log.warning("retrieve_raptor failed (%s) — leg empty", e)
                        raw = []
                    # ``retrieve_raptor`` returns (memory_id, score, level, path).
                    raptor_pairs = [(mid, float(score)) for mid, score, _l, _p in raw]
            else:
                log.warning(
                    "retrieval_mode=%s: no RAPTOR tree built (run `memoirs raptor "
                    "build`); raptor leg returns [].",
                    mode,
                )

            if mode == "raptor":
                candidates = gr.hydrate_memories(db, raptor_pairs, as_of=as_of)
            else:
                # hybrid_raptor: RRF-fuse hybrid_search with raptor results, the
                # same shape as hybrid_graph_search. When the raptor leg is
                # empty we silently degrade to pure hybrid.
                use_dense = emb.should_use_dense(query)
                over_k = max(top_k * 2, 20)
                try:
                    hybrid_results = hr.hybrid_search(
                        db, query, top_k=over_k, as_of=as_of,
                        with_embeddings=use_dense,
                    )
                except Exception as exc:
                    log.warning(
                        "hybrid_raptor: hybrid leg failed (%s) — raptor-only", exc,
                    )
                    hybrid_results = []
                hybrid_pairs = [
                    (r["id"], float(r.get("score", 0.0))) for r in hybrid_results
                ]
                if not hybrid_pairs and not raptor_pairs:
                    candidates = []
                else:
                    rrf_k = 60
                    fused: dict[str, float] = {}
                    for ranking in (hybrid_pairs, raptor_pairs):
                        for rank, (mid, _s) in enumerate(ranking, start=1):
                            fused[mid] = fused.get(mid, 0.0) + 1.0 / (rrf_k + rank)
                    fused_pairs = sorted(
                        fused.items(), key=lambda kv: kv[1], reverse=True
                    )[:top_k]
                    candidates = gr.hydrate_memories(db, fused_pairs, as_of=as_of)
        else:
            # hybrid (default) — skip the dense leg entirely for trivial queries
            # (single keyword, only stop-words, …) to avoid the embed call.
            use_dense = emb.should_use_dense(query)
            fused = hr.hybrid_search(
                db, query, top_k=top_k, as_of=as_of, with_embeddings=use_dense
            )
            candidates = hr.hydrate_memories(db, fused, as_of=as_of)
    # Scoping (P0-5 / P3-3): hydrate scope columns and apply the optional
    # post-fetch ScopeFilter. We do this BEFORE record_access so denied rows
    # don't get their last_accessed_at bumped — otherwise an unauthorized
    # query would still leak retrieval-side-effect signal across tenants.
    if scope_filter is not None and not scope_filter.is_empty():
        _hydrate_scope_columns(db, candidates)
        candidates = [c for c in candidates if scope_filter.matches(c)]
    # ACL (Fase 5A): apply per-row can_read on the surviving candidates.
    # Fast-path: when every row is owned by ``"local"`` we are in single-user
    # mode and ACL is a no-op — skip the import + per-row predicate to keep
    # the hot path overhead-free for the common case.
    if candidates:
        _hydrate_scope_columns(db, candidates)
        owners = {(c.get("user_id") or "local") for c in candidates}
        if owners != {"local"}:
            from . import acl as _acl
            requester = _resolve_requester_scope(scope, scope_filter)
            candidates = [
                c for c in candidates
                if _acl.can_read(c, requester, conn=db.conn)
            ]
    # P1-7 hook: Ebbinghaus reinforcement. Each retrieved memory gets its
    # ``last_accessed_at`` bumped and its ``strength`` multiplied by 1.5.
    # Skipped on time-travel queries (``as_of`` is set) so audits don't
    # mutate the corpus. Failures are swallowed inside ``record_access``.
    if as_of is None and candidates:
        for _m in candidates:
            mid = _m.get("id")
            if mid:
                record_access(db, mid)
        try:
            db.conn.commit()
        except Exception:
            pass
    return candidates


def _apply_hyde(query: str) -> tuple[str, dict[str, Any] | None]:
    """Stage 1 — query expansion (HyDE). Returns ``(query_to_use, info|None)``.

    Off by default (env ``MEMOIRS_HYDE``); when on, replaces the query string
    fed into ``_retrieve_candidates`` with the expanded ``combined`` field
    so both BM25 and dense legs benefit. The original query is preserved in
    the meta payload for observability.
    """
    from . import hyde as _hyde
    if not _hyde.is_enabled():
        return query, None
    expanded = _hyde.expand_query(query)
    if expanded.is_empty() or not expanded.combined:
        return query, None
    info = {
        "backend": expanded.backend,
        "keywords": expanded.keywords,
        "hypothetical_doc": expanded.hypothetical_doc,
    }
    return expanded.combined, info


ENV_PRF = "MEMOIRS_PRF"
ENV_PRF_TOPN = "MEMOIRS_PRF_TOPN"


def _rrf_fuse(lists: list[list[dict]], *, k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion across ranked candidate lists keyed by ``id``.

    Each row's RRF score is ``sum_l 1/(k + rank_l)`` over the lists in
    which it appears. The first occurrence of each id is kept as the
    canonical row (so ``similarity``, ``score``, …, survive for the
    downstream rerank/MMR stages); only the ``score`` field is replaced
    by the fused RRF score so the rest of the pipeline picks it up.
    """
    rrf: dict[str, float] = {}
    canonical: dict[str, dict] = {}
    for ranked in lists:
        for rank, c in enumerate(ranked, start=1):
            mid = c["id"]
            rrf[mid] = rrf.get(mid, 0.0) + 1.0 / (k + rank)
            canonical.setdefault(mid, c)
    out: list[dict] = []
    for mid, score in sorted(rrf.items(), key=lambda kv: -kv[1]):
        row = dict(canonical[mid])
        row["score"] = float(score)
        out.append(row)
    return out


def _apply_prf(
    db: MemoirsDB,
    query: str,
    initial: list[dict],
    *,
    top_k: int,
    as_of: str | None,
    mode: str,
    scope_filter: ScopeFilter | None,
    scope: Scope | None,
) -> tuple[list[dict], dict[str, Any] | None]:
    """Stage 2.5 — Pseudo-Relevance Feedback for multi-hop bridging.

    Multi-hop queries (e.g. "how does the team that owns project apollo
    handle on-call?") often retrieve the bridge memory at rank 1
    ("apollo is owned by squad X") but miss the answer memory ("squad X
    runs on-call via Y") because the original query has no lexical or
    semantic overlap with the answer. PRF expands the query with the
    content of the top-N candidates and runs a second retrieval pass,
    then RRF-fuses both rankings.

    Off by default (env ``MEMOIRS_PRF``). ``MEMOIRS_PRF_TOPN`` controls
    how many anchor docs feed the expansion (default 1 — adding more
    helps recall on harder hops at a small cost to precision).
    """
    if os.environ.get(ENV_PRF, "off").strip().lower() not in {"on", "1", "true"}:
        return initial, None
    if not initial:
        return initial, None
    try:
        n_anchor = max(1, int(os.environ.get(ENV_PRF_TOPN, "1")))
    except ValueError:
        n_anchor = 1
    anchors = [
        (c.get("content") or "").strip()
        for c in initial[:n_anchor]
    ]
    anchors = [a for a in anchors if a]
    if not anchors:
        return initial, None
    expanded = (query + " " + " ".join(anchors)).strip()
    try:
        pass2 = _retrieve_candidates(
            db, expanded, top_k=top_k, as_of=as_of, mode=mode,
            scope_filter=scope_filter, scope=scope,
        )
    except Exception as e:
        log.warning("PRF pass2 failed: %s — using pass1 only", e)
        return initial, None
    fused = _rrf_fuse([initial, pass2])[:top_k]
    return fused, {"n_anchor": len(anchors), "pass2_count": len(pass2)}


def _apply_reranker(query: str, candidates: list[dict]) -> tuple[list[dict], str]:
    """Stage 3 — cross-encoder reranking. Returns ``(candidates, backend_name)``.

    No-op when the configured backend is ``NoopReranker`` — that path
    preserves both order and scores.
    """
    from . import reranker as _rk
    rk = _rk.get_reranker()
    if isinstance(rk, _rk.NoopReranker):
        return candidates, rk.name
    return _rk.apply_rerank(query, candidates, reranker=rk), rk.name


def _apply_mmr(
    db: MemoirsDB, candidates: list[dict], k: int
) -> tuple[list[dict], bool]:
    """Stage 4 — MMR diversification. Returns ``(candidates, applied)``.

    Pulls embeddings from ``memory_embeddings`` on demand (cached per call)
    so MMR works even when the candidate row didn't carry the vector.
    """
    from . import mmr as _mmr
    if not _mmr.is_enabled():
        return candidates, False
    if len(candidates) <= k:
        return candidates, False

    cache: dict[str, list[float] | None] = {}

    def lookup(memory_id: str) -> list[float] | None:
        if memory_id in cache:
            return cache[memory_id]
        try:
            row = db.conn.execute(
                "SELECT embedding, dim FROM memory_embeddings WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        except Exception:
            cache[memory_id] = None
            return None
        if not row or not row["embedding"]:
            cache[memory_id] = None
            return None
        try:
            vec = emb._unpack(bytes(row["embedding"]), int(row["dim"]))
        except Exception:
            cache[memory_id] = None
            return None
        cache[memory_id] = vec
        return vec

    selected = _mmr.mmr_select(
        candidates, k, lambda_=_mmr.get_lambda(), embedding_lookup=lookup
    )
    return selected, True


def assemble_context_stream(
    db: MemoirsDB,
    query: str,
    *,
    top_k: int = 20,
    max_lines: int = 15,
    as_of: str | None = None,
    retrieval_mode: str | None = None,
    scope: Scope = DEFAULT_SCOPE,
    scope_filter: ScopeFilter | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Generator variant of `assemble_context`. Yields tagged events:

      ("meta",    {"query","as_of","live","retrieval_mode"})  — emitted first
      ("memory",  {"id","type","score","similarity","summary"})  — once per ranked memory
      ("context", {"context","memories","token_estimate","conflicts_resolved","as_of","live"})
                                                       — emitted last, full payload

    Designed for SSE streaming. The non-streaming `assemble_context` consumes
    this generator and returns the trailing `context` payload.

    Side effects (usage_count++) happen exactly as in the non-streaming version:
    only when `as_of is None` (live), and only for memories that survive
    conflict resolution and fit under `max_lines`.

    `retrieval_mode` ∈ {"hybrid","dense","bm25"} (default: env
    `MEMOIRS_RETRIEVAL_MODE`, else "hybrid"). Hybrid combines BM25 + dense
    via Reciprocal Rank Fusion and is recommended.

    Pipeline (each stage is opt-in via env, except MMR which defaults on)::

        query
          → [HyDE expand]            (MEMOIRS_HYDE=on)
          → _retrieve_candidates
          → [Reranker]               (MEMOIRS_RERANKER_BACKEND≠none)
          → [MMR]                    (MEMOIRS_MMR=on, default ON)
          → score+conflict resolve
          → emit
    """
    live = as_of is None
    mode = _resolve_retrieval_mode(retrieval_mode)

    # Stage 1 — HyDE query expansion (opt-in, default off).
    retrieval_query, hyde_info = _apply_hyde(query)

    # Yield meta IMMEDIATELY — gives clients TTFT well below retrieval latency.
    meta_payload: dict[str, Any] = {
        "query": query,
        "as_of": as_of,
        "live": live,
        "retrieval_mode": mode,
    }
    if hyde_info is not None:
        meta_payload["hyde"] = hyde_info
    yield "meta", meta_payload

    # Stage 2 — Retrieval. Dispatch by mode (default: hybrid BM25 + dense via RRF).
    # ``scope_filter`` is applied post-fetch inside ``_retrieve_candidates``.
    candidates = _retrieve_candidates(
        db, retrieval_query, top_k=top_k, as_of=as_of, mode=mode,
        scope_filter=scope_filter, scope=scope,
    )
    n_in = len(candidates)

    # Stage 2.5 — Pseudo-Relevance Feedback (multi-hop bridging, opt-in).
    candidates, prf_info = _apply_prf(
        db, retrieval_query, candidates, top_k=top_k, as_of=as_of,
        mode=mode, scope_filter=scope_filter, scope=scope,
    )

    # Stage 3 — Cross-encoder reranking (opt-in). Mutates ``score`` in-place
    # so the existing ``combined`` formula picks up the new ranking.
    candidates, rerank_backend = _apply_reranker(retrieval_query, candidates)

    for m in candidates:
        m["combined"] = round(
            m.get("similarity", 0.0) * 0.6 + float(m.get("score", 0.0)) * 0.4, 4
        )
    candidates.sort(key=lambda m: m["combined"], reverse=True)

    # Stage 4 — MMR diversification (default on). Re-orders the top-K so we
    # don't return five near-duplicate memories. Only kicks in when we have
    # more candidates than we can ship.
    candidates, mmr_applied = _apply_mmr(db, candidates, max_lines)

    log.debug(
        "retrieval_pipeline hyde=%s prf=%s rerank=%s mmr=%s in=%d out=%d",
        "on" if hyde_info is not None else "off",
        "on" if prf_info is not None else "off",
        rerank_backend,
        "on" if mmr_applied else "off",
        n_in,
        min(len(candidates), max_lines),
    )

    # Detect + resolve conflicts before streaming, so the order we stream
    # matches the order in the final `context` payload.
    conflicts = _detect_conflicting(candidates)
    resolved = _resolve_conflicts(candidates, conflicts)
    chosen = resolved[:max_lines]

    # Stream each surviving memory progressively. This is what the chat UI
    # renders as the "memory loading…" list.
    for m in chosen:
        yield "memory", {
            "id": m["id"],
            "type": m["type"],
            "score": m.get("score"),
            "similarity": m.get("similarity"),
            "summary": _summary_for(m),
        }

    # Side-effects (usage tracking) happen on the live path only.
    if live:
        now = utc_now()
        with db.conn:
            for m in chosen:
                db.conn.execute(
                    "UPDATE memories SET usage_count = usage_count + 1, last_used_at = ? WHERE id = ?",
                    (now, m["id"]),
                )

    # Procedural memories are agent-style instructions ("when X, do Y"). They
    # are surfaced in their own field so the caller can inject them as system
    # prompt text, not mixed into the fact list. We also pull all currently-
    # active procedural memorias (regardless of retrieval ranking) so an
    # agent's persistent policies survive when the query doesn't lexically
    # match them.
    procedural_chosen = [m for m in chosen if m.get("type") == "procedural"]
    fact_chosen = [m for m in chosen if m.get("type") != "procedural"]
    procedural_extra: list[dict] = []
    if live:
        try:
            extra_rows = db.conn.execute(
                "SELECT id, type, content, importance, score "
                "FROM memories WHERE type = 'procedural' AND archived_at IS NULL "
                "ORDER BY importance DESC, score DESC LIMIT 20"
            ).fetchall()
            seen = {m["id"] for m in procedural_chosen}
            for row in extra_rows:
                if row["id"] in seen:
                    continue
                procedural_extra.append({
                    "id": row["id"],
                    "type": row["type"],
                    "content": row["content"],
                    "importance": row["importance"],
                    "score": row["score"],
                })
        except Exception:
            pass

    context_lines = _compress_context(fact_chosen)
    token_estimate = sum(len(l) for l in context_lines) // 4
    payload: dict[str, Any] = {
        "context": context_lines,
        "memories": [
            {
                "id": m["id"],
                "type": m["type"],
                "score": m.get("score"),
                "similarity": m.get("similarity"),
            }
            for m in fact_chosen
        ],
        "token_estimate": token_estimate,
        "conflicts_resolved": len(conflicts),
        "as_of": as_of,
        "live": live,
    }
    procedural_all = procedural_chosen + procedural_extra
    if procedural_all:
        payload["system_instructions"] = [
            {
                "id": m["id"],
                "content": m.get("content", _summary_for(m)),
                "importance": m.get("importance"),
            }
            for m in procedural_all
        ]
    yield "context", payload


def assemble_context(
    db: MemoirsDB,
    query: str,
    *,
    top_k: int = 20,
    max_lines: int = 15,
    as_of: str | None = None,
    retrieval_mode: str | None = None,
    scope: Scope = DEFAULT_SCOPE,
    scope_filter: ScopeFilter | None = None,
) -> dict:
    """Reduce a long conversation to compact, ranked, conflict-resolved memory.

    Pass `as_of=<ISO timestamp>` to ask "what would the system have answered at
    moment t?" — useful for auditing or re-creating context at a past point.
    Live queries (default) increment `usage_count`; time-travel queries do NOT
    mutate the DB.

    `retrieval_mode` ∈ {"hybrid","dense","bm25"} (default: env
    `MEMOIRS_RETRIEVAL_MODE`, else "hybrid").

    Scoping (P0-5 / P3-3): pass ``scope_filter`` to restrict retrieval to
    specific ``user_ids`` / ``agent_ids`` / ``namespaces`` / ``visibilities``.
    Defaults to ``None`` so legacy callers see every row exactly as before.

    Thin wrapper over `assemble_context_stream` — preserves the original API
    by draining the generator and returning the final `context` payload.
    """
    final: dict[str, Any] = {}
    for event, data in assemble_context_stream(
        db, query, top_k=top_k, max_lines=max_lines, as_of=as_of,
        retrieval_mode=retrieval_mode, scope=scope, scope_filter=scope_filter,
    ):
        if event == "context":
            final = data
    return final


# ----------------------------------------------------------------------
# 5.7 Event-driven ingestion
# ----------------------------------------------------------------------


def enqueue_event(db: MemoirsDB, event_type: str, payload: dict) -> int:
    cur = db.conn.execute(
        "INSERT INTO event_queue (event_type, payload_json, status, created_at) VALUES (?, ?, 'pending', ?)",
        (event_type, json.dumps(payload, ensure_ascii=False), utc_now()),
    )
    db.conn.commit()
    return int(cur.lastrowid)


def process_event_queue(db: MemoirsDB, *, limit: int = 50) -> dict:
    """Drain queue: trigger extraction for closed conversations / file imports."""
    rows = db.conn.execute(
        "SELECT id, event_type, payload_json FROM event_queue WHERE status = 'pending' ORDER BY id LIMIT ?",
        (limit,),
    ).fetchall()
    processed = 0
    failed = 0
    for r in rows:
        try:
            payload = json.loads(r["payload_json"])
            _dispatch_event(db, r["event_type"], payload)
            db.conn.execute(
                "UPDATE event_queue SET status = 'done', processed_at = ? WHERE id = ?",
                (utc_now(), r["id"]),
            )
            processed += 1
        except Exception as e:
            log.exception("event %s failed", r["id"])
            db.conn.execute(
                "UPDATE event_queue SET status = 'failed', error = ?, processed_at = ? WHERE id = ?",
                (str(e), utc_now(), r["id"]),
            )
            failed += 1
        db.conn.commit()
    return {"processed": processed, "failed": failed}


def _dispatch_event(db: MemoirsDB, event_type: str, payload: dict) -> None:
    from . import curator as _curator  # local to avoid circular at import time

    if event_type in {"conversation_closed", "file_imported", "chat_message"}:
        cid = payload.get("conversation_id")
        if cid:
            _curator.extract_memory_candidates(db, cid)
            consolidate_pending(db, limit=50)
    elif event_type == "manual_memory_added":
        # payload: {type, content, importance?, confidence?, user_signal?}
        cand = Candidate(
            type=payload["type"],
            content=payload["content"],
            importance=int(payload.get("importance", 4)),
            confidence=float(payload.get("confidence", 0.95)),
        )
        decision = decide_memory_action(db, cand)
        apply_decision(db, cand, decision)


# ----------------------------------------------------------------------
# 5.8 Tool-call memory (P1-8 GAP / Letta + Claude Memory)
# ----------------------------------------------------------------------
#
# Each agent tool invocation can be persisted as a first-class memory of
# ``type='tool_call'`` so the corpus learns "what tool was used, with what
# args, what came back". Stored fields (added by migration 005):
#
#   tool_name        — TEXT
#   tool_args_json   — TEXT (JSON-encoded args dict)
#   tool_result_hash — sha256(result)[:16]
#   tool_status      — 'success' | 'error' | 'cancelled'
#
# These rows ride the same scoring / retrieval pipeline as everything else;
# they just have extra structured fields the rest of the engine ignores.

_TOOL_CALL_STATUSES = frozenset({"success", "error", "cancelled"})


def _summarize_args(args: dict, *, max_chars: int = 80) -> str:
    """Compact one-line preview of an args dict for human-readable content.

    Avoids dumping huge payloads into the searchable text. Long string values
    are truncated; nested objects are shown as ``<dict[N]>``.
    """
    if not isinstance(args, dict) or not args:
        return ""
    parts: list[str] = []
    for key, value in args.items():
        if isinstance(value, str):
            v = value if len(value) <= 40 else value[:37] + "..."
            v = v.replace("\n", " ")
        elif isinstance(value, dict):
            v = f"<dict[{len(value)}]>"
        elif isinstance(value, list):
            v = f"<list[{len(value)}]>"
        else:
            v = str(value)
            if len(v) > 40:
                v = v[:37] + "..."
        parts.append(f"{key}={v}")
    out = ", ".join(parts)
    if len(out) > max_chars:
        out = out[: max_chars - 1].rstrip() + "…"
    return out


def _summarize_result(result: Any, status: str, *, max_chars: int = 80) -> str:
    """Short human label for a tool result. Distinguishes status quickly so
    the row is greppable in retrieval output without re-parsing the args.
    """
    if status == "error":
        text = "" if result is None else str(result)
        text = text.strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[: max_chars - 1].rstrip() + "…"
        return f"error: {text}" if text else "error"
    if status == "cancelled":
        return "cancelled"
    if result is None:
        return "ok"
    if isinstance(result, (dict, list)):
        kind = "dict" if isinstance(result, dict) else "list"
        return f"ok <{kind}[{len(result)}]>"
    if isinstance(result, (int, float, bool)):
        return f"ok {result}"
    text = str(result).strip().replace("\n", " ")
    if not text:
        return "ok"
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return f"ok: {text}"


def _hash_tool_result(result: Any) -> str:
    """sha256(canonical(result))[:16]. Used to dedup repeat tool calls
    without storing the full payload.
    """
    import hashlib
    if result is None:
        payload = b""
    elif isinstance(result, (dict, list)):
        try:
            payload = json.dumps(result, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        except (TypeError, ValueError):
            payload = repr(result).encode("utf-8")
    elif isinstance(result, bytes):
        payload = result
    else:
        payload = str(result).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class ToolCallStats:
    """Per-tool aggregate over the recorded tool_call corpus."""

    tool_name: str
    count: int
    success_count: int
    error_count: int
    cancelled_count: int
    avg_importance: float
    success_rate: float
    last_recorded_at: str | None = None
    sample_args: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "count": self.count,
            "success": self.success_count,
            "error": self.error_count,
            "cancelled": self.cancelled_count,
            "avg_importance": round(self.avg_importance, 3),
            "success_rate": round(self.success_rate, 3),
            "last_recorded_at": self.last_recorded_at,
            "sample_args": self.sample_args or [],
        }


def record_tool_call(
    db: MemoirsDB,
    *,
    tool_name: str,
    args: dict,
    result: Any,
    status: str = "success",
    conversation_id: str | None = None,
    importance: int = 2,
) -> str:
    """Persist a tool invocation as a ``type='tool_call'`` memory.

    Parameters
    ----------
    tool_name:
        Required, non-empty. Validated against the tool registry only by the
        caller — the engine just stores whatever string the agent provides.
    args:
        Args dict that was passed to the tool. Serialized verbatim into
        ``tool_args_json``; consider redacting secrets at the call site.
    result:
        Whatever the tool returned. Hashed (sha256[:16]) and summarized into
        the human content; **not** stored in full so giant payloads do not
        bloat the row.
    status:
        ``'success'`` (default) / ``'error'`` / ``'cancelled'``. Anything
        else raises ``ValueError`` so we never persist a free-text status
        the agent can't aggregate over later.
    conversation_id:
        Optional. Persisted into ``metadata_json.conversation_id`` so
        ``get_tool_calls_for_conversation`` can filter cheaply.
    importance:
        1..5; defaults to 2 (tool calls are usually noisier than facts).

    Returns
    -------
    str
        The new memory's id. Idempotency: if the same content_hash already
        exists on an active row the ``ON CONFLICT`` guard keeps a single
        row — the existing id is returned.
    """
    if not tool_name or not tool_name.strip():
        raise ValueError("tool_name is required")
    if status not in _TOOL_CALL_STATUSES:
        raise ValueError(
            f"invalid tool_status {status!r} — expected one of "
            f"{sorted(_TOOL_CALL_STATUSES)}"
        )
    if "tool_call" not in _VALID_MEMORY_TYPES:  # defensive — should never trip
        raise ValueError("tool_call type not registered")

    args = args or {}
    if not isinstance(args, dict):
        raise ValueError("args must be a dict")

    # Compact JSON; default=str absorbs odd types (datetime, UUID) without
    # blowing up the recorder.
    try:
        args_json = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"args is not JSON-serializable: {exc}") from exc

    result_hash = _hash_tool_result(result)
    summary = f"{tool_name}({_summarize_args(args)}) → {_summarize_result(result, status)}"
    content = summary
    h = content_hash(content)
    now = utc_now()

    metadata: dict[str, Any] = {}
    if conversation_id:
        metadata["conversation_id"] = conversation_id
    metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else "{}"

    importance = max(1, min(5, int(importance)))
    mid = stable_id("mem", "tool_call", tool_name, args_json, result_hash, status, now)

    with db.conn:
        db.conn.execute(
            """
            INSERT INTO memories (
                id, type, content, content_hash, importance, confidence,
                score, usage_count, user_signal, valid_from, metadata_json,
                tool_name, tool_args_json, tool_result_hash, tool_status,
                created_at, updated_at
            )
            VALUES (?, 'tool_call', ?, ?, ?, ?, 0, 0, 0, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(content_hash) WHERE archived_at IS NULL DO NOTHING
            """,
            (
                mid, content, h, importance, 0.95, now, metadata_json,
                tool_name, args_json, result_hash, status, now, now,
            ),
        )

    # Either the insert succeeded (new id) or it collided with an existing
    # active row sharing the same content_hash — fetch whichever id won.
    row = db.conn.execute(
        "SELECT id FROM memories WHERE content_hash = ? AND archived_at IS NULL",
        (h,),
    ).fetchone()
    final_id = row["id"] if row else mid

    # Score + Zettelkasten link the same way as ADD does. Embeddings are
    # populated best-effort; tool_call rows are usually short so the embed
    # call is cheap.
    try:
        emb.upsert_memory_embedding(db, final_id, content)
    except Exception:
        log.debug("upsert_memory_embedding skipped for %s", final_id[:16])
    full = db.conn.execute("SELECT * FROM memories WHERE id = ?", (final_id,)).fetchone()
    if full:
        score = calculate_memory_score(dict(full))
        with db.conn:
            db.conn.execute(
                "UPDATE memories SET score = ? WHERE id = ?", (score, final_id)
            )
    _maybe_link_memory(db, final_id)
    log.info(
        "tool_call recorded id=%s tool=%s status=%s",
        final_id[:16], tool_name, status,
    )
    return final_id


def summarize_tool_calls(
    db: MemoirsDB,
    tool_name: str | None = None,
    *,
    limit: int = 50,
) -> list[ToolCallStats]:
    """Aggregate tool_call memorias by ``tool_name``.

    Returns a list of :class:`ToolCallStats` sorted by ``count`` desc. Pass
    ``tool_name`` to scope the result to a single tool (returns at most one
    entry). ``limit`` caps the number of distinct tools returned.
    """
    where = ["type = 'tool_call'", "archived_at IS NULL", "tool_name IS NOT NULL"]
    params: list[Any] = []
    if tool_name:
        where.append("tool_name = ?")
        params.append(tool_name)
    where_sql = " AND ".join(where)

    rows = db.conn.execute(
        f"""
        SELECT tool_name,
               COUNT(*)                                           AS count,
               SUM(CASE WHEN tool_status = 'success'   THEN 1 ELSE 0 END) AS success_count,
               SUM(CASE WHEN tool_status = 'error'     THEN 1 ELSE 0 END) AS error_count,
               SUM(CASE WHEN tool_status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_count,
               AVG(CAST(importance AS REAL))                      AS avg_importance,
               MAX(created_at)                                    AS last_recorded_at
        FROM memories
        WHERE {where_sql}
        GROUP BY tool_name
        ORDER BY count DESC, tool_name ASC
        LIMIT ?
        """,
        (*params, int(limit)),
    ).fetchall()

    out: list[ToolCallStats] = []
    for r in rows:
        # Sample up to 3 most-recent args strings (truncated), best-effort.
        sample_rows = db.conn.execute(
            """
            SELECT tool_args_json FROM memories
            WHERE type='tool_call' AND archived_at IS NULL AND tool_name = ?
            ORDER BY created_at DESC LIMIT 3
            """,
            (r["tool_name"],),
        ).fetchall()
        samples = []
        for sr in sample_rows:
            raw = sr["tool_args_json"] or "{}"
            samples.append(raw if len(raw) <= 120 else raw[:117] + "...")
        cnt = int(r["count"]) or 1
        succ = int(r["success_count"] or 0)
        out.append(
            ToolCallStats(
                tool_name=r["tool_name"],
                count=cnt,
                success_count=succ,
                error_count=int(r["error_count"] or 0),
                cancelled_count=int(r["cancelled_count"] or 0),
                avg_importance=float(r["avg_importance"] or 0.0),
                success_rate=succ / cnt,
                last_recorded_at=r["last_recorded_at"],
                sample_args=samples,
            )
        )
    return out


def get_tool_calls_for_conversation(
    db: MemoirsDB,
    conversation_id: str,
) -> list[dict]:
    """Return ``tool_call`` memorias whose metadata.conversation_id matches.

    Uses ``json_extract`` so a per-conversation index is not needed; the
    expected tool_call volume per conversation is small (tens to hundreds).
    """
    if not conversation_id:
        return []
    rows = db.conn.execute(
        """
        SELECT id, type, content, tool_name, tool_args_json,
               tool_result_hash, tool_status, importance, confidence,
               score, created_at, metadata_json
        FROM memories
        WHERE type = 'tool_call'
          AND archived_at IS NULL
          AND json_extract(metadata_json, '$.conversation_id') = ?
        ORDER BY created_at ASC
        """,
        (conversation_id,),
    ).fetchall()
    return [dict(r) for r in rows]
