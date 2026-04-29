"""Layer 5.1b — EXPIRE / ARCHIVE decision enrichment (P1-10 GAP).

The base curator (`decide_memory_action`) only emits ADD / UPDATE / MERGE /
IGNORE / CONTRADICTION. The two terminal lifecycle actions — EXPIRE and
ARCHIVE — were declared in :class:`memory_engine.Decision` but no caller
ever generated them.

This module fills the gap. It is intentionally side-effect free at decision
time: rules are evaluated against the candidate + its neighbors and produce
a (possibly enriched) :class:`Decision` whose ``secondary_actions`` list
carries any cascading EXPIRE / ARCHIVE the caller (``apply_decision``) must
materialise.

Rules
-----

EXPIRE — neighbor becomes obsolete because the candidate supersedes it.

  * Gemma path: ``gemma_resolve_conflict(candidate, neighbor)`` reports
    ``contradictory=True`` AND ``winner == 'A'`` (= candidate). In that case
    the neighbor is the loser and is expired.
  * Heuristic path (no Gemma): ``similarity > 0.8`` AND neighbor age > 30d
    AND type ∈ {fact, task, decision}. Preferences and styles never
    auto-expire without explicit confirmation — they are subjective and
    "the user changed their mind" is a higher-stakes claim than for facts.

ARCHIVE — neighbor is stale and not pulling weight.

  * ``usage_count == 0`` AND age > 90d → True ("nobody ever used you").
  * ``score < 0.2`` AND age > 60d → True ("you scored too low for too long").

Both archive predicates are independent: either suffices.

Cascade semantics
-----------------

* ``ADD`` / ``UPDATE`` + obsolete neighbor → primary action stays, an EXPIRE
  secondary action is appended for that neighbor.
* ``MERGE`` whose target neighbor is itself stale → demote to ARCHIVE
  (merging into a dying row is wasted work; archive instead, then keep the
  candidate as a fresh ADD via a secondary action when one is warranted).

The wrapper `enrich_decision` is the single entrypoint used by
``memory_engine.decide_memory_action``.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Iterable


log = logging.getLogger("memoirs.lifecycle_decisions")


# Types whose contradictions the heuristic is willing to act on without a
# model in the loop. Subjective types (preference, style) require Gemma.
_HEURISTIC_EXPIRE_TYPES = frozenset({"fact", "task", "decision"})

# Thresholds — exposed as module-level constants so tests can assert against
# them without re-deriving the policy.
EXPIRE_SIM_THRESHOLD = 0.8
EXPIRE_MIN_NEIGHBOR_AGE_DAYS = 30
ARCHIVE_USAGE_AGE_DAYS = 90
ARCHIVE_SCORE_AGE_DAYS = 60
ARCHIVE_SCORE_THRESHOLD = 0.2


def _parse_iso(ts: str | None) -> datetime | None:
    """Lenient ISO-8601 parser — same policy as memory_engine._parse_iso.

    Duplicated locally so this module has no inbound dependency on
    memory_engine (which imports *us* from inside decide_memory_action).
    """
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, AttributeError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _age_days(neighbor: dict, *, now: datetime | None = None) -> float | None:
    """Age (days) of ``neighbor`` based on ``created_at``. ``None`` if unknown."""
    ts = _parse_iso(neighbor.get("created_at"))
    if ts is None:
        return None
    ref = now or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    return max(0.0, (ref - ts).total_seconds() / 86400.0)


def _have_curator() -> bool:
    """Best-effort check for a loaded curator LLM. Never raises — if the
    import itself fails (test envs, optional dep), just say "no curator".
    """
    try:
        from . import curator as _c
        return bool(_c._have_curator())
    except Exception:
        return False


# Backward-compat alias: tests historically patched ``_have_gemma`` here.
_have_gemma = _have_curator


def _resolve_conflict(candidate_content: str, neighbor_content: str) -> dict | None:
    """Best-effort wrapper around ``curator.curator_resolve_conflict``.

    Returns ``None`` if the curator LLM is unavailable or raises. Otherwise
    the dict returned by the curator (with at least ``contradictory`` /
    ``winner`` keys).
    """
    if not _have_curator():
        return None
    try:
        from . import curator as _c
        return _c.curator_resolve_conflict(candidate_content, neighbor_content)
    except Exception as exc:  # pragma: no cover — defensive
        log.debug("lifecycle_decisions: curator_resolve_conflict raised %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public predicates
# ---------------------------------------------------------------------------


def should_expire(
    candidate: Any,
    neighbor: dict,
    *,
    now: datetime | None = None,
    use_gemma: bool | None = None,
) -> tuple[bool, str]:
    """Decide whether ``neighbor`` should be expired in favour of ``candidate``.

    Parameters
    ----------
    candidate:
        The new candidate. Accepts either a :class:`gemma.Candidate` or any
        object with ``content``/``type`` attributes (or a dict).
    neighbor:
        Row dict from the curator's neighbor pool. Expected keys: ``id``,
        ``type``, ``content``, ``created_at``, ``similarity``.
    now:
        Reference timestamp for age calculations (tests).
    use_gemma:
        Force Gemma on (``True``) / off (``False``). ``None`` (default)
        delegates to availability detection — Gemma when loaded, heuristic
        otherwise. Tests pass an explicit value to avoid env coupling.

    Returns
    -------
    tuple[bool, str]
        ``(should_expire, reason)``.
    """
    if not isinstance(neighbor, dict) or not neighbor.get("id"):
        return False, "neighbor missing id"

    cand_content = _attr(candidate, "content", "")
    cand_type = _attr(candidate, "type", "")
    nbr_content = neighbor.get("content", "")
    nbr_type = neighbor.get("type")

    if not cand_content or not nbr_content:
        return False, "empty content"

    # Honour MEMOIRS_CURATOR_ENABLED=off (legacy: MEMOIRS_GEMMA_CURATOR=off)
    # as an explicit opt-out — must precede the auto-detect branch so callers
    # can disable the curator without monkey-patching ``_have_curator``.
    import os as _os
    if use_gemma is None:
        flag = _os.environ.get("MEMOIRS_CURATOR_ENABLED")
        if flag is None:
            flag = _os.environ.get("MEMOIRS_GEMMA_CURATOR", "auto")
        if (flag or "auto").lower() == "off":
            use_gemma = False

    # Curator route: contradictory AND candidate wins.
    if use_gemma is True or (use_gemma is None and _have_curator()):
        verdict = _resolve_conflict(cand_content, nbr_content)
        if verdict is not None:
            contradictory = bool(verdict.get("contradictory"))
            winner = verdict.get("winner")
            if contradictory and winner == "A":
                reason = (
                    f"gemma: contradictory, candidate wins "
                    f"({str(verdict.get('reason', ''))[:80]})"
                )
                return True, reason
            # Curator spoke; trust it. Even if the heuristic would have
            # fired, an explicit "not contradictory" or "neighbor wins"
            # verdict is authoritative.
            return False, f"gemma: not expirable (winner={winner!r})"
        # Curator was requested but unavailable — fall through to heuristic
        # only if we weren't *forced* on. ``use_gemma=True`` with no curator
        # means "abstain": never silently downgrade authority.
        if use_gemma is True:
            return False, "gemma unavailable (use_gemma=True)"

    # Heuristic route.
    if nbr_type not in _HEURISTIC_EXPIRE_TYPES:
        return False, f"heuristic: type {nbr_type!r} not auto-expirable"

    # Same-type guard: heuristic only expires rows of the same type.
    if cand_type and nbr_type and cand_type != nbr_type:
        return False, "heuristic: type mismatch"

    sim = float(neighbor.get("similarity") or 0.0)
    if sim <= EXPIRE_SIM_THRESHOLD:
        return False, f"heuristic: similarity {sim:.3f} <= {EXPIRE_SIM_THRESHOLD}"

    age = _age_days(neighbor, now=now)
    if age is None:
        return False, "heuristic: neighbor age unknown"
    if age <= EXPIRE_MIN_NEIGHBOR_AGE_DAYS:
        return False, f"heuristic: neighbor age {age:.1f}d <= {EXPIRE_MIN_NEIGHBOR_AGE_DAYS}"

    return True, (
        f"heuristic: sim={sim:.3f}, age={age:.0f}d, type={nbr_type}"
    )


def should_archive(
    neighbor: dict,
    *,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """Decide whether ``neighbor`` should be archived as stale."""
    if not isinstance(neighbor, dict) or not neighbor.get("id"):
        return False, "neighbor missing id"

    age = _age_days(neighbor, now=now)
    if age is None:
        return False, "age unknown"

    usage = int(neighbor.get("usage_count") or 0)
    if usage == 0 and age > ARCHIVE_USAGE_AGE_DAYS:
        return True, f"unused for {age:.0f}d (>{ARCHIVE_USAGE_AGE_DAYS}d), usage_count=0"

    score = float(neighbor.get("score") or 0.0)
    if score < ARCHIVE_SCORE_THRESHOLD and age > ARCHIVE_SCORE_AGE_DAYS:
        return True, (
            f"low score {score:.3f} (<{ARCHIVE_SCORE_THRESHOLD}) "
            f"for {age:.0f}d (>{ARCHIVE_SCORE_AGE_DAYS}d)"
        )

    return False, f"healthy (usage={usage}, score={score:.3f}, age={age:.0f}d)"


# ---------------------------------------------------------------------------
# Decision enrichment — main entrypoint
# ---------------------------------------------------------------------------


def enrich_decision(
    decision: Any,
    candidate: Any,
    neighbors: Iterable[dict] | None,
    *,
    now: datetime | None = None,
    use_gemma: bool | None = None,
) -> Any:
    """Augment a base :class:`Decision` with EXPIRE / ARCHIVE side-effects.

    Behavior:

    * ``IGNORE`` / ``CONTRADICTION`` are passed through unchanged — those
      paths already hold the corpus stable; layering more side-effects on
      top would be too aggressive.
    * ``ADD`` / ``UPDATE``: scan neighbors, emit one secondary EXPIRE per
      obsolete neighbor (max 3 to bound work and avoid mass-expiring during
      a single ingest).
    * ``MERGE`` whose target is stale → primary becomes ARCHIVE on the same
      target (merging into a dying row is pointless; just archive it).
    * Any other primary: archive predicate also runs against neighbors and
      attaches secondary ARCHIVE actions, so a single ingest can prune
      multiple stale rows opportunistically.

    The function never raises; failures in Gemma / parsing are absorbed and
    the original decision is returned untouched.
    """
    # Late import keeps this module decoupled at load time and avoids the
    # cycle (memory_engine → lifecycle_decisions → memory_engine).
    from .memory_engine import Decision  # noqa: WPS433 — intentional late binding

    if decision is None:
        return decision

    # Normalize: caller may pass either a Decision dataclass or a dict.
    as_dict = isinstance(decision, dict)
    action = (decision.get("action") if as_dict else getattr(decision, "action", "")) or ""
    target_id = (
        decision.get("target_memory_id") if as_dict else getattr(decision, "target_memory_id", None)
    )

    if action in {"IGNORE", "CONTRADICTION", "EXPIRE", "ARCHIVE"}:
        # Already terminal; nothing useful to add.
        return decision

    neighbors_list: list[dict] = list(neighbors or [])
    if not neighbors_list:
        return decision

    secondary: list[Decision] = []
    seen_targets: set[str] = set()

    # MERGE cascade: if the merge target is itself stale, downgrade to ARCHIVE.
    if action == "MERGE" and target_id:
        target_row = next(
            (n for n in neighbors_list if n.get("id") == target_id), None
        )
        if target_row is not None:
            archive, archive_reason = should_archive(target_row, now=now)
            if archive:
                new_decision = Decision(
                    action="ARCHIVE",
                    target_memory_id=target_id,
                    reason=f"merge-target stale → archive ({archive_reason})",
                )
                # Preserve any pre-existing secondary_actions on the original.
                _copy_secondaries(decision, new_decision)
                return new_decision

    # ADD / UPDATE / MERGE: scan all neighbors for cascade effects.
    if action in {"ADD", "UPDATE", "MERGE"}:
        max_secondary = 3
        for nb in neighbors_list:
            nb_id = nb.get("id")
            if not nb_id or nb_id in seen_targets:
                continue
            # Don't EXPIRE the same row we're merging into.
            if action == "MERGE" and nb_id == target_id:
                continue
            if len(secondary) >= max_secondary:
                break

            expire, expire_reason = should_expire(
                candidate, nb, now=now, use_gemma=use_gemma,
            )
            if expire:
                secondary.append(
                    Decision(
                        action="EXPIRE",
                        target_memory_id=nb_id,
                        reason=expire_reason,
                    )
                )
                seen_targets.add(nb_id)
                continue

            archive, archive_reason = should_archive(nb, now=now)
            if archive:
                secondary.append(
                    Decision(
                        action="ARCHIVE",
                        target_memory_id=nb_id,
                        reason=archive_reason,
                    )
                )
                seen_targets.add(nb_id)

    if not secondary:
        return decision

    # Attach. Decision is a slotless dataclass; assigning a new attribute is
    # legal and the dataclass already carries a ``secondary_actions`` field
    # (added in this gap-fill).
    if as_dict:
        merged = dict(decision)
        merged.setdefault("secondary_actions", [])
        merged["secondary_actions"] = list(merged["secondary_actions"]) + secondary
        return merged

    existing = list(getattr(decision, "secondary_actions", None) or [])
    decision.secondary_actions = existing + secondary
    return decision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from either a dataclass-ish object or a dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def sweep_archive_predicate(db: Any) -> dict:
    """CLI helper: scan active memories and archive any matching
    :func:`should_archive`. Used by ``memoirs maintenance --enrich-decisions``.

    Returns a small JSON-friendly summary so the CLI command can print it.
    """
    rows = db.conn.execute(
        """
        SELECT id, type, content, importance, confidence, score,
               usage_count, last_used_at, valid_to, archived_at,
               created_at, updated_at
        FROM memories
        WHERE archived_at IS NULL
        """
    ).fetchall()
    archived = 0
    reasons: list[dict] = []
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with db.conn:
        for r in rows:
            row = dict(r)
            archive, reason = should_archive(row)
            if not archive:
                continue
            db.conn.execute(
                """
                UPDATE memories
                SET archived_at = ?,
                    archive_reason = COALESCE(archive_reason, ?),
                    metadata_json = json_set(
                        COALESCE(metadata_json, '{}'),
                        '$.archive_rule', ?,
                        '$.archived_at', ?
                    ),
                    updated_at = ?
                WHERE id = ?
                """,
                (now_iso, reason, reason, now_iso, now_iso, row["id"]),
            )
            archived += 1
            if len(reasons) < 10:
                reasons.append({"id": row["id"], "reason": reason})
    return {"scanned": len(rows), "archived": archived, "samples": reasons}


def _copy_secondaries(src: Any, dst: Any) -> None:
    """Carry over any ``secondary_actions`` from ``src`` to ``dst``."""
    existing = (
        src.get("secondary_actions") if isinstance(src, dict)
        else getattr(src, "secondary_actions", None)
    )
    if not existing:
        return
    if isinstance(dst, dict):
        dst["secondary_actions"] = list(existing)
    else:
        dst.secondary_actions = list(existing)
