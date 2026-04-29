"""Sleep-time async consolidation (P1-4).

A Letta-inspired idle-time scheduler that runs housekeeping jobs while the
extraction daemon is otherwise quiet. Each cycle (``run_once``) walks
through a small fixed pipeline:

    1. Consolidate pending memory candidates.
    2. Auto-merge near-duplicate memorias.
    3. Rebuild Zettelkasten (A-MEM) links for memorias created since the
       last successful run.
    4. Prune low-value memorias (Ebbinghaus already decays scores; we just
       archive what falls below the dynamic threshold).
    5. (Optional) Flag potential contradictions among active memorias when
       Gemma is available.

Each job runs in its own try/except — a failure in one step never blocks
the others. The end-to-end report is persisted to ``sleep_runs`` so the
user can inspect history via ``memoirs sleep history`` / ``status``.

The scheduler enforces two pre-conditions before doing any work:

- ``max_load`` — system load1 / cpu_count must be ≤ this ratio. Defaults
  to 0.5 so we never compete with foreground workloads.
- ``min_idle_minutes`` — if any new message arrived in the last N minutes
  (or any source was touched), skip the cycle. Picks up freshly-extracted
  data on the next loop without racing the extract daemon.

Threading model: ``start_loop`` spawns a single non-daemon thread that
sleeps on a ``threading.Event``. ``stop_loop`` sets the event; the thread
exits within `interval_seconds` (or instantly if it was sleeping). No
external dependencies — stdlib only.

NOTE: this module deliberately uses lazy imports for ``zettelkasten`` and
``gemma`` because those modules pull in heavyweight optional deps
(sentence-transformers, llama-cpp). The scheduler must remain importable
even on a minimal install.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from ..db import MemoirsDB

log = logging.getLogger("memoirs.sleep")


# ----------------------------------------------------------------------
# Job registry
# ----------------------------------------------------------------------

#: Names of jobs the scheduler knows about, in execution order. Useful for
#: ``--jobs`` filtering in the CLI and for tests that want to exercise a
#: subset.
JOB_NAMES: tuple[str, ...] = (
    "consolidate",
    "dedup",
    "link_rebuild",
    "prune",
    "contradictions",
    "event_queue",
    "thread_summaries",
)


@dataclass
class JobReport:
    name: str
    started_at: str
    finished_at: str
    duration_ms: float
    status: str  # "ok", "error", "skipped"
    result: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SleepReport:
    started_at: str
    finished_at: Optional[str] = None
    jobs: list[JobReport] = field(default_factory=list)
    error: Optional[str] = None
    skipped_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "jobs": [j.to_dict() for j in self.jobs],
            "error": self.error,
            "skipped_reason": self.skipped_reason,
        }


# ----------------------------------------------------------------------
# Pre-conditions: load + idle
# ----------------------------------------------------------------------

def _system_load_ratio() -> float:
    """Return load1 / cpu_count. Returns 0.0 on platforms without
    ``os.getloadavg`` (Windows)."""
    try:
        return os.getloadavg()[0] / max(1, os.cpu_count() or 1)
    except (OSError, AttributeError):
        return 0.0


def get_last_activity_ts(db: MemoirsDB) -> Optional[datetime]:
    """Return the most recent activity timestamp visible to the scheduler.

    We blend two signals:

    - ``messages.created_at`` / ``messages.first_seen_at`` / ``updated_at``
      — picks up new ingest work.
    - ``sources.updated_at`` (or ``mtime_ns``) — picks up file touches even
      when no new messages were extracted.

    Returns ``None`` if the DB is empty (treated as "infinitely idle").
    """
    candidates: list[datetime] = []

    # Messages — fastest path. Works on legacy DBs where created_at may be
    # NULL, in which case we fall back to first_seen_at / updated_at.
    row = db.conn.execute(
        """
        SELECT MAX(
            COALESCE(updated_at, first_seen_at, created_at)
        ) AS ts FROM messages
        """
    ).fetchone()
    if row and row["ts"]:
        ts = _parse_iso(row["ts"])
        if ts:
            candidates.append(ts)

    # Sources — file watcher signal. mtime_ns is monotonic so prefer it
    # when present; otherwise fall back to updated_at.
    row = db.conn.execute(
        """
        SELECT MAX(updated_at) AS ts, MAX(mtime_ns) AS mtime_ns
        FROM sources
        """
    ).fetchone()
    if row:
        if row["ts"]:
            ts = _parse_iso(row["ts"])
            if ts:
                candidates.append(ts)
        if row["mtime_ns"]:
            try:
                ts = datetime.fromtimestamp(int(row["mtime_ns"]) / 1e9, tz=timezone.utc)
                candidates.append(ts)
            except (OverflowError, OSError, ValueError):
                pass

    return max(candidates) if candidates else None


def _parse_iso(text: str) -> Optional[datetime]:
    if not text:
        return None
    try:
        # Accept "Z" suffix as well as +00:00.
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


# ----------------------------------------------------------------------
# sleep_runs persistence
# ----------------------------------------------------------------------

_ENSURE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sleep_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    jobs_json TEXT NOT NULL DEFAULT '[]',
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_sleep_runs_started
    ON sleep_runs(started_at DESC);
"""


def ensure_sleep_runs_table(conn: sqlite3.Connection) -> None:
    """Idempotent table create, used as a belt-and-suspenders fallback when
    the migration runner hasn't been invoked yet (e.g. a test opens a raw
    sqlite3 connection)."""
    conn.executescript(_ENSURE_TABLE_SQL)


def _persist_run(db: MemoirsDB, report: SleepReport) -> int:
    ensure_sleep_runs_table(db.conn)
    jobs_json = json.dumps([j.to_dict() for j in report.jobs], ensure_ascii=False)
    cur = db.conn.execute(
        "INSERT INTO sleep_runs (started_at, finished_at, jobs_json, error) "
        "VALUES (?, ?, ?, ?)",
        (report.started_at, report.finished_at, jobs_json, report.error),
    )
    db.conn.commit()
    return int(cur.lastrowid)


def list_recent_runs(db: MemoirsDB, *, limit: int = 10) -> list[dict[str, Any]]:
    ensure_sleep_runs_table(db.conn)
    rows = db.conn.execute(
        "SELECT id, started_at, finished_at, jobs_json, error "
        "FROM sleep_runs ORDER BY id DESC LIMIT ?",
        (int(limit),),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            jobs = json.loads(r["jobs_json"] or "[]")
        except json.JSONDecodeError:
            jobs = []
        out.append({
            "id": int(r["id"]),
            "started_at": r["started_at"],
            "finished_at": r["finished_at"],
            "jobs": jobs,
            "error": r["error"],
        })
    return out


def get_last_run(db: MemoirsDB) -> Optional[dict[str, Any]]:
    runs = list_recent_runs(db, limit=1)
    return runs[0] if runs else None


# ----------------------------------------------------------------------
# Job implementations
# ----------------------------------------------------------------------

def _job_consolidate(db: MemoirsDB) -> dict[str, Any]:
    from .memory_engine import consolidate_pending
    return consolidate_pending(db, limit=200)


def _job_dedup(db: MemoirsDB) -> dict[str, Any]:
    from .lifecycle import auto_merge_near_duplicates
    return auto_merge_near_duplicates(db, threshold=0.92, dry_run=False)


def _job_link_rebuild(
    db: MemoirsDB,
    *,
    since: Optional[datetime] = None,
) -> dict[str, Any]:
    """Re-run ``link_memory`` for memorias created/updated since ``since``.

    When ``since`` is None the pre-condition layer falls back to "all
    memorias touched since the last successful sleep run" — the scheduler
    passes the last finished_at in. If no prior run exists, default to the
    last 24h to bound the work.
    """
    from . import zettelkasten as zk

    cutoff = since or (_utc_now() - timedelta(hours=24))
    cutoff_iso = cutoff.isoformat()
    rows = db.conn.execute(
        """
        SELECT id FROM memories
        WHERE archived_at IS NULL
          AND COALESCE(updated_at, created_at) >= ?
        ORDER BY updated_at DESC
        LIMIT 500
        """,
        (cutoff_iso,),
    ).fetchall()
    linked = 0
    for r in rows:
        try:
            zk.link_memory(db, r["id"])
            linked += 1
        except Exception:
            log.exception("link_memory failed for %s", r["id"])
    return {"scanned": len(rows), "linked": linked, "since": cutoff_iso}


def _job_prune(db: MemoirsDB) -> dict[str, Any]:
    from .memory_engine import archive_low_value_memories
    n = archive_low_value_memories(db)
    return {"archived": int(n)}


def _job_contradictions(db: MemoirsDB) -> dict[str, Any]:
    """Optional curator-LLM pass: look at the top-100 highest-similarity
    cross-type semantic links and flag contradictions.

    No-op (returns ``available=False``) when llama-cpp / the curator GGUF
    isn't installed — tests on minimal envs still pass.
    """
    try:
        from .curator import _have_curator, curator_detect_contradiction
    except ImportError:
        return {"available": False, "checked": 0, "flagged": 0}

    if not _have_curator():
        return {"available": False, "checked": 0, "flagged": 0}

    rows = db.conn.execute(
        """
        SELECT ml.source_memory_id AS a_id, ml.target_memory_id AS b_id,
               ml.similarity AS sim,
               ma.type AS a_type, mb.type AS b_type,
               ma.content AS a_content, mb.content AS b_content
          FROM memory_links ml
          JOIN memories ma ON ma.id = ml.source_memory_id
          JOIN memories mb ON mb.id = ml.target_memory_id
         WHERE ma.archived_at IS NULL
           AND mb.archived_at IS NULL
           AND ma.type != mb.type
           AND ml.similarity >= 0.85
         ORDER BY ml.similarity DESC
         LIMIT 100
        """
    ).fetchall()

    checked = 0
    flagged = 0
    for r in rows:
        checked += 1
        try:
            verdict = curator_detect_contradiction(r["a_content"], r["b_content"])
        except Exception:
            log.exception("curator_detect_contradiction failed")
            continue
        if isinstance(verdict, dict) and verdict.get("contradictory"):
            flagged += 1
            # P5-2: persist for the conflict resolution UI / CLI. Failures
            # here must not break the housekeeping cycle — best-effort only.
            try:
                from .conflicts import record_conflict
                record_conflict(
                    db,
                    memory_a_id=r["a_id"],
                    memory_b_id=r["b_id"],
                    similarity=float(r["sim"]) if r["sim"] is not None else None,
                    detector="gemma",
                    reason=str(verdict.get("reason") or "")[:500] or None,
                )
            except Exception:  # noqa: BLE001
                log.exception("record_conflict failed for %s/%s", r["a_id"], r["b_id"])
    return {"available": True, "checked": checked, "flagged": flagged}


def _job_event_queue(db: MemoirsDB) -> dict[str, Any]:
    """Drain pending rows from ``event_queue`` (P0-4).

    The handler table is intentionally minimal — most event types are
    informational signals consumed by other agents (audit, graph indexers,
    etc.) that subscribe out-of-band. Unknown event types are marked ``done``
    by ``process_pending`` so the queue never grows unbounded just because a
    consumer hasn't been wired up yet.
    """
    from .event_queue import process_pending

    handlers: dict[str, Callable[..., Any]] = {
        # No-ops by default. Once downstream consumers land they can extend
        # this dict (or pass their own at call time). Marked as "skipped" by
        # process_pending when no handler is registered, which is the
        # desired behavior for now — we want the queue to flush.
    }
    return process_pending(db, batch_size=200, handlers=handlers)


def _job_thread_summaries(db: MemoirsDB) -> dict[str, Any]:
    """Auto-resume thread (P-resume): generate durable summaries for idle convs.

    Capped at 10 conversations per tick to keep the curator budget bounded;
    the loop runs ~hourly so a busy session catches up within a few cycles.
    """
    from .thread_resume import (
        DEFAULT_IDLE_MINUTES,
        DEFAULT_MAX_CONVS_PER_TICK,
        sleep_thread_summaries_job,
    )
    return sleep_thread_summaries_job(
        db,
        idle_minutes=DEFAULT_IDLE_MINUTES,
        max_convs=DEFAULT_MAX_CONVS_PER_TICK,
    )


#: Mapping job name → callable. Each callable takes ``db`` plus optional
#: ``**kwargs`` and returns a JSON-serializable result dict.
_JOB_FNS: dict[str, Callable[..., dict[str, Any]]] = {
    "consolidate": _job_consolidate,
    "dedup": _job_dedup,
    "link_rebuild": _job_link_rebuild,
    "prune": _job_prune,
    "contradictions": _job_contradictions,
    "event_queue": _job_event_queue,
    "thread_summaries": _job_thread_summaries,
}


def _run_job(
    name: str,
    fn: Callable[..., dict[str, Any]],
    db: MemoirsDB,
    **kwargs: Any,
) -> JobReport:
    started = _utc_now()
    started_iso = started.isoformat()
    t0 = time.perf_counter()
    try:
        result = fn(db, **kwargs) if kwargs else fn(db)
        finished = _utc_now()
        return JobReport(
            name=name,
            started_at=started_iso,
            finished_at=finished.isoformat(),
            duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            status="ok",
            result=result if isinstance(result, dict) else {"value": result},
        )
    except Exception as e:  # noqa: BLE001 — we want to report any failure
        finished = _utc_now()
        log.exception("sleep job %s failed", name)
        return JobReport(
            name=name,
            started_at=started_iso,
            finished_at=finished.isoformat(),
            duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            status="error",
            error=f"{type(e).__name__}: {e}",
        )


# ----------------------------------------------------------------------
# Scheduler
# ----------------------------------------------------------------------

class SleepScheduler:
    """Run housekeeping jobs during idle periods.

    Parameters
    ----------
    db_path:
        SQLite database to operate on. The scheduler opens a *fresh*
        :class:`MemoirsDB` per ``run_once`` to keep its connection isolated
        from the daemon's main connection (SQLite is multi-connection
        safe; ``check_same_thread=False`` is set in ``MemoirsDB``).
    interval_seconds:
        Wall-clock wait between cycles when running in a loop.
    max_load:
        Skip a cycle when ``loadavg1 / cpu_count`` exceeds this.
    min_idle_minutes:
        Skip a cycle if the most recent activity timestamp is more recent
        than ``now - min_idle_minutes``.
    enabled_jobs:
        Iterable of job names to run; defaults to all of :data:`JOB_NAMES`.
        Order is preserved.
    """

    def __init__(
        self,
        db_path: Path | str,
        *,
        interval_seconds: int = 3600,
        max_load: float = 0.5,
        min_idle_minutes: int = 10,
        enabled_jobs: Optional[Iterable[str]] = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.interval_seconds = int(interval_seconds)
        self.max_load = float(max_load)
        self.min_idle_minutes = int(min_idle_minutes)
        self.enabled_jobs: tuple[str, ...] = tuple(enabled_jobs) if enabled_jobs else JOB_NAMES
        unknown = [j for j in self.enabled_jobs if j not in _JOB_FNS]
        if unknown:
            raise ValueError(
                f"unknown sleep jobs: {unknown}; expected subset of {list(JOB_NAMES)}"
            )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_started_at: Optional[str] = None
        self._last_finished_at: Optional[str] = None
        self._last_report: Optional[SleepReport] = None

    # ------------------------------------------------------------------
    # Pre-conditions
    # ------------------------------------------------------------------

    def _check_preconditions(self, db: MemoirsDB) -> Optional[str]:
        load = _system_load_ratio()
        if load > self.max_load:
            return f"system_load={load:.3f} exceeds max_load={self.max_load:.3f}"
        last_activity = get_last_activity_ts(db)
        if last_activity is not None:
            delta = _utc_now() - last_activity
            if delta < timedelta(minutes=self.min_idle_minutes):
                mins = delta.total_seconds() / 60.0
                return (
                    f"recent activity {mins:.1f}m ago "
                    f"< min_idle_minutes={self.min_idle_minutes}"
                )
        return None

    # ------------------------------------------------------------------
    # Single cycle
    # ------------------------------------------------------------------

    def run_once(
        self,
        *,
        force: bool = False,
        jobs: Optional[Iterable[str]] = None,
    ) -> SleepReport:
        """Run one full cycle.

        Parameters
        ----------
        force:
            Skip pre-condition checks. Useful for ``memoirs sleep run-once``
            invoked by a human who knows what they're asking for.
        jobs:
            Override the configured ``enabled_jobs`` for this single
            invocation. Names outside :data:`JOB_NAMES` raise ValueError.
        """
        report = SleepReport(started_at=_utc_now_iso())
        db = MemoirsDB(self.db_path)
        try:
            if not force:
                skip = self._check_preconditions(db)
                if skip:
                    report.skipped_reason = skip
                    report.finished_at = _utc_now_iso()
                    log.info("sleep: skip — %s", skip)
                    self._last_report = report
                    self._last_started_at = report.started_at
                    self._last_finished_at = report.finished_at
                    # Persist skipped runs too — gives operators a paper
                    # trail of "we ran but pre-conditions said no".
                    try:
                        report.error = None
                        _persist_run(db, report)
                    except sqlite3.Error:
                        log.exception("sleep: failed to persist skipped run")
                    return report

            # Resolve job order.
            if jobs is None:
                ordered = self.enabled_jobs
            else:
                jobs_set = list(jobs)
                unknown = [j for j in jobs_set if j not in _JOB_FNS]
                if unknown:
                    raise ValueError(
                        f"unknown sleep jobs: {unknown}; "
                        f"expected subset of {list(JOB_NAMES)}"
                    )
                # Preserve canonical order while honoring the subset.
                ordered = tuple(j for j in JOB_NAMES if j in set(jobs_set))

            # Determine the "since" timestamp for link_rebuild based on the
            # last successful sleep run that finished cleanly. Fall back to
            # the past 24h if no prior run exists.
            since = self._link_rebuild_since(db)

            for name in ordered:
                fn = _JOB_FNS[name]
                kwargs: dict[str, Any] = {}
                if name == "link_rebuild":
                    kwargs["since"] = since
                jr = _run_job(name, fn, db, **kwargs)
                report.jobs.append(jr)
                log.info(
                    "sleep job %s: %s (%.0fms)%s",
                    name,
                    jr.status,
                    jr.duration_ms,
                    f" — {jr.error}" if jr.error else "",
                )

            report.finished_at = _utc_now_iso()
            try:
                _persist_run(db, report)
            except sqlite3.Error:
                log.exception("sleep: failed to persist run")
            self._last_report = report
            self._last_started_at = report.started_at
            self._last_finished_at = report.finished_at
            return report
        finally:
            db.close()

    def _link_rebuild_since(self, db: MemoirsDB) -> Optional[datetime]:
        last = get_last_run(db)
        if not last or not last.get("finished_at"):
            return None
        return _parse_iso(last["finished_at"])

    # ------------------------------------------------------------------
    # Loop control
    # ------------------------------------------------------------------

    def start_loop(self) -> threading.Thread:
        """Start the background loop. Returns the thread."""
        if self._thread and self._thread.is_alive():
            return self._thread
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="memoirs-sleep", daemon=False,
        )
        self._thread.start()
        log.info(
            "sleep: loop started (interval=%ds max_load=%.2f min_idle_min=%d jobs=%s)",
            self.interval_seconds, self.max_load, self.min_idle_minutes,
            list(self.enabled_jobs),
        )
        return self._thread

    def stop_loop(self, *, timeout: float = 5.0) -> None:
        """Signal the loop to stop and wait up to ``timeout`` seconds."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                log.warning("sleep: loop did not exit within %.1fs", timeout)
            self._thread = None
        log.info("sleep: loop stopped")

    def _loop(self) -> None:
        # Run immediately on start, then sleep between cycles. Operators
        # generally want to see something happen right after `daemon
        # start`; deferring would feel unresponsive.
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception:
                log.exception("sleep: run_once raised, will retry next cycle")
            # Wait on the event so stop_loop() returns quickly.
            self._stop_event.wait(self.interval_seconds)


# ----------------------------------------------------------------------
# CLI helpers
# ----------------------------------------------------------------------

def run_once_cli(
    db_path: Path | str,
    *,
    jobs: Optional[Iterable[str]] = None,
    force: bool = True,
) -> dict[str, Any]:
    """Helper used by ``memoirs sleep run-once``. Always forces past
    pre-conditions because the user explicitly asked for a cycle."""
    sched = SleepScheduler(db_path)
    report = sched.run_once(force=force, jobs=jobs)
    return report.to_dict()


def status_cli(db_path: Path | str) -> dict[str, Any]:
    """Compact status summary used by ``memoirs sleep status``."""
    db = MemoirsDB(db_path)
    try:
        last = get_last_run(db)
        next_eta: Optional[str] = None
        if last and last.get("finished_at"):
            ft = _parse_iso(last["finished_at"])
            if ft:
                next_eta = (ft + timedelta(seconds=3600)).isoformat()
        return {
            "db": str(Path(db_path).resolve()),
            "last_run": last,
            "next_run_estimate": next_eta,
            "jobs": list(JOB_NAMES),
        }
    finally:
        db.close()


def history_cli(db_path: Path | str, *, limit: int = 10) -> list[dict[str, Any]]:
    db = MemoirsDB(db_path)
    try:
        return list_recent_runs(db, limit=limit)
    finally:
        db.close()
