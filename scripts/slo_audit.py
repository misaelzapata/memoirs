"""Phase 5D — SLO audit harness for memoirs.

Validates that the memoirs engine meets concrete latency / throughput / RAM
budgets against a real corpus. Self-contained: no HTTP layer, drives
``assemble_context`` and ``search_similar_memories`` in-process.

Usage:
    python scripts/slo_audit.py latency      [--db PATH] [--iters N]
    python scripts/slo_audit.py cold-start   [--db PATH] [--runs N]
    python scripts/slo_audit.py sustained    [--db PATH] [--seconds 60] [--workers 50]
    python scripts/slo_audit.py memory       [--db PATH] [--queries 100]
    python scripts/slo_audit.py all          [--db PATH] [--out PATH]

Output: a pass/fail table on stdout + a JSON report at
``.memoirs/slo_report.json`` (overridable via ``--out``).

The script never writes to ``memoirs/``; it opens the live DB read-mostly
(``PRAGMA query_only = ON`` is applied to every connection it creates).

SLO targets are encoded as constants near the top so they can be tweaked or
overridden by env vars without editing the runner.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import os
import statistics
import subprocess
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

# Make ``memoirs`` importable when run directly from the repo.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# SLO targets
# ---------------------------------------------------------------------------

SLO: dict[str, dict[str, float]] = {
    "mcp_get_context": {"p50_ms": 50.0, "p95_ms": 200.0, "p99_ms": 1000.0},
    "mcp_search_memory": {"p50_ms": 20.0, "p95_ms": 100.0},
    "assemble_context_stream_ttft": {"ms": 50.0},
    "mcp_extract_pending": {"p50_s": 5.0, "p95_s": 20.0},
    "cold_start": {"p50_s": 8.0},
    "sustained": {"rps": 50.0},
    "memory": {"idle_mb": 200.0, "active_mb": 4096.0, "peak_mb": 6144.0},
}

# A small ring of representative queries — exercises a mix of topics so
# embedding LRU + BM25 don't dominate the measurement.
QUERIES: tuple[str, ...] = (
    "memoirs MCP server",
    "gocracker hash benchmark",
    "gemma extract daemon configuration",
    "cursor ingestion pipeline",
    "feedback score memory",
    "raptor consolidation tree",
    "graph retrieval entities",
    "embedding cache reranker",
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _silence_logs() -> None:
    """Mute noisy loggers — the SLO runner has its own structured stdout."""
    for name in ("memoirs", "memoirs.embeddings", "memoirs.db",
                 "sentence_transformers", "httpx", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _percentile(values: list[float], pct: float) -> float:
    """Return the ``pct``-th percentile (0 ≤ pct ≤ 100) using nearest-rank."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return float(s[k])


def _open_db_readonly(db_path: Path, *, query_only: bool = True):
    """Open a MemoirsDB.

    When ``query_only`` is True the connection is pinned to read-only via
    ``PRAGMA query_only = ON`` — safe but breaks ``assemble_context``'s
    ``usage_count`` bump. Bench callers that need the full live path should
    snapshot the DB to a temp file first (see :func:`_snapshot_db`).

    We import lazily so callers measuring cold-start can take a snapshot
    BEFORE any memoirs import happens.
    """
    from memoirs.db import MemoirsDB
    db = MemoirsDB(db_path, auto_migrate=False)
    if query_only:
        with contextlib.suppress(Exception):
            db.conn.execute("PRAGMA query_only = ON")
    return db


def _snapshot_db(src: Path) -> Path:
    """Copy ``src`` (and its WAL/SHM siblings) to a fresh temp file.

    Returns the path of the writable copy. Caller is responsible for
    deleting it. Using ``sqlite3``'s online backup API would be marginally
    nicer but a flat copy works fine when no writer is active and avoids a
    second DB connection.
    """
    import shutil
    import tempfile
    fd, tmp = tempfile.mkstemp(prefix="slo_audit_", suffix=".sqlite")
    os.close(fd)
    tmp_path = Path(tmp)
    shutil.copy2(src, tmp_path)
    # WAL + SHM siblings, if present, contain pending writes — copy them
    # too so we observe a consistent snapshot.
    for suffix in ("-wal", "-shm"):
        sib = src.with_name(src.name + suffix)
        if sib.exists():
            shutil.copy2(sib, tmp_path.with_name(tmp_path.name + suffix))
    return tmp_path


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Latency benchmarks
# ---------------------------------------------------------------------------


def _bench_callable(fn: Callable[[int], Any], iters: int, warmup: int = 2) -> list[float]:
    """Time ``fn(i)`` for ``iters`` iterations after ``warmup`` ignored runs.

    Returns the per-call wall-clock latency in milliseconds.
    """
    for i in range(warmup):
        fn(i)
    out: list[float] = []
    for i in range(iters):
        t0 = time.perf_counter()
        fn(i)
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def _evaluate_latency(slo_key: str, samples_ms: list[float]) -> dict[str, Any]:
    """Compare measured percentiles against the SLO target table."""
    target = SLO[slo_key]
    p50 = _percentile(samples_ms, 50)
    p95 = _percentile(samples_ms, 95)
    p99 = _percentile(samples_ms, 99)

    out: dict[str, Any] = {
        "samples": len(samples_ms),
        "p50_actual_ms": round(p50, 2),
        "p95_actual_ms": round(p95, 2),
        "p99_actual_ms": round(p99, 2),
    }
    if "p50_ms" in target:
        out["p50_target_ms"] = target["p50_ms"]
        out["p50_pass"] = p50 <= target["p50_ms"]
    if "p95_ms" in target:
        out["p95_target_ms"] = target["p95_ms"]
        out["p95_pass"] = p95 <= target["p95_ms"]
    if "p99_ms" in target:
        out["p99_target_ms"] = target["p99_ms"]
        out["p99_pass"] = p99 <= target["p99_ms"]

    out["pass"] = all(v for k, v in out.items() if k.endswith("_pass"))
    return out


def _run_latency(db_path: Path, iters: int = 30) -> dict[str, Any]:
    """Drive each MCP-equivalent code path and report percentiles."""
    from memoirs.engine import embeddings as embed_mod
    from memoirs.engine import gemma as gemma_mod
    from memoirs.engine.memory_engine import (
        assemble_context, assemble_context_stream,
    )

    _silence_logs()
    # Snapshot so live ``assemble_context`` (which bumps ``usage_count``)
    # doesn't mutate the source DB. The copy is deleted at the end.
    snap = _snapshot_db(db_path)
    db = _open_db_readonly(snap, query_only=False)
    try:
        # --- mcp_get_context (== assemble_context) ---
        # Use a higher warmup count: the first call loads the embedder /
        # reranker / sqlite-vec / FTS5 caches and is dramatically slower
        # than steady state. Warmup of 3 absorbs all of those.
        ctx_samples = _bench_callable(
            lambda i: assemble_context(
                db, QUERIES[i % len(QUERIES)], top_k=20, max_lines=15,
            ),
            iters=iters,
            warmup=5,
        )

        # --- mcp_search_memory (search_similar_memories) ---
        try:
            search_samples = _bench_callable(
                lambda i: embed_mod.search_similar_memories(
                    db, QUERIES[i % len(QUERIES)], top_k=10,
                ),
                iters=iters,
                warmup=3,
            )
            search_err: str | None = None
        except Exception as e:  # pragma: no cover - depends on extras
            search_samples = []
            search_err = f"{type(e).__name__}: {e}"

        # --- assemble_context_stream TTFT (time to first event) ---
        # Warmup: fully drain a few calls so HyDE / reranker imports + caches
        # don't skew the first-iter measurement.
        for i in range(3):
            for _ in assemble_context_stream(
                db, QUERIES[i % len(QUERIES)], top_k=20, max_lines=15,
            ):
                pass
        ttft_samples: list[float] = []
        for i in range(iters):
            t0 = time.perf_counter()
            gen = assemble_context_stream(
                db, QUERIES[i % len(QUERIES)], top_k=20, max_lines=15,
            )
            next(gen)  # consume "meta"
            ttft_samples.append((time.perf_counter() - t0) * 1000.0)
            # Drain so usage_count side-effects happen consistently.
            for _ in gen:
                pass

        # --- mcp_extract_pending (single conversation, dry-ish) ---
        # ``extract_pending(limit=1)`` may invoke a real Gemma pass that takes
        # tens of seconds. To keep the audit bounded:
        #  * If MEMOIRS_SKIP_EXTRACT is set, mock with a single SQL probe.
        #  * Otherwise cap iterations at 2 (one to absorb model load, one
        #    steady-state). The SLO is p50 < 5s / p95 < 20s.
        extract_samples_s: list[float] = []
        if os.environ.get("MEMOIRS_SKIP_EXTRACT") == "1":
            # Dispatcher-only timing: just probe the pending-convs SQL.
            for _ in range(3):
                t0 = time.perf_counter()
                db.conn.execute(
                    "SELECT 1 FROM conversations c "
                    "LEFT JOIN memory_candidates mc ON mc.conversation_id = c.id "
                    "WHERE mc.id IS NULL AND c.message_count >= 3 LIMIT 1"
                ).fetchone()
                extract_samples_s.append(time.perf_counter() - t0)
        else:
            for _ in range(2):
                t0 = time.perf_counter()
                try:
                    gemma_mod.extract_pending(db, limit=1, min_messages=3)
                except Exception:  # pragma: no cover - DB-specific
                    pass
                extract_samples_s.append(time.perf_counter() - t0)

    finally:
        db.close()
        with contextlib.suppress(Exception):
            snap.unlink()
            for suffix in ("-wal", "-shm"):
                sib = snap.with_name(snap.name + suffix)
                if sib.exists():
                    sib.unlink()

    # ----- evaluate -----
    ctx_eval = _evaluate_latency("mcp_get_context", ctx_samples)

    if search_samples:
        search_eval = _evaluate_latency("mcp_search_memory", search_samples)
    else:
        search_eval = {
            "samples": 0, "skipped": True, "error": search_err,
            "p50_target_ms": SLO["mcp_search_memory"]["p50_ms"],
            "p95_target_ms": SLO["mcp_search_memory"]["p95_ms"],
            "pass": False,
        }

    ttft_target = SLO["assemble_context_stream_ttft"]["ms"]
    ttft_p50 = _percentile(ttft_samples, 50)
    ttft_p95 = _percentile(ttft_samples, 95)
    ttft_eval = {
        "samples": len(ttft_samples),
        "target_ms": ttft_target,
        "p50_actual_ms": round(ttft_p50, 2),
        "p95_actual_ms": round(ttft_p95, 2),
        "pass": ttft_p50 <= ttft_target,
    }

    extract_p50 = _percentile([s * 1000 for s in extract_samples_s], 50) / 1000.0
    extract_p95 = _percentile([s * 1000 for s in extract_samples_s], 95) / 1000.0
    extract_eval = {
        "samples": len(extract_samples_s),
        "p50_target_s": SLO["mcp_extract_pending"]["p50_s"],
        "p50_actual_s": round(extract_p50, 3),
        "p95_target_s": SLO["mcp_extract_pending"]["p95_s"],
        "p95_actual_s": round(extract_p95, 3),
        "pass": (extract_p50 <= SLO["mcp_extract_pending"]["p50_s"]
                 and extract_p95 <= SLO["mcp_extract_pending"]["p95_s"]),
    }

    return {
        "mcp_get_context": ctx_eval,
        "mcp_search_memory": search_eval,
        "assemble_context_stream_ttft": ttft_eval,
        "mcp_extract_pending": extract_eval,
    }


# ---------------------------------------------------------------------------
# Cold-start benchmark
# ---------------------------------------------------------------------------


_COLD_START_SNIPPET = (
    "import sys, time;"
    "from datetime import datetime, timezone;"
    "t0=time.perf_counter();"
    "from memoirs.db import MemoirsDB;"
    "from memoirs.engine.memory_engine import assemble_context;"
    "db=MemoirsDB({db!r}, auto_migrate=False);"
    # Pass as_of=now so the call stays read-only — assemble_context skips
    # the usage_count UPDATE on the time-travel path. Avoids needing
    # PRAGMA query_only (which would crash on write).
    "now=datetime.now(timezone.utc).isoformat();"
    "r=assemble_context(db, 'cold start probe', top_k=10, max_lines=5, as_of=now);"
    "sys.stdout.write(f'READY {{(time.perf_counter()-t0)*1000:.1f}}ms n={{len(r.get(\"memories\",[]))}}\\n');"
    "sys.stdout.flush()"
)


def _run_cold_start(db_path: Path, runs: int = 3) -> dict[str, Any]:
    """Spawn a fresh Python and measure time-to-first-stdout-line.

    We run the child with ``-S`` (skip site initialisation) and an explicit
    minimal env so user-level shells don't perturb the result. The reported
    number includes process spawn + `import memoirs` + DB open + one
    ``assemble_context`` call.
    """
    py = sys.executable
    durations_s: list[float] = []
    last_stdout = ""

    snippet = _COLD_START_SNIPPET.format(db=str(db_path))
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "PYTHONPATH": str(ROOT),
        # Disable HF network probes that add seconds on a cold cache.
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "MEMOIRS_RETRIEVAL_MODE": os.environ.get("MEMOIRS_RETRIEVAL_MODE", "bm25"),
    }

    for _ in range(runs):
        t0 = time.perf_counter()
        proc = subprocess.run(
            [py, "-c", snippet],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        elapsed = time.perf_counter() - t0
        durations_s.append(elapsed)
        last_stdout = proc.stdout.strip()

    target = SLO["cold_start"]["p50_s"]
    p50 = _percentile([d * 1000 for d in durations_s], 50) / 1000.0
    return {
        "runs": runs,
        "target_s": target,
        "actual_p50_s": round(p50, 3),
        "samples_s": [round(d, 3) for d in durations_s],
        "last_stdout": last_stdout,
        "pass": p50 <= target,
    }


# ---------------------------------------------------------------------------
# Sustained throughput
# ---------------------------------------------------------------------------


def _run_sustained(db_path: Path, seconds: int = 60, workers: int = 50,
                   target_rps: float | None = None,
                   mode: str = "hybrid_graph") -> dict[str, Any]:
    """Hammer ``assemble_context`` from a thread pool for ``seconds`` seconds.

    Each worker gets its own ``MemoirsDB`` connection (Python's sqlite3
    raises ``InterfaceError`` on cross-thread cursor concurrency even with
    ``check_same_thread=False``). WAL mode lets the readers proceed in
    parallel against the same on-disk DB.
    """
    import threading

    from memoirs.engine.memory_engine import assemble_context

    _silence_logs()
    target = target_rps if target_rps is not None else SLO["sustained"]["rps"]

    # Snapshot so concurrent ``usage_count`` UPDATEs don't touch the
    # production DB. The copy lives only for the duration of the bench.
    snap = _snapshot_db(db_path)

    # Per-thread DB handle. Built lazily on first use, closed at teardown.
    _local = threading.local()
    _all_dbs: list = []
    _all_dbs_lock = threading.Lock()

    def _thread_db():
        d = getattr(_local, "db", None)
        if d is None:
            d = _open_db_readonly(snap, query_only=False)
            _local.db = d
            with _all_dbs_lock:
                _all_dbs.append(d)
        return d

    deadline = time.perf_counter() + seconds
    latencies_ms: list[float] = []
    errors = 0
    done = 0

    def _one(i: int) -> tuple[float, bool]:
        q = QUERIES[i % len(QUERIES)]
        t0 = time.perf_counter()
        try:
            assemble_context(_thread_db(), q, top_k=20, max_lines=15,
                             retrieval_mode=mode)
            return ((time.perf_counter() - t0) * 1000.0, True)
        except Exception:
            return ((time.perf_counter() - t0) * 1000.0, False)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        # Submit a batch sized to keep the pool busy. We loop until the
        # wall-clock budget runs out, then drain.
        in_flight: list = []
        i = 0
        while time.perf_counter() < deadline:
            while len(in_flight) < workers and time.perf_counter() < deadline:
                in_flight.append(ex.submit(_one, i))
                i += 1
            if not in_flight:
                break
            # Reap whatever finished; non-blocking poll.
            fresh: list = []
            for fut in in_flight:
                if fut.done():
                    lat, ok = fut.result()
                    latencies_ms.append(lat)
                    done += 1
                    if not ok:
                        errors += 1
                else:
                    fresh.append(fut)
            in_flight = fresh
            if all(not f.done() for f in in_flight):
                time.sleep(0.001)
        # Drain remaining
        for fut in as_completed(in_flight, timeout=max(5, seconds)):
            lat, ok = fut.result()
            latencies_ms.append(lat)
            done += 1
            if not ok:
                errors += 1

    for d in _all_dbs:
        with contextlib.suppress(Exception):
            d.close()
    with contextlib.suppress(Exception):
        snap.unlink()
        for suffix in ("-wal", "-shm"):
            sib = snap.with_name(snap.name + suffix)
            if sib.exists():
                sib.unlink()

    elapsed = max(0.001, seconds)
    rps = done / elapsed
    p99 = _percentile(latencies_ms, 99)
    p50 = _percentile(latencies_ms, 50)
    return {
        "seconds": seconds,
        "workers": workers,
        "mode": mode,
        "completed": done,
        "errors": errors,
        "actual_rps": round(rps, 2),
        "target_rps": target,
        "p50_ms": round(p50, 2),
        "p99_ms": round(p99, 2),
        "pass": (rps >= target and errors == 0),
    }


# ---------------------------------------------------------------------------
# Memory footprint
# ---------------------------------------------------------------------------


def _rss_mb() -> float:
    """Return current RSS in MiB via ``/proc/self/status`` (Linux) or ``resource``."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0
    except OSError:
        pass
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # Linux reports KiB, BSD reports bytes.
        return float(ru.ru_maxrss) / (1024.0 if sys.platform.startswith("linux") else 1024.0 * 1024.0)
    except Exception:
        return 0.0


_MEMORY_CHILD_SNIPPET = r"""
import json, os, sys, gc, tracemalloc, contextlib

def _rss_mb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return 0.0
    return 0.0

DB_PATH = sys.argv[1]
QUERIES = sys.argv[2].split('|')
N = int(sys.argv[3])

tracemalloc.start()
idle_rss = _rss_mb()
idle_trace = tracemalloc.get_traced_memory()[1] / (1024 * 1024)

import shutil, tempfile
fd, tmp = tempfile.mkstemp(prefix='slo_audit_mem_', suffix='.sqlite')
os.close(fd); shutil.copy2(DB_PATH, tmp)
for suf in ('-wal', '-shm'):
    sib = DB_PATH + suf
    if os.path.exists(sib):
        shutil.copy2(sib, tmp + suf)

from memoirs.db import MemoirsDB
from memoirs.engine import embeddings as embed_mod
from memoirs.engine.memory_engine import assemble_context
db = MemoirsDB(tmp, auto_migrate=False)

curator_loaded = False
try:
    embed_mod._require_embedder()
    curator_loaded = True
except Exception:
    pass

gc.collect()
active_rss = _rss_mb()
active_trace = tracemalloc.get_traced_memory()[1] / (1024 * 1024)

peak_rss = active_rss
for i in range(N):
    try:
        assemble_context(db, QUERIES[i % len(QUERIES)], top_k=20, max_lines=15)
    except Exception:
        pass
    if i % 10 == 0:
        peak_rss = max(peak_rss, _rss_mb())
gc.collect()
peak_rss = max(peak_rss, _rss_mb())
peak_trace = tracemalloc.get_traced_memory()[1] / (1024 * 1024)

db.close()
with contextlib.suppress(Exception):
    os.unlink(tmp)
    for suf in ('-wal', '-shm'):
        if os.path.exists(tmp + suf):
            os.unlink(tmp + suf)

print(json.dumps({
    'queries': N,
    'curator_loaded': curator_loaded,
    'idle_rss_mb': idle_rss,
    'active_rss_mb': active_rss,
    'peak_rss_mb': peak_rss,
    'idle_trace_mb': idle_trace,
    'active_trace_mb': active_trace,
    'peak_trace_mb': peak_trace,
}))
"""


def _run_memory(db_path: Path, queries: int = 100) -> dict[str, Any]:
    """Snapshot RSS at three milestones: idle / curator-loaded / post-load.

    We spawn a fresh child interpreter and run the milestones there. The
    parent process ``slo_audit`` may already have ``memoirs`` imported
    (e.g. when called from ``all``), so measuring RSS in-process would
    overstate "idle". The child starts clean.
    """
    py = sys.executable
    args = [
        py, "-c", _MEMORY_CHILD_SNIPPET, str(db_path), "|".join(QUERIES), str(queries),
    ]
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(ROOT))
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    proc = subprocess.run(args, env=env, capture_output=True, text=True, timeout=300)
    out = proc.stdout.strip().splitlines()
    payload: dict[str, Any] = {}
    for line in reversed(out):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
                break
            except Exception:
                continue
    if not payload:
        raise RuntimeError(
            f"memory child produced no JSON. stdout={proc.stdout[-500:]!r} "
            f"stderr={proc.stderr[-500:]!r}"
        )

    idle_rss = float(payload.get("idle_rss_mb", 0))
    active_rss = float(payload.get("active_rss_mb", 0))
    peak_rss = float(payload.get("peak_rss_mb", 0))
    targets = SLO["memory"]
    return {
        "queries": int(payload.get("queries", queries)),
        "curator_loaded": bool(payload.get("curator_loaded", False)),
        "idle_mb": round(idle_rss, 1),
        "active_mb": round(active_rss, 1),
        "peak_mb": round(peak_rss, 1),
        "idle_target_mb": targets["idle_mb"],
        "active_target_mb": targets["active_mb"],
        "peak_target_mb": targets["peak_mb"],
        "tracemalloc_idle_mb": round(float(payload.get("idle_trace_mb", 0)), 1),
        "tracemalloc_active_mb": round(float(payload.get("active_trace_mb", 0)), 1),
        "tracemalloc_peak_mb": round(float(payload.get("peak_trace_mb", 0)), 1),
        "pass": (idle_rss <= targets["idle_mb"]
                 and active_rss <= targets["active_mb"]
                 and peak_rss <= targets["peak_mb"]),
    }


def _run_memory_inproc(db_path: Path, queries: int = 100) -> dict[str, Any]:
    """In-process variant of :func:`_run_memory` — used by tests so we don't
    need to spawn a subprocess. Reports the same shape but measures the
    parent process (so ``idle`` will include any imports already done)."""
    tracemalloc.start()

    idle_rss = _rss_mb()
    idle_trace = tracemalloc.get_traced_memory()[1] / (1024 * 1024)

    from memoirs.db import MemoirsDB
    from memoirs.engine import embeddings as embed_mod
    from memoirs.engine.memory_engine import assemble_context
    snap = _snapshot_db(db_path)
    db = MemoirsDB(snap, auto_migrate=False)

    # Load the curator (ST embedder) — this is the dominant cost.
    try:
        embed_mod._require_embedder()
        curator_loaded = True
    except Exception:
        curator_loaded = False

    gc.collect()
    active_rss = _rss_mb()
    active_trace = tracemalloc.get_traced_memory()[1] / (1024 * 1024)

    # Drive 100 queries — peak RAM after this is what we care about.
    peak_rss = active_rss
    for i in range(queries):
        try:
            assemble_context(db, QUERIES[i % len(QUERIES)], top_k=20, max_lines=15)
        except Exception:
            pass
        if i % 10 == 0:
            peak_rss = max(peak_rss, _rss_mb())
    gc.collect()
    peak_rss = max(peak_rss, _rss_mb())
    peak_trace = tracemalloc.get_traced_memory()[1] / (1024 * 1024)

    db.close()
    tracemalloc.stop()
    with contextlib.suppress(Exception):
        snap.unlink()
        for suffix in ("-wal", "-shm"):
            sib = snap.with_name(snap.name + suffix)
            if sib.exists():
                sib.unlink()

    targets = SLO["memory"]
    return {
        "queries": queries,
        "curator_loaded": curator_loaded,
        "idle_mb": round(idle_rss, 1),
        "active_mb": round(active_rss, 1),
        "peak_mb": round(peak_rss, 1),
        "idle_target_mb": targets["idle_mb"],
        "active_target_mb": targets["active_mb"],
        "peak_target_mb": targets["peak_mb"],
        "tracemalloc_idle_mb": round(idle_trace, 1),
        "tracemalloc_active_mb": round(active_trace, 1),
        "tracemalloc_peak_mb": round(peak_trace, 1),
        "pass": (idle_rss <= targets["idle_mb"]
                 and active_rss <= targets["active_mb"]
                 and peak_rss <= targets["peak_mb"]),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _flatten_slos(report: dict[str, Any]) -> list[tuple[str, str, str, bool]]:
    """Return a list of (slo, target, actual, pass) rows for tabular output."""
    rows: list[tuple[str, str, str, bool]] = []
    slos = report.get("slos", {})

    def _fmt_ms(v: Any) -> str:
        return f"{v:.1f}ms" if isinstance(v, (int, float)) else str(v)

    def _fmt_s(v: Any) -> str:
        return f"{v:.2f}s" if isinstance(v, (int, float)) else str(v)

    for name in ("mcp_get_context", "mcp_search_memory"):
        d = slos.get(name)
        if not d:
            continue
        for pct in ("p50", "p95", "p99"):
            tgt = d.get(f"{pct}_target_ms")
            act = d.get(f"{pct}_actual_ms")
            ok = d.get(f"{pct}_pass")
            if tgt is None:
                continue
            rows.append((f"{name} {pct}", _fmt_ms(tgt), _fmt_ms(act), bool(ok)))

    ttft = slos.get("assemble_context_stream_ttft")
    if ttft:
        rows.append(("assemble_context_stream TTFT", _fmt_ms(ttft.get("target_ms")),
                     _fmt_ms(ttft.get("p50_actual_ms")), bool(ttft.get("pass"))))

    ext = slos.get("mcp_extract_pending")
    if ext:
        rows.append(("extract_pending p50", _fmt_s(ext.get("p50_target_s")),
                     _fmt_s(ext.get("p50_actual_s")), bool(ext.get("p50_actual_s", 0) <= ext.get("p50_target_s", 0))))
        rows.append(("extract_pending p95", _fmt_s(ext.get("p95_target_s")),
                     _fmt_s(ext.get("p95_actual_s")), bool(ext.get("p95_actual_s", 0) <= ext.get("p95_target_s", 0))))

    cs = slos.get("cold_start")
    if cs:
        rows.append(("cold_start p50", _fmt_s(cs.get("target_s")),
                     _fmt_s(cs.get("actual_p50_s")), bool(cs.get("pass"))))

    sus = slos.get("sustained")
    if sus:
        rows.append(("sustained throughput",
                     f"≥{sus.get('target_rps')} rps",
                     f"{sus.get('actual_rps')} rps",
                     bool(sus.get("pass"))))

    mem = slos.get("memory")
    if mem:
        rows.append(("RAM idle", f"<{mem.get('idle_target_mb')} MB",
                     f"{mem.get('idle_mb')} MB",
                     mem.get("idle_mb", 9e9) <= mem.get("idle_target_mb", 0)))
        rows.append(("RAM active", f"<{mem.get('active_target_mb')} MB",
                     f"{mem.get('active_mb')} MB",
                     mem.get("active_mb", 9e9) <= mem.get("active_target_mb", 0)))
        rows.append(("RAM peak", f"<{mem.get('peak_target_mb')} MB",
                     f"{mem.get('peak_mb')} MB",
                     mem.get("peak_mb", 9e9) <= mem.get("peak_target_mb", 0)))

    return rows


def _print_table(report: dict[str, Any]) -> None:
    rows = _flatten_slos(report)
    if not rows:
        print("(no SLO results to display)")
        return
    name_w = max(len(r[0]) for r in rows)
    tgt_w = max(len(r[1]) for r in rows)
    act_w = max(len(r[2]) for r in rows)
    print(f"{'SLO':<{name_w}}  {'TARGET':<{tgt_w}}  {'ACTUAL':<{act_w}}  RESULT")
    print("-" * (name_w + tgt_w + act_w + 12))
    for name, tgt, act, ok in rows:
        verdict = "PASS" if ok else "FAIL"
        print(f"{name:<{name_w}}  {tgt:<{tgt_w}}  {act:<{act_w}}  {verdict}")
    summary = report.get("summary", {})
    print(f"\n{summary.get('passed', 0)}/{summary.get('total', 0)} SLOs passed "
          f"({summary.get('failed', 0)} failed)")


def _summarise(report: dict[str, Any]) -> dict[str, int]:
    rows = _flatten_slos(report)
    passed = sum(1 for r in rows if r[3])
    return {"passed": passed, "failed": len(rows) - passed, "total": len(rows)}


def _write_report(report: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_latency(args: argparse.Namespace) -> int:
    res = _run_latency(Path(args.db), iters=args.iters)
    report = {"timestamp": _utcnow_iso(), "slos": res}
    report["summary"] = _summarise(report)
    _print_table(report)
    if args.out:
        _write_report(report, Path(args.out))
    return 0 if report["summary"]["failed"] == 0 else 1


def _cmd_cold_start(args: argparse.Namespace) -> int:
    res = _run_cold_start(Path(args.db), runs=args.runs)
    report = {"timestamp": _utcnow_iso(), "slos": {"cold_start": res}}
    report["summary"] = _summarise(report)
    _print_table(report)
    if args.out:
        _write_report(report, Path(args.out))
    return 0 if report["summary"]["failed"] == 0 else 1


def _cmd_sustained(args: argparse.Namespace) -> int:
    res = _run_sustained(Path(args.db), seconds=args.seconds,
                         workers=args.workers, target_rps=args.target_rps,
                         mode=args.mode)
    report = {"timestamp": _utcnow_iso(), "slos": {"sustained": res}}
    report["summary"] = _summarise(report)
    _print_table(report)
    if args.out:
        _write_report(report, Path(args.out))
    return 0 if report["summary"]["failed"] == 0 else 1


def _cmd_memory(args: argparse.Namespace) -> int:
    res = _run_memory(Path(args.db), queries=args.queries)
    report = {"timestamp": _utcnow_iso(), "slos": {"memory": res}}
    report["summary"] = _summarise(report)
    _print_table(report)
    if args.out:
        _write_report(report, Path(args.out))
    return 0 if report["summary"]["failed"] == 0 else 1


def _cmd_all(args: argparse.Namespace) -> int:
    db = Path(args.db)
    slos: dict[str, Any] = {}

    print("[1/4] latency...", flush=True)
    slos.update(_run_latency(db, iters=args.iters))

    print("[2/4] cold-start...", flush=True)
    slos["cold_start"] = _run_cold_start(db, runs=args.runs)

    print("[3/4] sustained...", flush=True)
    slos["sustained"] = _run_sustained(
        db, seconds=args.seconds, workers=args.workers, mode=args.mode,
    )

    print("[4/4] memory...", flush=True)
    slos["memory"] = _run_memory(db, queries=args.queries)

    report = {"timestamp": _utcnow_iso(), "slos": slos}
    report["summary"] = _summarise(report)

    print()
    _print_table(report)

    out = Path(args.out)
    _write_report(report, out)
    print(f"\nreport written to {out}")
    return 0 if report["summary"]["failed"] == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="slo_audit", description="Phase 5D SLO audit")
    p.add_argument("--db", default=os.environ.get("MEMOIRS_DB", ".memoirs/memoirs.sqlite"),
                   help="path to the memoirs SQLite DB")
    p.add_argument("--out", default=".memoirs/slo_report.json",
                   help="JSON output path (default: .memoirs/slo_report.json)")
    sub = p.add_subparsers(dest="command", required=True)

    lat = sub.add_parser("latency", help="per-endpoint p50/p95/p99 vs SLO")
    lat.add_argument("--iters", type=int, default=30)
    lat.set_defaults(func=_cmd_latency)

    cs = sub.add_parser("cold-start", help="fork a fresh interpreter, time first result")
    cs.add_argument("--runs", type=int, default=3)
    cs.set_defaults(func=_cmd_cold_start)

    sus = sub.add_parser("sustained", help="60s × 50 RPS thread pool against assemble_context")
    sus.add_argument("--seconds", type=int, default=60)
    sus.add_argument("--workers", type=int, default=50)
    sus.add_argument("--target-rps", type=float, default=None)
    sus.add_argument("--mode", default="hybrid_graph",
                     help="retrieval_mode for assemble_context (default hybrid_graph)")
    sus.set_defaults(func=_cmd_sustained)

    mem = sub.add_parser("memory", help="tracemalloc + RSS at idle / active / peak")
    mem.add_argument("--queries", type=int, default=100)
    mem.set_defaults(func=_cmd_memory)

    a = sub.add_parser("all", help="run every benchmark + emit JSON report")
    a.add_argument("--iters", type=int, default=30)
    a.add_argument("--runs", type=int, default=3)
    a.add_argument("--seconds", type=int, default=60)
    a.add_argument("--workers", type=int, default=50)
    a.add_argument("--queries", type=int, default=100)
    a.add_argument("--mode", default="hybrid_graph")
    a.set_defaults(func=_cmd_all)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
