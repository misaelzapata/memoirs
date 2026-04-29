#!/usr/bin/env python
"""Sustained throughput benchmark for ``assemble_context`` (GAP fix #4).

Hammers ``assemble_context`` from a thread pool against the live corpus
and reports:

  * sustained RPS over the run window
  * p50 / p95 / p99 per-call latency
  * configured embedding backend
  * per-backend comparison when ``--backend all`` is selected

Usage::

    python scripts/bench_throughput.py [--backend BACKEND] [--workers 50]
                                       [--seconds 60] [--mode hybrid_graph]
                                       [--db PATH] [--out PATH]

``--backend`` accepts:

* ``st``               — sentence-transformers, no pool (current default)
* ``st_pool``          — sentence-transformers + process pool (default 4 workers)
* ``fastembed``        — fastembed ONNX (skipped with a notice if not installed)
* ``all``              — run every available backend and report a comparison

The audit baseline (Phase 5D) measured ``hybrid_graph`` at 2 rps with the
ST default. Anything in the 30-50 rps range counts as the GAP #4 fix
landing.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shutil
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Make the in-tree package importable when running the script directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------- #
# Workload — mirrors scripts/slo_audit.py so numbers stay comparable.
# ---------------------------------------------------------------------- #


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


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _silence_logs() -> None:
    for name in ("memoirs", "memoirs.embeddings", "memoirs.embed_pool",
                 "memoirs.db", "sentence_transformers", "fastembed",
                 "huggingface_hub", "httpx", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return float(s[k])


def _snapshot_db(src: Path) -> Path:
    """Copy ``src`` (+ WAL/SHM siblings) to a writable temp file."""
    fd, tmp = tempfile.mkstemp(prefix="bench_throughput_", suffix=".sqlite")
    os.close(fd)
    tmp_path = Path(tmp)
    shutil.copy2(src, tmp_path)
    for suffix in ("-wal", "-shm"):
        sib = src.with_name(src.name + suffix)
        if sib.exists():
            shutil.copy2(sib, tmp_path.with_name(tmp_path.name + suffix))
    return tmp_path


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------- #
# Backend control — set env BEFORE importing memoirs.engine.embeddings so
# the singleton module picks up the right configuration.
# ---------------------------------------------------------------------- #


def _apply_backend(backend: str, pool_workers: int) -> None:
    """Mutate the env to select ``backend`` for the current run."""
    if backend == "st":
        os.environ["MEMOIRS_EMBED_BACKEND"] = "sentence-transformers"
    elif backend == "st_pool":
        os.environ["MEMOIRS_EMBED_BACKEND"] = "process_pool"
        os.environ["MEMOIRS_EMBED_POOL_WORKERS"] = str(pool_workers)
    elif backend == "fastembed":
        os.environ["MEMOIRS_EMBED_BACKEND"] = "fastembed"
    else:
        raise ValueError(f"unknown backend: {backend!r}")


def _reset_backend_singletons() -> None:
    """Drop cached backend singletons so a backend swap mid-process takes effect."""
    from memoirs.engine import embeddings as emb
    from memoirs.engine import embed_pool as ep

    emb._FASTEMBED_SINGLETON = None  # type: ignore[attr-defined]
    emb._FASTEMBED_WARNED = False    # type: ignore[attr-defined]
    emb._MODEL_SINGLETON = None      # type: ignore[attr-defined]
    emb.clear_embed_cache()
    ep.shutdown_pool()


# ---------------------------------------------------------------------- #
# Bench driver
# ---------------------------------------------------------------------- #


def run_bench(db_path: Path, *, seconds: int, workers: int, mode: str,
              unique_queries: bool = False) -> dict:
    """Drive ``assemble_context`` from a thread pool for ``seconds`` seconds."""
    import threading

    from memoirs.db import MemoirsDB
    from memoirs.engine.memory_engine import assemble_context

    snap = _snapshot_db(db_path)

    _local = threading.local()
    _all_dbs: list = []
    _all_dbs_lock = threading.Lock()

    def _thread_db():
        d = getattr(_local, "db", None)
        if d is None:
            d = MemoirsDB(snap, auto_migrate=False)
            _local.db = d
            with _all_dbs_lock:
                _all_dbs.append(d)
        return d

    # Warm: one synchronous call so the embedder / pool / FTS5 caches are
    # populated before the timed window starts.
    warm_db = MemoirsDB(snap, auto_migrate=False)
    try:
        assemble_context(warm_db, QUERIES[0], top_k=20, max_lines=15,
                         retrieval_mode=mode)
    except Exception as e:  # pragma: no cover - depends on extras
        print(f"warmup failed: {type(e).__name__}: {e}", file=sys.stderr)
    finally:
        with contextlib.suppress(Exception):
            warm_db.close()

    deadline = time.perf_counter() + seconds
    latencies_ms: list[float] = []
    errors = 0
    done = 0

    def _one(i: int) -> tuple[float, bool]:
        q = QUERIES[i % len(QUERIES)]
        if unique_queries:
            # Append a per-call suffix so the embed LRU never hits — the
            # raw GIL-bound embed cost dominates instead of being masked
            # by a 100% hit rate after the first pass through the ring.
            q = f"{q} (variation {i})"
        t0 = time.perf_counter()
        try:
            assemble_context(_thread_db(), q, top_k=20, max_lines=15,
                             retrieval_mode=mode)
            return ((time.perf_counter() - t0) * 1000.0, True)
        except Exception:
            return ((time.perf_counter() - t0) * 1000.0, False)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        in_flight: list = []
        i = 0
        while time.perf_counter() < deadline:
            while len(in_flight) < workers and time.perf_counter() < deadline:
                in_flight.append(ex.submit(_one, i))
                i += 1
            if not in_flight:
                break
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
        for fut in as_completed(in_flight, timeout=max(5, seconds)):
            lat, ok = fut.result()
            latencies_ms.append(lat)
            done += 1
            if not ok:
                errors += 1
    elapsed = max(0.001, time.perf_counter() - started)

    for d in _all_dbs:
        with contextlib.suppress(Exception):
            d.close()
    with contextlib.suppress(Exception):
        snap.unlink()
        for suffix in ("-wal", "-shm"):
            sib = snap.with_name(snap.name + suffix)
            if sib.exists():
                sib.unlink()

    return {
        "seconds": seconds,
        "workers": workers,
        "mode": mode,
        "completed": done,
        "errors": errors,
        "elapsed_s": round(elapsed, 3),
        "rps": round(done / elapsed, 2),
        "p50_ms": round(_percentile(latencies_ms, 50), 2),
        "p95_ms": round(_percentile(latencies_ms, 95), 2),
        "p99_ms": round(_percentile(latencies_ms, 99), 2),
        "mean_ms": round(statistics.fmean(latencies_ms), 2) if latencies_ms else 0.0,
    }


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #


_BACKENDS_ALL = ("st", "st_pool", "fastembed")


def _backend_available(backend: str) -> tuple[bool, str | None]:
    """Return (ok, reason-if-not-ok). Never raises."""
    if backend == "fastembed":
        try:
            import fastembed  # noqa: F401
            return True, None
        except ImportError:
            return False, "fastembed not installed (pip install -e '.[embeddings_fast]')"
    if backend in {"st", "st_pool"}:
        try:
            import sentence_transformers  # noqa: F401
            return True, None
        except ImportError:
            return False, "sentence-transformers not installed"
    return False, f"unknown backend: {backend!r}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(Path(".memoirs/memoirs.sqlite")),
                        help="SQLite DB path (default: .memoirs/memoirs.sqlite)")
    parser.add_argument("--backend", default="all",
                        choices=("st", "st_pool", "fastembed", "all"),
                        help="embedding backend (default: all)")
    parser.add_argument("--workers", type=int, default=50,
                        help="concurrent worker threads (default: 50)")
    parser.add_argument("--seconds", type=int, default=60,
                        help="run window in seconds (default: 60)")
    parser.add_argument("--pool-workers", type=int, default=4,
                        help="process-pool worker count for st_pool (default: 4)")
    parser.add_argument("--mode", default="hybrid_graph",
                        choices=("hybrid_graph", "hybrid", "bm25", "semantic"),
                        help="retrieval mode passed to assemble_context "
                             "(default: hybrid_graph — matches Phase 5D audit)")
    parser.add_argument("--unique-queries", action="store_true",
                        help=(
                            "rotate a unique suffix into every query so the "
                            "embed LRU cache always misses — exposes the raw "
                            "embed cost (the small ring otherwise turns into "
                            "a cache-hit storm after the first pass)."
                        ))
    parser.add_argument("--out", default=str(Path(".memoirs/throughput_report.json")),
                        help="JSON report path (default: .memoirs/throughput_report.json)")
    args = parser.parse_args()

    _silence_logs()
    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"error: DB not found at {db_path}", file=sys.stderr)
        return 1

    targets = list(_BACKENDS_ALL) if args.backend == "all" else [args.backend]

    print(f"db:       {db_path}")
    print(f"workers:  {args.workers}")
    print(f"seconds:  {args.seconds}")
    print(f"mode:     {args.mode}")
    print(f"backends: {', '.join(targets)}")
    print()

    results: dict[str, dict] = {}
    for backend in targets:
        ok, reason = _backend_available(backend)
        if not ok:
            print(f"[skip] {backend}: {reason}")
            results[backend] = {"skipped": True, "reason": reason}
            continue
        print(f"[run]  {backend} ...", flush=True)
        _apply_backend(backend, args.pool_workers)
        _reset_backend_singletons()
        try:
            res = run_bench(
                db_path, seconds=args.seconds, workers=args.workers,
                mode=args.mode, unique_queries=args.unique_queries,
            )
        except Exception as e:
            print(f"[fail] {backend}: {type(e).__name__}: {e}")
            results[backend] = {"failed": True, "error": f"{type(e).__name__}: {e}"}
            continue
        res["backend"] = backend
        if backend == "st_pool":
            res["pool_workers"] = args.pool_workers
        results[backend] = res
        print(f"        rps={res['rps']:.2f}  p50={res['p50_ms']:.0f}ms  "
              f"p95={res['p95_ms']:.0f}ms  errors={res['errors']}")
        # Tear the pool down between runs so subsequent backends start clean.
        _reset_backend_singletons()

    print()
    header = f"{'backend':<14} {'rps':>8} {'p50 ms':>10} {'p95 ms':>10} {'p99 ms':>10} {'errors':>8}"
    print(header)
    print("-" * len(header))
    for backend in targets:
        r = results[backend]
        if r.get("skipped"):
            print(f"{backend:<14} {'(skipped)':>8}  {r.get('reason', '')}")
            continue
        if r.get("failed"):
            print(f"{backend:<14} {'(failed)':>8}  {r.get('error', '')}")
            continue
        print(f"{backend:<14} {r['rps']:>8.2f} {r['p50_ms']:>10.1f} "
              f"{r['p95_ms']:>10.1f} {r['p99_ms']:>10.1f} {r['errors']:>8d}")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": _utcnow_iso(),
        "db": str(db_path),
        "workers": args.workers,
        "seconds": args.seconds,
        "mode": args.mode,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nreport: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
