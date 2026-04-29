"""Microbenchmark: SQLite PRAGMA tuning vs defaults on retrieval latency.

Runs `assemble_context(..., retrieval_mode="hybrid")` against a memoirs DB
under two configurations:

  1. "default" — only the bare-minimum WAL + foreign_keys + busy_timeout
                 (mmap_size=0, temp_store=file, cache_size=-2000 i.e. 2 MiB).
  2. "tuned"   — the full `_apply_pragmas` set: mmap_size=256 MiB,
                 temp_store=MEMORY, cache_size=-65536 (64 MiB).

For each config, runs N warmups + M timed iterations of 5 representative
queries and reports p50 / p95 / max in milliseconds.

Usage:
  .venv/bin/python scripts/bench_pragmas.py
  MEMOIRS_DB=/path/to/memoirs.sqlite .venv/bin/python scripts/bench_pragmas.py
  .venv/bin/python scripts/bench_pragmas.py --iterations 20 --warmup 3

Important: the script opens the live DB read-mostly (assemble_context bumps
usage_count, so it does write — but only a handful of UPDATEs per query).
If you want a pristine bench, copy the DB first.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import statistics
import sys
import time
from pathlib import Path

# Make sure we import from the repo, not site-packages.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memoirs.db import MemoirsDB, _apply_pragmas  # noqa: E402

QUERIES = [
    "memoirs MCP server",
    "gocracker hash",
    "gemma extract daemon",
    "cursor ingestion",
    "feedback score",
]


def _apply_default_pragmas(conn: sqlite3.Connection) -> None:
    """Minimal baseline: WAL + busy_timeout + foreign_keys.

    Mirrors the *pre-tuning* state described in the task prompt:
      mmap_size  = 0
      temp_store = 0 (default = file)
      cache_size = -2000 (= 2 MiB, SQLite default)
    """
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    # Force the SQLite defaults so we measure honestly even if the DB header
    # has been touched by a previous run.
    conn.execute("PRAGMA mmap_size = 0")
    conn.execute("PRAGMA temp_store = 0")
    conn.execute("PRAGMA cache_size = -2000")


def _open(db_path: Path, *, tuned: bool) -> MemoirsDB:
    """Open a MemoirsDB without re-running migrations, then re-apply PRAGMAs."""
    db = MemoirsDB(db_path, auto_migrate=False)
    if tuned:
        _apply_pragmas(db.conn)  # the new defaults
    else:
        _apply_default_pragmas(db.conn)
    return db


def _time_query(db: MemoirsDB, query: str) -> float:
    """Run one assemble_context call, return wall time in milliseconds."""
    from memoirs.engine.memory_engine import assemble_context
    t0 = time.perf_counter()
    assemble_context(db, query, top_k=20, max_lines=15, retrieval_mode="hybrid")
    return (time.perf_counter() - t0) * 1000.0


def _bench(db_path: Path, *, tuned: bool, warmup: int, iterations: int) -> dict:
    """Returns {query: [latencies_ms...]} plus an aggregate "_all" key."""
    db = _open(db_path, tuned=tuned)
    try:
        results: dict[str, list[float]] = {q: [] for q in QUERIES}
        # Warm-up — fills page cache, JIT-compiles SQL, lets the OS settle.
        for _ in range(warmup):
            for q in QUERIES:
                _time_query(db, q)
        # Timed
        for _ in range(iterations):
            for q in QUERIES:
                results[q].append(_time_query(db, q))
        return results
    finally:
        db.close()


def _stats(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0, "n": 0}
    s = sorted(samples)
    n = len(s)
    p50 = statistics.median(s)
    # nearest-rank p95 (good enough for tiny n)
    p95_idx = max(0, min(n - 1, int(round(0.95 * (n - 1)))))
    p95 = s[p95_idx]
    return {"p50": p50, "p95": p95, "max": max(s), "n": n}


def _format_table(default_res: dict, tuned_res: dict) -> str:
    """Build a side-by-side comparison table."""
    header = f"{'query':<28} {'p50 def':>9} {'p50 tun':>9} {'Δp50%':>7}   {'p95 def':>9} {'p95 tun':>9} {'Δp95%':>7}"
    sep = "-" * len(header)
    lines = [header, sep]
    all_def: list[float] = []
    all_tun: list[float] = []
    for q in QUERIES:
        d = _stats(default_res[q])
        t = _stats(tuned_res[q])
        all_def.extend(default_res[q])
        all_tun.extend(tuned_res[q])
        d50, t50 = d["p50"], t["p50"]
        d95, t95 = d["p95"], t["p95"]
        delta50 = ((t50 - d50) / d50 * 100.0) if d50 else 0.0
        delta95 = ((t95 - d95) / d95 * 100.0) if d95 else 0.0
        lines.append(
            f"{q:<28} {d50:>8.1f}ms {t50:>8.1f}ms {delta50:>+6.1f}%   "
            f"{d95:>8.1f}ms {t95:>8.1f}ms {delta95:>+6.1f}%"
        )
    lines.append(sep)
    da = _stats(all_def)
    ta = _stats(all_tun)
    delta50 = ((ta["p50"] - da["p50"]) / da["p50"] * 100.0) if da["p50"] else 0.0
    delta95 = ((ta["p95"] - da["p95"]) / da["p95"] * 100.0) if da["p95"] else 0.0
    lines.append(
        f"{'OVERALL':<28} {da['p50']:>8.1f}ms {ta['p50']:>8.1f}ms {delta50:>+6.1f}%   "
        f"{da['p95']:>8.1f}ms {ta['p95']:>8.1f}ms {delta95:>+6.1f}%"
    )
    lines.append(
        f"{'  max':<28} {da['max']:>8.1f}ms {ta['max']:>8.1f}ms"
    )
    return "\n".join(lines)


def _resolve_db_path(arg: str | None) -> Path:
    if arg:
        p = Path(arg).expanduser().resolve()
    else:
        env = os.environ.get("MEMOIRS_DB")
        p = Path(env).expanduser().resolve() if env else (ROOT / ".memoirs" / "memoirs.sqlite")
    if not p.exists():
        raise SystemExit(f"DB not found: {p}\nSet MEMOIRS_DB=... or pass --db")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", help="path to memoirs.sqlite (default: $MEMOIRS_DB or .memoirs/memoirs.sqlite)")
    ap.add_argument("--iterations", type=int, default=10, help="timed iterations per query (default 10)")
    ap.add_argument("--warmup", type=int, default=2, help="warmup iterations per query (default 2)")
    args = ap.parse_args()

    db_path = _resolve_db_path(args.db)
    print(f"DB: {db_path}")
    print(f"queries: {len(QUERIES)}   warmup: {args.warmup}   iterations: {args.iterations}")
    print()

    print("[1/2] benchmarking DEFAULT (mmap=0, temp=file, cache=2MiB)...")
    default_res = _bench(db_path, tuned=False, warmup=args.warmup, iterations=args.iterations)

    print("[2/2] benchmarking TUNED   (mmap=256MiB, temp=MEMORY, cache=64MiB)...")
    tuned_res = _bench(db_path, tuned=True, warmup=args.warmup, iterations=args.iterations)

    print()
    print(_format_table(default_res, tuned_res))


if __name__ == "__main__":
    main()
