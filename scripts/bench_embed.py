#!/usr/bin/env python
"""Microbenchmark for memoirs query embedding.

Compares three flavors of embed:
  1. ``sentence-transformers`` raw (no cache)
  2. ``sentence-transformers`` with the LRU cache (``embed_text_cached``)
  3. ``fastembed`` (raw, if installed)

Workload: 100 queries — 50 unique strings each emitted twice. The LRU
column should be roughly ``0.5 × raw + 0.5 × cache_hit`` since half the
calls are warm. The fastembed column is skipped with a friendly message
if the package is not installed.

Usage::

    python scripts/bench_embed.py [--n 100] [--unique 50]

Reports p50 / p95 in milliseconds.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

# Allow running from a checkout without `pip install -e .`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memoirs.engine import embeddings as emb  # noqa: E402


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------


_TOPICS = [
    "memoirs MCP server architecture",
    "sqlite-vec ANN search latency",
    "sentence-transformers all-MiniLM-L6-v2 dim 384",
    "Zettelkasten autolinks via top-k cosine",
    "BM25 + dense hybrid retrieval RRF fusion",
    "consolidate pending candidates from queue",
    "Ebbinghaus forgetting curve in MemoryBank",
    "PPR HippoRAG multi-hop entity graph",
    "Streaming SSE assemble_context generator",
    "lifecycle promote demote archive policies",
    "user prefers Python over Go for prototyping",
    "Gemma 2 2B Q4_K_M GGUF quantized model",
    "FastAPI 0.136 pydantic 2.13 TypeError annotation",
    "spaCy NER extractor + heuristic fallback",
    "Vulkan AMD Radeon 890M offload llama-cpp",
    "presidio PII redaction pipeline regex",
    "scoring weights importance confidence recency",
    "stable_id content_hash deterministic id helper",
    "MCP get_context conflict resolved tokens",
    "memory candidate promotion to memory row",
    "vec_memories virtual table upsert pattern",
    "FTS5 sync triggers on memories insert update delete",
    "time-travel queries with as_of timestamp parameter",
    "decision contradiction detection negation polarity",
    "auto_merge_near_duplicates threshold 0.92",
    "metadata_json contradiction field flag",
    "chat-blueprint Layer 5 reasoning module",
    "user signal feedback boost score",
    "session-id retrieval mechanism design",
    "hash-based template cache for snippets",
    "older snapshots participate in cache invalidation",
    "fastfn local-first memory engine project",
    "long-term memory survey papers 2023 2026",
    "reflection daemon distill top-100 insights",
    "watchdog filesystem watcher debounce",
    "stable embed model swap invalidates index",
    "GitHub PAT lives in 1Password vault entry",
    "Claude Code chat export ingester pipeline",
    "Cursor IDE conversation export format",
    "GAP doc tracks open product gaps",
    "STATUS doc summarizes weekly progress",
    "audit corpus integrity check tool",
    "extract_pending throttle CPU and memory",
    "migration 002 introduces memory_links table",
    "Jaccard similarity on shared entities Zettelkasten",
    "rebuild fts index backfill maintenance task",
    "low-value archive percentile 10 dynamic threshold",
    "summarize_thread compresses 50+ messages durable",
    "score_feedback marks stale memory useful=false",
    "credential_pointer type for secrets references",
]


def _make_workload(n: int, unique: int) -> list[str]:
    """Return a list of ``n`` queries drawn from ``unique`` distinct strings.

    The first ``unique`` entries are unique; the next ``n-unique`` are repeats
    of the first half (so cached runs see ~50% hits).
    """
    if unique > len(_TOPICS):
        unique = len(_TOPICS)
    base = _TOPICS[:unique]
    repeat = base[: n - unique] if n > unique else []
    return list(base) + list(repeat)


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------


def _time_calls(fn: Callable[[str], object], queries: Iterable[str]) -> list[float]:
    """Return per-call wall-clock latencies in milliseconds."""
    out: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        fn(q)
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def _percentile(samples: list[float], p: float) -> float:
    if not samples:
        return float("nan")
    s = sorted(samples)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def _summarize(samples: list[float]) -> tuple[float, float, float]:
    """Return (mean, p50, p95) in ms."""
    if not samples:
        return float("nan"), float("nan"), float("nan")
    return statistics.fmean(samples), _percentile(samples, 50), _percentile(samples, 95)


# ---------------------------------------------------------------------------
# Bench drivers
# ---------------------------------------------------------------------------


def bench_raw(queries: list[str]) -> list[float]:
    """sentence-transformers raw — no caching layer."""
    # Warm the model so cold-load doesn't poison the first sample.
    emb.embed_text(queries[0])
    return _time_calls(emb.embed_text, queries)


def bench_cached(queries: list[str]) -> list[float]:
    """sentence-transformers + LRU. Exercises both miss and hit paths."""
    emb.clear_embed_cache()
    emb.embed_text_cached(queries[0])  # warm model + cache slot
    emb.clear_embed_cache()
    return _time_calls(emb.embed_text_cached, queries)


def bench_fastembed(queries: list[str]) -> list[float] | None:
    """fastembed raw, if available. Returns ``None`` when not installed."""
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return None
    import os
    prev = os.environ.get("MEMOIRS_EMBED_BACKEND")
    os.environ["MEMOIRS_EMBED_BACKEND"] = "fastembed"
    # Reset singleton so the new backend takes effect for this run.
    emb._FASTEMBED_SINGLETON = None  # type: ignore[attr-defined]
    try:
        emb.embed_text(queries[0])  # warm model
        samples = _time_calls(emb.embed_text, queries)
    finally:
        if prev is None:
            os.environ.pop("MEMOIRS_EMBED_BACKEND", None)
        else:
            os.environ["MEMOIRS_EMBED_BACKEND"] = prev
    return samples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="total query count (default: 100)")
    parser.add_argument("--unique", type=int, default=50, help="unique strings (default: 50)")
    args = parser.parse_args()

    queries = _make_workload(args.n, args.unique)
    print(f"workload: {len(queries)} queries, {args.unique} unique, {len(queries) - args.unique} repeats")
    print()

    rows: list[tuple[str, list[float] | None]] = []

    print("running: sentence-transformers (raw) ...", flush=True)
    rows.append(("sentence-transformers (raw)", bench_raw(queries)))

    print("running: sentence-transformers + LRU ...", flush=True)
    rows.append(("sentence-transformers + LRU", bench_cached(queries)))

    print("running: fastembed (raw) ...", flush=True)
    rows.append(("fastembed (raw)", bench_fastembed(queries)))

    print()
    header = f"{'backend':<32} {'mean ms':>10} {'p50 ms':>10} {'p95 ms':>10}"
    print(header)
    print("-" * len(header))
    for name, samples in rows:
        if samples is None:
            print(f"{name:<32} {'(not installed)':>32}")
            continue
        mean, p50, p95 = _summarize(samples)
        print(f"{name:<32} {mean:>10.4f} {p50:>10.4f} {p95:>10.4f}")
    print()

    # Cache-hit-only profile: every query a hit. Useful sanity check on the
    # claim that hits are ~60× faster than the model call.
    emb.clear_embed_cache()
    for q in queries[: args.unique]:
        emb.embed_text_cached(q)  # populate
    hit_samples = _time_calls(emb.embed_text_cached, queries[: args.unique])
    mean, p50, p95 = _summarize(hit_samples)
    print(f"{'LRU cache HIT only':<32} {mean:>10.4f} {p50:>10.4f} {p95:>10.4f}  (n={len(hit_samples)})")
    print()
    print("cache info:", emb.embed_cache_info())
    return 0


if __name__ == "__main__":
    sys.exit(main())
