"""Eval harness — runs a suite of queries against memoirs retrieval and
reports retrieval-quality + latency metrics per mode.

Why a separate harness instead of pytest assertions?
  * Pytest is for *correctness*; this is for *quality regression*. We want
    JSON output, comparison across modes, and tables we can paste into
    blog posts / READMEs / GAP.md.
  * The same suite must run against arbitrary backends (current memoirs
    DB, time-traveled snapshot, future Mem0 / Zep adapters). A simple
    dataclass-driven runner is the cheapest way to make that happen.

Metrics (per query, then aggregated):
  * precision@k = |retrieved_top_k ∩ gold| / k
  * recall@k    = |retrieved_top_k ∩ gold| / |gold|
  * MRR         = 1 / rank_of_first_gold (0 if no gold in top_k)
  * hit@1, hit@5 = 1 if any gold in top-1 / top-5 else 0
  * time_to_first_relevant_ms — wall-clock from query start to the moment
    the first gold ID is observed in the ranked list. We approximate it by
    measuring total retrieval latency and only counting it when ≥1 gold
    appears in the result. Streaming retrieval (P4-1) could refine this
    later.
  * latency_p50_ms, latency_p95_ms — over all queries in the mode.

The harness deliberately does NOT depend on pytest, FastAPI, or anything
beyond the standard lib. It only imports memoirs internals it needs.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class EvalCase:
    """One eval query.

    `gold_memory_ids` is the set of memory IDs that *must* appear in the
    retrieved list for the case to be considered a hit. `category` lets us
    slice metrics by query type (single-hop / multi-hop / temporal /
    preference) when reporting. `as_of` triggers a time-travel query when
    set (matches the public ``assemble_context`` signature).
    """

    query: str
    gold_memory_ids: list[str]
    category: str = "single-hop"
    as_of: str | None = None
    # Free-form metadata, e.g. dataset-specific tags.
    notes: str = ""


@dataclass
class EvalSuite:
    """Named collection of cases.

    The name shows up in every report so multiple suites can be tracked in
    parallel (synthetic_basic, longmemeval_oracle, locomo_50turn, …).
    """

    name: str
    cases: list[EvalCase]
    description: str = ""


# ---------------------------------------------------------------------------
# Metric primitives — pure functions over (retrieved_ids, gold_ids).
#
# Kept as small free functions so they're trivially unit-testable and can
# be reused by adapter authors for sanity checks.
# ---------------------------------------------------------------------------


def precision_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """|top_k ∩ gold| / k. Returns 0.0 when k <= 0."""
    if k <= 0:
        return 0.0
    gold_set = set(gold)
    if not gold_set:
        # No gold means we can't penalize — but precision is undefined.
        # Following IR convention: return 0.0 to avoid hiding eval bugs.
        return 0.0
    top = list(retrieved)[:k]
    hits = sum(1 for m in top if m in gold_set)
    return hits / k


def recall_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """|top_k ∩ gold| / |gold|. Returns 0.0 when gold is empty."""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    top = list(retrieved)[:k]
    hits = sum(1 for m in top if m in gold_set)
    return hits / len(gold_set)


def mrr(retrieved: Sequence[str], gold: Iterable[str]) -> float:
    """Reciprocal rank of the FIRST gold ID in `retrieved`. 0 if absent.

    Standard IR definition. We don't average here — the aggregator does.
    """
    gold_set = set(gold)
    for i, mid in enumerate(retrieved, start=1):
        if mid in gold_set:
            return 1.0 / i
    return 0.0


def hit_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """1.0 if any gold appears in top-k, else 0.0."""
    gold_set = set(gold)
    return 1.0 if any(m in gold_set for m in list(retrieved)[:k]) else 0.0


def compute_metrics(
    retrieved: Sequence[str],
    gold: Iterable[str],
    *,
    top_k: int,
) -> dict[str, float]:
    """Compute every per-query metric in one shot.

    Returns a flat dict so downstream serialization stays trivial.
    """
    return {
        "precision_at_k": precision_at_k(retrieved, gold, top_k),
        "recall_at_k": recall_at_k(retrieved, gold, top_k),
        "mrr": mrr(retrieved, gold),
        "hit_at_1": hit_at_k(retrieved, gold, 1),
        "hit_at_5": hit_at_k(retrieved, gold, 5),
    }


# ---------------------------------------------------------------------------
# Per-query result + per-mode aggregation
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """Outcome of running a single case under a single retrieval mode."""

    query: str
    category: str
    gold_memory_ids: list[str]
    retrieved_memory_ids: list[str]
    metrics: dict[str, float]
    latency_ms: float
    # Latency to first gold hit. We approximate this as the total retrieval
    # latency when ≥1 gold appears in the result; None if no gold landed.
    time_to_first_relevant_ms: float | None


@dataclass
class ModeResults:
    """Aggregate metrics for a retrieval mode across all cases."""

    mode: str
    top_k: int
    n_cases: int
    # Per-query results retained so callers can drill down / export.
    queries: list[QueryResult] = field(default_factory=list)
    # Aggregates — populated by `finalize()`.
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    hit_at_1: float = 0.0
    hit_at_5: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_mean_ms: float = 0.0
    time_to_first_relevant_ms_p50: float | None = None
    # Per-category mean precision / recall — handy for spotting weakness
    # on multi-hop / temporal queries.
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)

    def finalize(self) -> None:
        """Compute aggregates from the captured per-query results.

        Called once `queries` has been populated. Idempotent.
        """
        if not self.queries:
            return
        self.precision_at_k = _mean(q.metrics["precision_at_k"] for q in self.queries)
        self.recall_at_k = _mean(q.metrics["recall_at_k"] for q in self.queries)
        self.mrr = _mean(q.metrics["mrr"] for q in self.queries)
        self.hit_at_1 = _mean(q.metrics["hit_at_1"] for q in self.queries)
        self.hit_at_5 = _mean(q.metrics["hit_at_5"] for q in self.queries)
        latencies = [q.latency_ms for q in self.queries]
        self.latency_p50_ms = _percentile(latencies, 50)
        self.latency_p95_ms = _percentile(latencies, 95)
        self.latency_mean_ms = _mean(latencies)
        ttfr = [q.time_to_first_relevant_ms for q in self.queries
                if q.time_to_first_relevant_ms is not None]
        self.time_to_first_relevant_ms_p50 = _percentile(ttfr, 50) if ttfr else None
        # By-category breakdown
        cats: dict[str, list[QueryResult]] = {}
        for q in self.queries:
            cats.setdefault(q.category, []).append(q)
        self.by_category = {
            cat: {
                "n": len(qs),
                "precision_at_k": _mean(q.metrics["precision_at_k"] for q in qs),
                "recall_at_k": _mean(q.metrics["recall_at_k"] for q in qs),
                "mrr": _mean(q.metrics["mrr"] for q in qs),
                "hit_at_1": _mean(q.metrics["hit_at_1"] for q in qs),
            }
            for cat, qs in cats.items()
        }


@dataclass
class EvalResults:
    """Top-level results bundle: suite + per-mode aggregates.

    JSON-serializable through ``to_json`` / ``from_json``. Use ``print_table``
    for a human-readable summary.
    """

    suite_name: str
    top_k: int
    modes: list[ModeResults] = field(default_factory=list)
    # Free-form info: timestamp, git SHA, env vars, etc.
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_table(self, *, file=None) -> None:
        """Pretty per-mode table to stdout (or `file`).

        The table is ASCII-only so it stays readable in terminals, CI logs,
        and emails. Numbers are rounded to 4 decimals for retrieval
        metrics, 1 for latency.
        """
        import sys
        out = file or sys.stdout
        cols = [
            ("mode",      9),
            ("P@k",       7),
            ("R@k",       7),
            ("MRR",       7),
            ("Hit@1",     7),
            ("Hit@5",     7),
            ("p50ms",     8),
            ("p95ms",     8),
            ("ttfr_p50",  9),
        ]
        header = " | ".join(name.ljust(w) for name, w in cols)
        sep = "-+-".join("-" * w for _, w in cols)
        title = f"=== Eval results: {self.suite_name} (top_k={self.top_k}) ==="
        print(title, file=out)
        print(header, file=out)
        print(sep, file=out)
        for m in self.modes:
            ttfr = (
                f"{m.time_to_first_relevant_ms_p50:.1f}"
                if m.time_to_first_relevant_ms_p50 is not None else "-"
            )
            row = [
                m.mode.ljust(cols[0][1]),
                f"{m.precision_at_k:.4f}".ljust(cols[1][1]),
                f"{m.recall_at_k:.4f}".ljust(cols[2][1]),
                f"{m.mrr:.4f}".ljust(cols[3][1]),
                f"{m.hit_at_1:.4f}".ljust(cols[4][1]),
                f"{m.hit_at_5:.4f}".ljust(cols[5][1]),
                f"{m.latency_p50_ms:.1f}".ljust(cols[6][1]),
                f"{m.latency_p95_ms:.1f}".ljust(cols[7][1]),
                ttfr.ljust(cols[8][1]),
            ]
            print(" | ".join(row), file=out)
        # Per-category breakdown for the first mode (compact: one block).
        if self.modes:
            print("", file=out)
            for m in self.modes:
                if not m.by_category:
                    continue
                print(f"-- {m.mode}: by category --", file=out)
                for cat, vals in sorted(m.by_category.items()):
                    print(
                        f"  {cat:<12} n={int(vals['n']):<3} "
                        f"P@k={vals['precision_at_k']:.3f} "
                        f"R@k={vals['recall_at_k']:.3f} "
                        f"MRR={vals['mrr']:.3f} "
                        f"Hit@1={vals['hit_at_1']:.3f}",
                        file=out,
                    )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialize to a JSON string. Optionally write to `path`."""
        payload = {
            "suite_name": self.suite_name,
            "top_k": self.top_k,
            "meta": self.meta,
            "modes": [_mode_to_dict(m) for m in self.modes],
        }
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, path: str | Path) -> "EvalResults":
        """Load a previously saved EvalResults JSON. Round-trips ``to_json``."""
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalResults":
        modes = [_mode_from_dict(m) for m in data.get("modes", [])]
        return cls(
            suite_name=data["suite_name"],
            top_k=int(data["top_k"]),
            modes=modes,
            meta=dict(data.get("meta", {})),
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


# Default modes — mirrors the public retrieval modes wired through
# `_resolve_retrieval_mode` in memoirs.engine.memory_engine.
DEFAULT_MODES: tuple[str, ...] = ("hybrid", "dense", "bm25")


def run_eval(
    db,
    suite: EvalSuite,
    *,
    top_k: int = 10,
    retrieval_modes: Sequence[str] = DEFAULT_MODES,
    use_assemble_context: bool = False,
) -> EvalResults:
    """Run every case in `suite` under each retrieval mode and aggregate.

    Two retrieval paths are supported:

    * Default (``use_assemble_context=False``) → calls the lower-level
      ``_retrieve_candidates`` so we measure pure retrieval latency without
      the conflict-resolver / compress overhead. This is the apples-to-
      apples mode for comparing memoirs to Mem0 / Zep retrieval numbers.

    * ``use_assemble_context=True`` → runs the full ``assemble_context``
      pipeline (retrieval + scoring + conflict resolution + compression),
      i.e. what an MCP client actually sees. The retrieved IDs come from
      ``result["memories"]``.

    The latency clock starts before the retrieval call and stops as soon
    as we have the ranked ID list. ``time_to_first_relevant_ms`` is the
    same wall-clock time, but reported only when ≥1 gold ID appears in
    the ranking — otherwise None (so the aggregator can ignore it).
    """
    # Lazy import: keeps `from memoirs.evals import EvalCase` zero-cost.
    from memoirs.engine import memory_engine as me

    results = EvalResults(
        suite_name=suite.name,
        top_k=top_k,
        modes=[],
        meta={
            "n_cases": len(suite.cases),
            "modes": list(retrieval_modes),
            "use_assemble_context": use_assemble_context,
            "description": suite.description,
        },
    )

    for mode in retrieval_modes:
        mode_res = ModeResults(mode=mode, top_k=top_k, n_cases=len(suite.cases))
        for case in suite.cases:
            t0 = time.perf_counter()
            try:
                if use_assemble_context:
                    payload = me.assemble_context(
                        db, case.query,
                        top_k=top_k, max_lines=top_k,
                        as_of=case.as_of,
                        retrieval_mode=mode,
                    )
                    retrieved = [m["id"] for m in payload.get("memories", [])]
                else:
                    candidates = me._retrieve_candidates(
                        db, case.query,
                        top_k=top_k, as_of=case.as_of, mode=mode,
                    )
                    retrieved = [c["id"] for c in candidates]
            except Exception as e:
                # Don't let one bad mode crash the whole eval — record an
                # empty retrieval so metrics still aggregate. The error is
                # surfaced via meta so callers can see it.
                retrieved = []
                mode_res_meta = results.meta.setdefault("errors", [])
                mode_res_meta.append({
                    "mode": mode, "query": case.query, "error": repr(e),
                })
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            metrics = compute_metrics(retrieved, case.gold_memory_ids, top_k=top_k)
            ttfr = elapsed_ms if metrics["hit_at_5"] or any(
                gid in retrieved for gid in case.gold_memory_ids
            ) else None
            mode_res.queries.append(QueryResult(
                query=case.query,
                category=case.category,
                gold_memory_ids=list(case.gold_memory_ids),
                retrieved_memory_ids=retrieved[:top_k],
                metrics=metrics,
                latency_ms=elapsed_ms,
                time_to_first_relevant_ms=ttfr,
            ))
        mode_res.finalize()
        results.modes.append(mode_res)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mean(xs: Iterable[float]) -> float:
    xs = [float(x) for x in xs]
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def _percentile(xs: Sequence[float], p: int) -> float:
    """p in [0,100]. Uses statistics.quantiles for n>=2, lerp at endpoints.

    For very small n we fall back to ``max`` / ``min`` so the function
    always returns a number (avoids special-casing in callers). p must
    be 0..100 inclusive.
    """
    xs = sorted(float(x) for x in xs)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    # Linear interpolation between adjacent ranks (numpy-equivalent).
    frac = p / 100.0
    idx = frac * (len(xs) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(xs) - 1)
    weight = idx - lo
    return xs[lo] * (1 - weight) + xs[hi] * weight


def _mode_to_dict(m: ModeResults) -> dict[str, Any]:
    return {
        "mode": m.mode,
        "top_k": m.top_k,
        "n_cases": m.n_cases,
        "precision_at_k": m.precision_at_k,
        "recall_at_k": m.recall_at_k,
        "mrr": m.mrr,
        "hit_at_1": m.hit_at_1,
        "hit_at_5": m.hit_at_5,
        "latency_p50_ms": m.latency_p50_ms,
        "latency_p95_ms": m.latency_p95_ms,
        "latency_mean_ms": m.latency_mean_ms,
        "time_to_first_relevant_ms_p50": m.time_to_first_relevant_ms_p50,
        "by_category": m.by_category,
        "queries": [asdict(q) for q in m.queries],
    }


def _mode_from_dict(d: dict[str, Any]) -> ModeResults:
    m = ModeResults(
        mode=d["mode"],
        top_k=int(d["top_k"]),
        n_cases=int(d.get("n_cases", 0)),
        precision_at_k=float(d.get("precision_at_k", 0.0)),
        recall_at_k=float(d.get("recall_at_k", 0.0)),
        mrr=float(d.get("mrr", 0.0)),
        hit_at_1=float(d.get("hit_at_1", 0.0)),
        hit_at_5=float(d.get("hit_at_5", 0.0)),
        latency_p50_ms=float(d.get("latency_p50_ms", 0.0)),
        latency_p95_ms=float(d.get("latency_p95_ms", 0.0)),
        latency_mean_ms=float(d.get("latency_mean_ms", 0.0)),
        time_to_first_relevant_ms_p50=d.get("time_to_first_relevant_ms_p50"),
        by_category=dict(d.get("by_category", {})),
    )
    for q in d.get("queries", []):
        m.queries.append(QueryResult(
            query=q["query"],
            category=q.get("category", "single-hop"),
            gold_memory_ids=list(q.get("gold_memory_ids", [])),
            retrieved_memory_ids=list(q.get("retrieved_memory_ids", [])),
            metrics=dict(q.get("metrics", {})),
            latency_ms=float(q.get("latency_ms", 0.0)),
            time_to_first_relevant_ms=q.get("time_to_first_relevant_ms"),
        ))
    return m
