"""Phase 5E head-to-head bench: memoirs vs Mem0 vs Zep vs Letta vs Cognee.

Runs the same synthetic dataset against every locally-reachable engine
and emits both a JSON artifact and a human-readable markdown table.

Why one script instead of pytest?
  * Docker-driven engines need to be brought up / torn down. That kind
    of side-effect doesn't belong in pytest's per-process model.
  * The output IS a deliverable (table goes into docs / GAP.md), so we
    want a CLI we can invoke ad-hoc and pipe to a file.
  * Engines that can't start should produce a SKIP row, not a test
    failure — the bench should run end-to-end on any machine.

Wire-up:
  * Dataset comes from `scripts.bench_vs_others_dataset.build_dataset()`.
  * Metric primitives reused from `memoirs.evals.harness.compute_metrics`
    (no duplication).
  * Adapters live in `scripts.adapters.*`. Each is a stand-alone module
    so adding a new engine = adding one file + registering it here.

CLI:
  python scripts/bench_vs_others.py
  python scripts/bench_vs_others.py --engines memoirs,cognee
  python scripts/bench_vs_others.py --top-k 10 --out /tmp/report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# Make sure `scripts.*` resolves regardless of where Python is invoked from.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from memoirs.evals.harness import compute_metrics
from scripts.adapters._ollama import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_MODEL,
    apply_ollama_env,
    ollama_install_hint,
    ollama_is_up,
)
from scripts.adapters.base import EngineAdapter
from scripts.bench_vs_others_dataset import (
    BenchConversation,
    BenchDataset,
    BenchMemory,
    BenchQuery,
    BenchSuite,
    build_dataset,
    build_end_to_end_suite,
)


log = logging.getLogger("bench_vs_others")


# ---------------------------------------------------------------------------
# Per-engine result containers
# ---------------------------------------------------------------------------


@dataclass
class EngineQueryResult:
    """One query × one engine."""

    query: str
    category: str
    gold_memory_ids: list[str]
    retrieved_memory_ids: list[str]
    metrics: dict[str, float]
    latency_ms: float


@dataclass
class EngineReport:
    """All results + aggregates for one engine.

    `status` is "OK" when the adapter ingested + answered queries; any
    other value is a SKIP reason that ends up in the markdown row.
    """

    engine: str
    status: str = "OK"
    n_cases: int = 0
    queries: list[EngineQueryResult] = field(default_factory=list)
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    hit_at_1: float = 0.0
    hit_at_5: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_mean_ms: float = 0.0
    ram_peak_mb: float = 0.0
    ingest_seconds: float = 0.0
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    # End-to-end-only fields (zero in retrieval-only mode).
    mode: str = "retrieval-only"
    tokens_used: int = 0
    ingest_p50_ms: float = 0.0


# ---------------------------------------------------------------------------
# RAM probe (best-effort)
# ---------------------------------------------------------------------------


def _ram_peak_mb() -> float:
    """Return current process peak RSS in MB.

    Uses ``resource.getrusage`` which is portable across Linux/macOS.
    On Linux the value is KiB; on macOS it's bytes — we normalise.
    Returns 0.0 if the resource module is unavailable (Windows).
    """
    try:
        import resource
    except ImportError:  # pragma: no cover — non-POSIX
        return 0.0
    ru = resource.getrusage(resource.RUSAGE_SELF)
    raw = float(ru.ru_maxrss)
    # Heuristic: macOS reports bytes (>1e9 for any real process), Linux KiB.
    if raw > 1e9:
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0


def _estimate_tokens(text: str) -> int:
    """Best-effort token count using tiktoken, falling back to a 4-chars
    heuristic when the package is unavailable.

    Used in end-to-end mode to attribute LLM cost to each engine — it's
    a *floor* for memoirs (which uses Qwen GGUF locally so we can't
    introspect provider tokens) and a tight estimate for engines that
    forward conversations to OpenAI verbatim.
    """
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Cheap fallback: ~4 chars per token for English text.
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _percentile(xs: list[float], p: int) -> float:
    """Linear-interpolation percentile, p in [0,100]. 0.0 when empty."""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    frac = p / 100.0
    idx = frac * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    weight = idx - lo
    return s[lo] * (1 - weight) + s[hi] * weight


def _aggregate(report: EngineReport) -> None:
    """Populate aggregate numbers from `report.queries`. Idempotent."""
    qs = report.queries
    if not qs:
        return
    report.precision_at_k = _mean([q.metrics["precision_at_k"] for q in qs])
    report.recall_at_k = _mean([q.metrics["recall_at_k"] for q in qs])
    report.mrr = _mean([q.metrics["mrr"] for q in qs])
    report.hit_at_1 = _mean([q.metrics["hit_at_1"] for q in qs])
    report.hit_at_5 = _mean([q.metrics["hit_at_5"] for q in qs])
    lats = [q.latency_ms for q in qs]
    report.latency_p50_ms = _percentile(lats, 50)
    report.latency_p95_ms = _percentile(lats, 95)
    report.latency_mean_ms = _mean(lats)
    cats: dict[str, list[EngineQueryResult]] = {}
    for q in qs:
        cats.setdefault(q.category, []).append(q)
    report.by_category = {
        cat: {
            "n": float(len(items)),
            "precision_at_k": _mean([q.metrics["precision_at_k"] for q in items]),
            "recall_at_k": _mean([q.metrics["recall_at_k"] for q in items]),
            "mrr": _mean([q.metrics["mrr"] for q in items]),
            "hit_at_1": _mean([q.metrics["hit_at_1"] for q in items]),
        }
        for cat, items in cats.items()
    }


# ---------------------------------------------------------------------------
# Engine runner
# ---------------------------------------------------------------------------


def run_engine(
    adapter: EngineAdapter,
    dataset: BenchDataset,
    *,
    top_k: int,
) -> EngineReport:
    """Ingest + query one engine; always returns a report (never raises).

    The runner catches exceptions per-stage so one engine going down
    doesn't poison the bench. Status flips to a human-readable reason
    string for the markdown table.
    """
    rep = EngineReport(engine=adapter.name, n_cases=len(dataset.queries),
                       mode="retrieval-only")
    if not adapter.status.ok:
        rep.status = f"SKIP ({adapter.status.reason})"
        return rep

    # ----- Ingest -----
    t0 = time.perf_counter()
    try:
        adapter.add_memories(dataset.memories)
    except Exception as e:
        log.warning("%s: ingest failed: %s", adapter.name, e)
        rep.status = f"SKIP (ingest failed: {e!r})"
        return rep
    rep.ingest_seconds = time.perf_counter() - t0

    # ----- Query -----
    for q in dataset.queries:
        t1 = time.perf_counter()
        try:
            retrieved = adapter.query(q, top_k=top_k)
        except Exception as e:
            log.warning("%s: query %r failed: %s", adapter.name, q.query, e)
            retrieved = []
        latency_ms = (time.perf_counter() - t1) * 1000.0
        metrics = compute_metrics(retrieved, q.gold_memory_ids, top_k=top_k)
        rep.queries.append(EngineQueryResult(
            query=q.query,
            category=q.category,
            gold_memory_ids=list(q.gold_memory_ids),
            retrieved_memory_ids=list(retrieved)[:top_k],
            metrics=metrics,
            latency_ms=latency_ms,
        ))
    _aggregate(rep)
    rep.ram_peak_mb = _ram_peak_mb()
    return rep


def run_engine_end_to_end(
    adapter: EngineAdapter,
    suite: BenchSuite,
    *,
    top_k: int,
) -> EngineReport:
    """End-to-end runner: raw conversations → engine pipeline → query.

    For each conversation we time `ingest_conversation` independently
    so the report can show p50 ingest latency. After all conversations
    are ingested, queries run normally — but retrieved memory IDs are
    translated back to *conversation* IDs via `adapter.resolve_conv_id`
    before metric computation. That makes "did the engine extract a
    memory traceable to the gold conversation?" the success criterion.

    Engines that do NOT override `ingest_conversation` get the default
    fallback (one synthetic memoria per message) — the bench tags this
    in the report as ``mode='e2e-standalone'`` so the reader knows the
    engine wasn't put through its real pipeline.
    """
    mode = ("native" if getattr(adapter, "supports_native_ingest", False)
            else "standalone")
    rep = EngineReport(engine=adapter.name, n_cases=len(suite.queries),
                       mode=f"e2e-{mode}")
    if not adapter.status.ok:
        rep.status = f"SKIP ({adapter.status.reason})"
        return rep

    # ----- Ingest conversations -----
    ingest_latencies: list[float] = []
    tokens_total = 0
    t0 = time.perf_counter()
    for conv in suite.conversations:
        # Account every message text as input tokens. This is the floor
        # — engines may add system prompts on top.
        tokens_total += sum(
            _estimate_tokens(m.get("content") or "") for m in conv.messages
        )
        ti = time.perf_counter()
        try:
            adapter.ingest_conversation(conv)
        except Exception as e:
            log.warning("%s: ingest_conversation %s failed: %s",
                        adapter.name, conv.id, e)
            rep.status = f"SKIP (ingest_conversation failed: {e!r})"
            return rep
        ingest_latencies.append((time.perf_counter() - ti) * 1000.0)
        if not adapter.status.ok:
            rep.status = f"SKIP ({adapter.status.reason})"
            return rep
    rep.ingest_seconds = time.perf_counter() - t0
    rep.tokens_used = tokens_total
    rep.ingest_p50_ms = _percentile(ingest_latencies, 50)

    # ----- Query (translate memory ids → conversation ids) -----
    for q in suite.queries:
        t1 = time.perf_counter()
        try:
            retrieved_raw = adapter.query(q, top_k=top_k)
        except Exception as e:
            log.warning("%s: query %r failed: %s", adapter.name, q.query, e)
            retrieved_raw = []
        latency_ms = (time.perf_counter() - t1) * 1000.0
        seen_conv: list[str] = []
        seen_set: set[str] = set()
        for mid in retrieved_raw:
            cid = adapter.resolve_conv_id(mid) or mid
            if cid in seen_set:
                continue
            seen_set.add(cid)
            seen_conv.append(cid)
        metrics = compute_metrics(seen_conv, q.gold_memory_ids, top_k=top_k)
        rep.queries.append(EngineQueryResult(
            query=q.query,
            category=q.category,
            gold_memory_ids=list(q.gold_memory_ids),
            retrieved_memory_ids=list(seen_conv)[:top_k],
            metrics=metrics,
            latency_ms=latency_ms,
        ))
    _aggregate(rep)
    rep.ram_peak_mb = _ram_peak_mb()
    return rep


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_markdown(reports: list[EngineReport], *, top_k: int,
                    suite: str = "retrieval-only") -> str:
    """Render the full per-engine + per-category table as Markdown.

    `suite` is "retrieval-only" or "end-to-end"; the latter expands the
    table with `mode`, `p50 ingest`, and `tokens` columns so the reader
    can tell at a glance which engines went through their native pipeline
    and what the LLM cost looked like.
    """
    if suite == "end-to-end":
        head = (
            f"# bench_vs_others (suite=end-to-end, top_k={top_k})\n\n"
            "| engine    | mode             | MRR  | Hit@1 | Hit@5 |"
            " p50 query ms | p50 ingest ms | tokens | ram MB | status |\n"
            "|-----------|------------------|------|-------|-------|"
            "--------------|---------------|--------|--------|--------|\n"
        )
        rows: list[str] = []
        for r in reports:
            if r.status == "OK":
                rows.append(
                    f"| {r.engine:<9} | {r.mode:<16} | {r.mrr:.2f} | "
                    f"{r.hit_at_1:.2f}  | {r.hit_at_5:.2f}  | "
                    f"{r.latency_p50_ms:>12.1f} | "
                    f"{r.ingest_p50_ms:>13.1f} | "
                    f"{r.tokens_used:>6} | {r.ram_peak_mb:6.0f} | OK     |"
                )
            else:
                rows.append(
                    f"| {r.engine:<9} | {r.mode:<16} |  -   |   -   |   -   |"
                    "      -       |       -       |   -    |   -    |"
                    f" {r.status} |"
                )
        body = "\n".join(rows)
        # Per-category breakdown for engines that ran (shared with
        # retrieval-only).
        by_cat_blocks: list[str] = []
        for r in reports:
            if r.status != "OK" or not r.by_category:
                continue
            block = [f"\n## {r.engine} — by category", "",
                     "| category    | n | P@k  | R@k  | MRR  | Hit@1 |",
                     "|-------------|---|------|------|------|-------|"]
            for cat in sorted(r.by_category):
                v = r.by_category[cat]
                block.append(
                    f"| {cat:<11} | {int(v['n']):>1} | {v['precision_at_k']:.2f} | "
                    f"{v['recall_at_k']:.2f} | {v['mrr']:.2f} | {v['hit_at_1']:.2f}  |"
                )
            by_cat_blocks.append("\n".join(block))
        return head + body + "\n" + "\n".join(by_cat_blocks) + "\n"

    head = (
        f"# bench_vs_others (top_k={top_k})\n\n"
        "| engine    | P@k  | R@k  | MRR  | Hit@1 | Hit@5 | p50 ms | p95 ms | ram MB | status |\n"
        "|-----------|------|------|------|-------|-------|--------|--------|--------|--------|\n"
    )
    rows: list[str] = []
    for r in reports:
        if r.status == "OK":
            rows.append(
                f"| {r.engine:<9} | {r.precision_at_k:.2f} | {r.recall_at_k:.2f} | "
                f"{r.mrr:.2f} | {r.hit_at_1:.2f}  | {r.hit_at_5:.2f}  | "
                f"{r.latency_p50_ms:6.1f} | {r.latency_p95_ms:6.1f} | "
                f"{r.ram_peak_mb:6.0f} | OK     |"
            )
        else:
            rows.append(
                f"| {r.engine:<9} |  -   |  -   |  -   |   -   |   -   |   -    |   -    |"
                f"   -    | {r.status} |"
            )
    body = "\n".join(rows)
    # Per-category breakdown for engines that ran.
    by_cat_blocks: list[str] = []
    for r in reports:
        if r.status != "OK" or not r.by_category:
            continue
        block = [f"\n## {r.engine} — by category", "",
                 "| category    | n | P@k  | R@k  | MRR  | Hit@1 |",
                 "|-------------|---|------|------|------|-------|"]
        for cat in sorted(r.by_category):
            v = r.by_category[cat]
            block.append(
                f"| {cat:<11} | {int(v['n']):>1} | {v['precision_at_k']:.2f} | "
                f"{v['recall_at_k']:.2f} | {v['mrr']:.2f} | {v['hit_at_1']:.2f}  |"
            )
        by_cat_blocks.append("\n".join(block))
    return head + body + "\n" + "\n".join(by_cat_blocks) + "\n"


def serialize_report(reports: list[EngineReport], *, top_k: int,
                     suite: str = "retrieval-only") -> dict[str, Any]:
    return {
        "suite": suite,
        "top_k": top_k,
        "n_engines": len(reports),
        "n_engines_ok": sum(1 for r in reports if r.status == "OK"),
        "engines": [asdict(r) for r in reports],
    }


# ---------------------------------------------------------------------------
# Engine registry
# ---------------------------------------------------------------------------


def build_adapter(name: str, **kwargs) -> EngineAdapter:
    """Instantiate an adapter by name. Unknown name raises ValueError."""
    name = name.lower()
    if name == "memoirs":
        from scripts.adapters.memoirs_adapter import MemoirsAdapter
        return MemoirsAdapter(**kwargs)
    if name == "mem0":
        from scripts.adapters.mem0_adapter import Mem0Adapter
        return Mem0Adapter(**kwargs)
    if name == "cognee":
        from scripts.adapters.cognee_adapter import CogneeAdapter
        return CogneeAdapter(**kwargs)
    if name == "zep":
        from scripts.adapters.zep_adapter import ZepAdapter
        return ZepAdapter(**kwargs)
    if name == "letta":
        from scripts.adapters.letta_adapter import LettaAdapter
        return LettaAdapter(**kwargs)
    if name == "langmem":
        from scripts.adapters.langmem_adapter import LangMemAdapter
        return LangMemAdapter(**kwargs)
    if name == "llamaindex":
        from scripts.adapters.llamaindex_adapter import LlamaIndexAdapter
        return LlamaIndexAdapter(**kwargs)
    if name == "memori":
        from scripts.adapters.memori_adapter import MemoriAdapter
        return MemoriAdapter(**kwargs)
    raise ValueError(f"unknown engine: {name}")


DEFAULT_ENGINES = (
    "memoirs", "mem0", "zep", "letta", "cognee",
    "langmem", "llamaindex", "memori",
)


# ---------------------------------------------------------------------------
# LongMemEval bridge
#
# When --longmemeval is supplied with a JSONL path we reuse the existing
# `memoirs.evals.longmemeval_adapter.load_longmemeval` parser (no copy)
# and project its EvalCases onto the bench's BenchMemory/BenchQuery shape.
# Each evidence ID becomes a BenchMemory whose content is the corresponding
# turn text — that way every adapter can ingest the haystack with its
# normal `add_memories` path.
# ---------------------------------------------------------------------------


_DEFAULT_LME_PATHS = (
    "~/datasets/longmemeval/longmemeval_oracle.json",
    "~/datasets/longmemeval/longmemeval_s.json",
    "~/datasets/longmemeval/longmemeval_oracle.jsonl",
    "~/datasets/longmemeval/longmemeval_s.jsonl",
    "~/datasets/longmemeval.jsonl",
)


def _resolve_longmemeval_path(explicit: Optional[str]) -> Optional[Path]:
    """Return the first existing LongMemEval dump, or the user's choice."""
    candidates: list[str] = [explicit] if explicit else list(_DEFAULT_LME_PATHS)
    for cand in candidates:
        if not cand:
            continue
        p = Path(cand).expanduser()
        if p.exists():
            return p
    # Return the explicit path even if missing so the loader can produce
    # a nice "dataset not installed" reason.
    if explicit:
        return Path(explicit).expanduser()
    return None


def _build_longmemeval_dataset(jsonl_path: Path,
                                limit: Optional[int] = None) -> BenchDataset:
    """Load LongMemEval through the existing adapter and reshape to BenchDataset.

    Memory IDs are stable across calls (same evidence string → same ID)
    so a repeated run produces an identical corpus, which is what the
    adapters expect. We synthesize content for missing turns from the
    raw JSONL when it's available.
    """
    from memoirs.evals.longmemeval_adapter import load_longmemeval

    suite, info = load_longmemeval(jsonl_path, limit=limit)
    if suite is None:
        reason = info.get("reason", "longmemeval load failed")
        raise FileNotFoundError(reason)

    # Build a per-turn content index from the raw oracle JSON. Bound the
    # haystack to the same N records the EvalSuite kept (so a smoke run
    # with --longmemeval-limit 50 only embeds the haystack from those 50
    # cases, not all 500).
    id_to_content: dict[str, str] = {}
    try:
        with jsonl_path.open("r", encoding="utf-8") as fh:
            head = fh.read(4096)
            fh.seek(0)
            if head.lstrip().startswith("["):
                records = json.load(fh)
            else:
                records = []
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if limit is not None:
            records = records[:limit]
        for rec in records:
            sids = rec.get("haystack_session_ids") or []
            sessions = rec.get("haystack_sessions") or []
            for sid, turns in zip(sids, sessions):
                if not isinstance(turns, list):
                    continue
                for tidx, turn in enumerate(turns):
                    if not isinstance(turn, dict):
                        continue
                    role = turn.get("role", "user")
                    text = (turn.get("content") or "").strip()
                    if not text:
                        continue
                    id_to_content[f"{sid}:{tidx}"] = f"{role}: {text}"
    except Exception as e:  # pragma: no cover — best-effort
        log.warning("longmemeval: could not pre-load haystack: %s", e)

    # If the gold IDs from the adapter are session-level only (older split
    # without per-turn evidence), fall back to mapping each session_id to
    # its first non-empty turn's content so retrieval still has signal.
    memories: list[BenchMemory] = []
    seen_ids: set[str] = set()
    queries: list[BenchQuery] = []
    for case in suite.cases:
        gold: list[str] = []
        for gid in case.gold_memory_ids:
            sid = str(gid)
            if sid not in seen_ids:
                content = id_to_content.get(sid, sid)
                memories.append(BenchMemory(id=sid, type="event", content=content))
                seen_ids.add(sid)
            gold.append(sid)
        queries.append(BenchQuery(
            query=case.query,
            gold_memory_ids=gold,
            category=case.category,
            as_of=case.as_of,
            notes=case.notes,
        ))

    # Also ingest the FULL haystack as distractor memories, so the retrieval
    # task is genuinely "find the gold turns among hundreds of others".
    # Without distractors the engine just retrieves the only memory it has.
    for tid, content in id_to_content.items():
        if tid in seen_ids:
            continue
        memories.append(BenchMemory(id=tid, type="event", content=content))
        seen_ids.add(tid)

    log.info("longmemeval: %d cases / %d unique memorias from %s",
             len(queries), len(memories), jsonl_path)
    return BenchDataset(memories=memories, queries=queries)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--engines", default=",".join(DEFAULT_ENGINES),
        help="Comma-separated engine names to run (default: all).",
    )
    p.add_argument(
        "--suite", choices=("retrieval-only", "end-to-end"),
        default="retrieval-only",
        help=("'retrieval-only' (default): pre-curated memorias go in, "
              "every engine ranks them. 'end-to-end': raw conversations go "
              "in and each engine runs its OWN extract+consolidate pipeline."),
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--out",
        default=str(_ROOT / ".memoirs" / "bench_others_report.json"),
        help="Path for the JSON report.",
    )
    p.add_argument(
        "--md-out",
        default=None,
        help="Optional path for a markdown rendering. Default: stdout only.",
    )
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-engine progress logging.")
    p.add_argument(
        "--ollama", action="store_true",
        help=("Point other engine OpenAI clients at a local Ollama daemon "
              "(http://localhost:11434/v1). Sets MEMOIRS_USE_OLLAMA=on "
              "and the OPENAI_*/LLM_*/EMBEDDING_* env vars before each "
              "other engine adapter imports its provider library."),
    )
    p.add_argument(
        "--ollama-model", default=DEFAULT_LLM_MODEL,
        help=f"Ollama LLM tag to advertise as OPENAI_MODEL (default: {DEFAULT_LLM_MODEL}).",
    )
    p.add_argument(
        "--ollama-embed-model", default=DEFAULT_EMBED_MODEL,
        help=f"Ollama embedding tag (default: {DEFAULT_EMBED_MODEL}).",
    )
    p.add_argument(
        "--longmemeval", nargs="?", const="",
        default=None,
        help=("Use the LongMemEval dataset instead of the synthetic "
              "fallback. Pass a path or rely on the default lookup at "
              "~/datasets/longmemeval/longmemeval_s.json."),
    )
    p.add_argument(
        "--longmemeval-limit", type=int, default=None,
        help="Truncate LongMemEval cases for quick smoke runs.",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # ---- Optional Ollama wiring ------------------------------------------
    if args.ollama:
        # Set the flag other engines look for, plus the OpenAI shim env. We do
        # this BEFORE constructing any adapter so its `__init__` sees the
        # vars during import.
        os.environ["MEMOIRS_USE_OLLAMA"] = "on"
        apply_ollama_env(model=args.ollama_model,
                         embed_model=args.ollama_embed_model)
        if not ollama_is_up():
            print(ollama_install_hint(model=args.ollama_model,
                                      embed_model=args.ollama_embed_model))
            log.warning("Ollama down: other engines that depend on an LLM will SKIP.")

    # ---- Dataset selection -----------------------------------------------
    dataset: Optional[BenchDataset] = None
    suite: Optional[BenchSuite] = None
    if args.suite == "end-to-end":
        # End-to-end suite: raw conversations. LongMemEval still maps
        # to retrieval-only since its cases are gold-id based, so we
        # ignore --longmemeval here.
        if args.longmemeval is not None:
            log.warning(
                "--longmemeval is ignored when --suite=end-to-end; "
                "using the synthetic conversation suite",
            )
        suite = build_end_to_end_suite()
        n_cases = len(suite.queries)
        log.info("end-to-end suite: %d conversations / %d queries",
                 len(suite.conversations), n_cases)
    elif args.longmemeval is not None:
        # `--longmemeval` (no arg) → autodiscover; with arg → explicit path.
        explicit = args.longmemeval or None
        path = _resolve_longmemeval_path(explicit)
        if path is None:
            print("LongMemEval dataset not found at any of: "
                  + ", ".join(_DEFAULT_LME_PATHS))
            return 2
        try:
            dataset = _build_longmemeval_dataset(
                path, limit=args.longmemeval_limit,
            )
        except FileNotFoundError as e:
            print(f"LongMemEval load failed: {e}")
            return 2
        n_cases = len(dataset.queries)
        log.info("dataset: %d memorias / %d queries",
                 len(dataset.memories), n_cases)
    else:
        dataset = build_dataset()
        n_cases = len(dataset.queries)
        log.info("dataset: %d memorias / %d queries",
                 len(dataset.memories), n_cases)

    engine_names = [e.strip() for e in args.engines.split(",") if e.strip()]
    reports: list[EngineReport] = []
    for name in engine_names:
        log.info("=== %s ===", name)
        try:
            adapter = build_adapter(name)
        except Exception as e:
            log.warning("%s: build_adapter failed: %s", name, e)
            stub = EngineReport(engine=name, n_cases=n_cases,
                                status=f"SKIP (adapter unavailable: {e!r})",
                                mode=("e2e-skip" if args.suite == "end-to-end"
                                      else "retrieval-only"))
            reports.append(stub)
            continue
        try:
            if args.suite == "end-to-end":
                assert suite is not None
                rep = run_engine_end_to_end(adapter, suite, top_k=args.top_k)
            else:
                assert dataset is not None
                rep = run_engine(adapter, dataset, top_k=args.top_k)
            reports.append(rep)
            log.info("%s: status=%s mode=%s MRR=%.2f Hit@5=%.2f "
                     "p50q=%.1fms p50i=%.1fms tokens=%d",
                     name, rep.status, rep.mode, rep.mrr, rep.hit_at_5,
                     rep.latency_p50_ms, rep.ingest_p50_ms, rep.tokens_used)
        finally:
            try:
                adapter.shutdown()
            except Exception as e:  # pragma: no cover
                log.warning("%s shutdown error: %s", name, e)

    payload = serialize_report(reports, top_k=args.top_k, suite=args.suite)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    log.info("wrote JSON report to %s", out_path)

    md = render_markdown(reports, top_k=args.top_k, suite=args.suite)
    if args.md_out:
        Path(args.md_out).write_text(md, encoding="utf-8")
        log.info("wrote markdown report to %s", args.md_out)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
