"""Tests for the head-to-head other engine bench (`scripts/bench_vs_others.py`).

What this suite proves:
  1. The synthetic dataset has the contracted shape (80 memorias / 20
     queries, four categories, no dangling gold IDs).
  2. The MemoirsAdapter can ingest + serve queries end-to-end against a
     temporary DB without touching production state.
  3. Other engine adapters that need a remote service (Mem0, Cognee, Zep,
     Letta) skip cleanly when the service / package is missing —
     ``status.ok == False`` with a non-empty reason.
  4. Per-engine aggregation produces the metrics the markdown table
     needs, with hand-crafted golden inputs.
  5. The CLI runs end-to-end with `--engines memoirs` and writes a
     valid JSON artifact.

Tests deliberately avoid any network or Docker calls so they pass in
sandboxed CI. The adapters are designed to *self-skip* when their
backend is unavailable, which is exactly what the bench reports.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Resolve the project root so `scripts.*` imports work from pytest's
# isolated module path.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# 1. Dataset shape
# ---------------------------------------------------------------------------


def test_dataset_has_80_memorias_and_20_queries():
    """The contract called out in the script docstring: 80 / 20 / 4 cats."""
    from scripts.bench_vs_others_dataset import build_dataset

    ds = build_dataset()
    assert len(ds.memories) == 80
    assert len(ds.queries) == 20

    cats = sorted(q.category for q in ds.queries)
    assert cats.count("single-hop") == 8
    assert cats.count("multi-hop") == 6
    assert cats.count("temporal") == 4
    assert cats.count("preference") == 2

    # Every memory ID is unique.
    ids = [m.id for m in ds.memories]
    assert len(set(ids)) == len(ids)

    # Every gold ID points at a real memory.
    known = set(ids)
    for q in ds.queries:
        assert q.gold_memory_ids, f"query {q.query!r} has empty gold list"
        for gid in q.gold_memory_ids:
            assert gid in known, f"gold id {gid!r} missing from corpus"


def test_temporal_queries_carry_as_of_or_signal_latest_wins():
    """Temporal cases must either set `as_of` (old should win) or omit
    it (latest should win) — both shapes are needed to prove the
    bi-temporal behavior."""
    from scripts.bench_vs_others_dataset import build_dataset

    ds = build_dataset()
    temporals = [q for q in ds.queries if q.category == "temporal"]
    assert len(temporals) == 4
    with_asof = [q for q in temporals if q.as_of]
    without_asof = [q for q in temporals if not q.as_of]
    assert len(with_asof) == 2
    assert len(without_asof) == 2


# ---------------------------------------------------------------------------
# 2. MemoirsAdapter end-to-end
# ---------------------------------------------------------------------------


def test_memoirs_adapter_runs_end_to_end_on_tempdir(tmp_path):
    """Adapter must ingest the corpus + serve queries against a fresh DB,
    and the easy single-hop cases should retrieve at least *some* gold.

    We don't pin a specific recall number (embedding model availability
    varies in CI) but BM25 alone should hit on the unique-keyword
    single-hop targets like "passport" / "dentist".
    """
    from scripts.adapters.memoirs_adapter import MemoirsAdapter
    from scripts.bench_vs_others_dataset import build_dataset

    adapter = MemoirsAdapter(db_dir=tmp_path / "memoirs_db")
    try:
        assert adapter.status.ok, adapter.status.reason
        ds = build_dataset()
        adapter.add_memories(ds.memories)
        # Pick a single-hop case with a very distinctive keyword.
        q = next(q for q in ds.queries if "passport" in q.query)
        ranked = adapter.query(q, top_k=10)
        assert isinstance(ranked, list)
        assert len(ranked) <= 10
        # The gold memory MUST surface for a unique-keyword single-hop.
        assert "mem_sh_passport" in ranked
    finally:
        adapter.shutdown()


# ---------------------------------------------------------------------------
# 3. Other engine adapters skip cleanly
# ---------------------------------------------------------------------------


def test_mem0_adapter_skips_clean_without_openai_key(monkeypatch):
    """Mem0 needs OPENAI_API_KEY (per the external container image).
    Without it the adapter must short-circuit to status.ok=False."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from scripts.adapters.mem0_adapter import Mem0Adapter

    adapter = Mem0Adapter()  # no base_url → tries Docker path
    try:
        assert adapter.status.ok is False
        assert adapter.status.reason
    finally:
        adapter.shutdown()


def test_cognee_adapter_skips_clean_when_package_missing():
    """If `cognee` is not importable the adapter must skip — the bench
    treats this as "engine unavailable", not a crash."""
    from scripts.adapters.cognee_adapter import CogneeAdapter

    adapter = CogneeAdapter()
    try:
        # Either cognee is installed (status OK) or it skips with reason.
        if not adapter.status.ok:
            assert "cognee" in adapter.status.reason.lower()
    finally:
        adapter.shutdown()


def test_zep_adapter_skips_clean_without_endpoint(monkeypatch):
    """Zep self-host is heavyweight; absent ZEP_BASE_URL must skip."""
    monkeypatch.delenv("ZEP_BASE_URL", raising=False)
    from scripts.adapters.zep_adapter import ZepAdapter

    adapter = ZepAdapter()
    assert adapter.status.ok is False
    assert "zep" in adapter.status.reason.lower() or "url" in adapter.status.reason.lower()


def test_letta_adapter_skips_clean_without_endpoint(monkeypatch):
    """Same contract as Zep — Letta is an agent runtime, skip without url."""
    monkeypatch.delenv("LETTA_BASE_URL", raising=False)
    from scripts.adapters.letta_adapter import LettaAdapter

    adapter = LettaAdapter()
    assert adapter.status.ok is False
    assert "letta" in adapter.status.reason.lower() or "url" in adapter.status.reason.lower()


# ---------------------------------------------------------------------------
# 4. Per-engine aggregation with hand-crafted golden inputs
# ---------------------------------------------------------------------------


def test_aggregation_matches_handcrafted_metrics():
    """Feed `_aggregate` two queries with known retrievals and verify
    every aggregate field matches the math by hand:

      Q1: gold=[a],   retrieved=[a, x, y]      → P@10=0.1, R@10=1.0, MRR=1.0, hit@1=1
      Q2: gold=[a,b], retrieved=[c, a, d, b]   → P@10=0.2, R@10=1.0, MRR=0.5, hit@1=0
    """
    from scripts.bench_vs_others import (
        EngineQueryResult, EngineReport, _aggregate,
    )
    from memoirs.evals.harness import compute_metrics

    rep = EngineReport(engine="x", n_cases=2)
    for q_idx, (gold, retrieved, cat) in enumerate([
        (["a"], ["a", "x", "y"], "single-hop"),
        (["a", "b"], ["c", "a", "d", "b"], "multi-hop"),
    ]):
        m = compute_metrics(retrieved, gold, top_k=10)
        rep.queries.append(EngineQueryResult(
            query=f"q{q_idx}", category=cat, gold_memory_ids=list(gold),
            retrieved_memory_ids=list(retrieved), metrics=m, latency_ms=10.0 * (q_idx + 1),
        ))
    _aggregate(rep)

    # P@10: (0.1 + 0.2) / 2 = 0.15
    assert rep.precision_at_k == pytest.approx(0.15)
    # R@10: (1.0 + 1.0) / 2
    assert rep.recall_at_k == pytest.approx(1.0)
    # MRR: (1.0 + 0.5) / 2
    assert rep.mrr == pytest.approx(0.75)
    # Hit@1: (1 + 0) / 2
    assert rep.hit_at_1 == pytest.approx(0.5)
    # Hit@5: both queries land gold within top-5
    assert rep.hit_at_5 == pytest.approx(1.0)
    # Latency p50: linear interp between 10 and 20 = 15
    assert rep.latency_p50_ms == pytest.approx(15.0)
    # By-category present for both buckets
    assert set(rep.by_category) == {"single-hop", "multi-hop"}


# ---------------------------------------------------------------------------
# 5. CLI smoke test
# ---------------------------------------------------------------------------


def test_cli_runs_with_memoirs_only_and_writes_json(tmp_path, capsys):
    """`bench_vs_others.main(['--engines', 'memoirs', ...])` must exit
    0, write a parseable JSON artifact, and print a markdown table that
    mentions the engine name."""
    from scripts.bench_vs_others import main

    out_json = tmp_path / "report.json"
    rc = main([
        "--engines", "memoirs",
        "--top-k", "10",
        "--out", str(out_json),
        "--quiet",
    ])
    assert rc == 0
    assert out_json.exists()
    data = json.loads(out_json.read_text())
    assert data["top_k"] == 10
    assert data["n_engines"] == 1
    assert data["engines"][0]["engine"] == "memoirs"
    # Status should be OK on a clean import path.
    assert data["engines"][0]["status"] == "OK"
    # Markdown stays predictable for downstream regex consumers.
    captured = capsys.readouterr().out
    assert "memoirs" in captured
    assert "P@k" in captured


def test_cli_skips_unknown_engines_via_skip_row(tmp_path, capsys, monkeypatch):
    """Engines that fail at adapter-construction or at startup should
    appear as SKIP rows, not crash the run."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ZEP_BASE_URL", raising=False)
    from scripts.bench_vs_others import main

    out_json = tmp_path / "report.json"
    rc = main([
        "--engines", "memoirs,zep",  # zep will skip
        "--top-k", "5",
        "--out", str(out_json),
        "--quiet",
    ])
    assert rc == 0
    data = json.loads(out_json.read_text())
    statuses = {e["engine"]: e["status"] for e in data["engines"]}
    assert statuses["memoirs"] == "OK"
    assert statuses["zep"].startswith("SKIP")
