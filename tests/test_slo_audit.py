"""Tests for ``scripts/slo_audit.py`` — Phase 5D.

These verify the audit harness end-to-end without depending on long
benchmarks. We exercise:

  * ``_run_latency`` against a fresh fixture DB → returns the expected
    report shape.
  * ``_evaluate_latency`` pass/fail logic with synthesized samples.
  * ``_run_sustained`` with a tight budget (1s, 2 workers, low target_rps).
  * ``_run_cold_start`` returns a sensible duration on a small fixture.
  * ``_run_memory`` returns non-null numbers and a ``pass`` flag.

The script itself lives in ``scripts/``; we import it via importlib so we
don't need to make the directory a package.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from memoirs.db import MemoirsDB


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "slo_audit.py"


@pytest.fixture(scope="module")
def slo_audit():
    """Load ``scripts/slo_audit.py`` as a module without packaging it."""
    spec = importlib.util.spec_from_file_location("slo_audit", SCRIPT_PATH)
    assert spec and spec.loader, "could not load slo_audit module"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["slo_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fixture_db(tmp_path: Path) -> Path:
    """A tiny initialised DB. ``assemble_context`` works on an empty DB —
    it just returns no memories — so we don't need to seed anything."""
    db = MemoirsDB(tmp_path / "memoirs.sqlite")
    db.init()
    db.close()
    return tmp_path / "memoirs.sqlite"


# ---------------------------------------------------------------------------
# _evaluate_latency: pass/fail logic
# ---------------------------------------------------------------------------


def test_evaluate_latency_all_pass(slo_audit):
    """When measured percentiles fit under the targets the result must pass."""
    samples = [1.0] * 30  # well under any SLO
    res = slo_audit._evaluate_latency("mcp_get_context", samples)
    assert res["pass"] is True
    assert res["p50_pass"] is True
    assert res["p95_pass"] is True
    assert res["p99_pass"] is True
    assert res["samples"] == 30


def test_evaluate_latency_p50_breach(slo_audit):
    """If p50 exceeds the target the row is marked failed."""
    # mcp_get_context p50 SLO = 50ms — feed it 200ms median.
    samples = [200.0] * 30
    res = slo_audit._evaluate_latency("mcp_get_context", samples)
    assert res["pass"] is False
    assert res["p50_pass"] is False


def test_evaluate_latency_search_targets(slo_audit):
    """``mcp_search_memory`` only has p50 + p95, no p99 — make sure absent
    targets don't sneak into the pass aggregate as ``False``."""
    res = slo_audit._evaluate_latency("mcp_search_memory", [5.0] * 20)
    assert res["pass"] is True
    assert "p99_pass" not in res or res["p99_pass"] is True


# ---------------------------------------------------------------------------
# _run_latency: shape + pass aggregation
# ---------------------------------------------------------------------------


def test_run_latency_returns_report_shape(slo_audit, fixture_db: Path, monkeypatch):
    """Smoke: the latency runner produces every expected SLO key."""
    # Force BM25 mode — fixture DB has no embeddings, so dense retrieval
    # would explode. BM25 is purely lexical.
    monkeypatch.setenv("MEMOIRS_RETRIEVAL_MODE", "bm25")
    res = slo_audit._run_latency(fixture_db, iters=3)
    assert {"mcp_get_context", "mcp_search_memory",
            "assemble_context_stream_ttft", "mcp_extract_pending"} <= set(res)

    ctx = res["mcp_get_context"]
    assert "p50_actual_ms" in ctx
    assert "p95_actual_ms" in ctx
    assert "pass" in ctx

    # mcp_search_memory should be skipped (no vec0 table on the fixture DB)
    # OR actually run if extras are installed — both are valid outcomes.
    sm = res["mcp_search_memory"]
    assert "pass" in sm

    ttft = res["assemble_context_stream_ttft"]
    assert ttft["target_ms"] == 50.0
    assert ttft["samples"] == 3


# ---------------------------------------------------------------------------
# _run_sustained: small budget completes
# ---------------------------------------------------------------------------


def test_run_sustained_short_budget(slo_audit, fixture_db: Path, monkeypatch):
    """A 1-second 2-worker pass with a low target_rps should report stats."""
    monkeypatch.setenv("MEMOIRS_RETRIEVAL_MODE", "bm25")
    res = slo_audit._run_sustained(
        fixture_db, seconds=1, workers=2, target_rps=1.0, mode="bm25",
    )
    assert res["seconds"] == 1
    assert res["workers"] == 2
    assert res["completed"] >= 1
    assert "actual_rps" in res
    assert "p99_ms" in res
    assert res["target_rps"] == 1.0
    # On an empty DB BM25 trivially completes — no errors expected.
    assert res["errors"] == 0


# ---------------------------------------------------------------------------
# _run_cold_start: child process boots and prints
# ---------------------------------------------------------------------------


def test_run_cold_start_returns_reasonable(slo_audit, fixture_db: Path):
    """Single child run; no SLO comparison — just shape + sanity."""
    res = slo_audit._run_cold_start(fixture_db, runs=1)
    assert res["runs"] == 1
    assert isinstance(res["actual_p50_s"], float)
    # Allow up to 30s in CI — the child loads memoirs from scratch.
    assert 0.0 < res["actual_p50_s"] < 30.0
    assert "READY" in res["last_stdout"], (
        f"expected READY marker, got: {res['last_stdout']!r}"
    )


# ---------------------------------------------------------------------------
# _run_memory: non-null snapshots
# ---------------------------------------------------------------------------


def test_run_memory_reports_numbers(slo_audit, fixture_db: Path, monkeypatch):
    """Ensure all three milestones produce positive numbers.

    Uses the in-process variant so the test doesn't fork a child interpreter
    (the production ``_run_memory`` path is exercised manually via the CLI).
    """
    monkeypatch.setenv("MEMOIRS_RETRIEVAL_MODE", "bm25")
    res = slo_audit._run_memory_inproc(fixture_db, queries=5)
    assert res["queries"] == 5
    assert res["idle_mb"] > 0
    assert res["active_mb"] > 0
    assert res["peak_mb"] >= res["active_mb"]
    assert "pass" in res
    assert "tracemalloc_idle_mb" in res


# ---------------------------------------------------------------------------
# _flatten_slos + _print_table: tabular reporting
# ---------------------------------------------------------------------------


def test_flatten_slos_tabular_rows(slo_audit):
    """Compose a synthetic report and verify the tabular flattening."""
    report = {
        "slos": {
            "mcp_get_context": {
                "p50_target_ms": 50.0, "p50_actual_ms": 5.0, "p50_pass": True,
                "p95_target_ms": 200.0, "p95_actual_ms": 80.0, "p95_pass": True,
                "p99_target_ms": 1000.0, "p99_actual_ms": 150.0, "p99_pass": True,
                "pass": True,
            },
            "sustained": {
                "target_rps": 50.0, "actual_rps": 134.0, "pass": True,
            },
        }
    }
    rows = slo_audit._flatten_slos(report)
    names = [r[0] for r in rows]
    assert "mcp_get_context p50" in names
    assert "mcp_get_context p95" in names
    assert "mcp_get_context p99" in names
    assert "sustained throughput" in names
    assert all(r[3] for r in rows), rows


def test_summarise_counts(slo_audit):
    report = {
        "slos": {
            "mcp_get_context": {
                "p50_target_ms": 50.0, "p50_actual_ms": 200.0, "p50_pass": False,
                "p95_target_ms": 200.0, "p95_actual_ms": 80.0, "p95_pass": True,
                "p99_target_ms": 1000.0, "p99_actual_ms": 150.0, "p99_pass": True,
                "pass": False,
            },
        }
    }
    s = slo_audit._summarise(report)
    assert s["total"] == 3
    assert s["passed"] == 2
    assert s["failed"] == 1
