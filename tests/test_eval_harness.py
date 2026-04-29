"""Tests for `memoirs.evals` (P0-3 eval harness).

Covers:
  1. Metric primitives (precision@k, recall@k, MRR, hit@k) on hand-crafted
     ranked lists with known ground truth.
  2. JSON round-trip of EvalResults preserves every metric + per-query row.
  3. Synthetic suite seeds the DB, run_eval reports the correct shape.
  4. End-to-end performance budget: 3 modes × 10 queries × 30 memorias
     finishes well under 30 seconds.
  5. LongMemEval adapter degrades softly when the dataset is missing.
  6. CLI subcommand prints the per-mode table.
"""
from __future__ import annotations

import io
import json
import time

import pytest

from memoirs.evals.harness import (
    EvalCase,
    EvalResults,
    EvalSuite,
    ModeResults,
    QueryResult,
    compute_metrics,
    hit_at_k,
    mrr,
    precision_at_k,
    recall_at_k,
    run_eval,
)


# ---------------------------------------------------------------------------
# 1. Metric primitives
# ---------------------------------------------------------------------------


def test_metrics_handcrafted_case():
    """The example from the spec: gold=[a,b], retrieved=[a,c,b,d], k=4
    → precision@4 = 0.5, recall@4 = 1.0, MRR = 1.0, hit@1 = 1, hit@5 = 1.
    """
    gold = ["a", "b"]
    retrieved = ["a", "c", "b", "d"]
    metrics = compute_metrics(retrieved, gold, top_k=4)
    assert metrics["precision_at_k"] == pytest.approx(0.5)
    assert metrics["recall_at_k"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["hit_at_1"] == 1.0
    assert metrics["hit_at_5"] == 1.0


def test_metrics_no_hits_returns_zeros():
    """Retrieval with zero overlap with gold should bottom out at 0
    everywhere — never NaN, never throw."""
    metrics = compute_metrics(["x", "y", "z"], ["a"], top_k=3)
    assert metrics["precision_at_k"] == 0.0
    assert metrics["recall_at_k"] == 0.0
    assert metrics["mrr"] == 0.0
    assert metrics["hit_at_1"] == 0.0
    assert metrics["hit_at_5"] == 0.0


def test_mrr_uses_first_gold_position():
    """MRR ignores subsequent gold hits — only the first rank counts."""
    # First gold at rank 3 → MRR = 1/3 even though there's another at 5.
    assert mrr(["x", "y", "a", "z", "b"], ["a", "b"]) == pytest.approx(1 / 3)


def test_recall_caps_at_one_and_precision_floors_above_zero():
    """recall@k cannot exceed 1.0; precision@k stays a fraction of k."""
    gold = ["a", "b", "c"]
    retrieved = ["a", "b", "c", "d", "e"]
    # All gold present at top-3 → recall@5 = 1.0, precision@5 = 3/5.
    assert recall_at_k(retrieved, gold, 5) == pytest.approx(1.0)
    assert precision_at_k(retrieved, gold, 5) == pytest.approx(0.6)


def test_hit_at_k_window_boundaries():
    """hit_at_k only counts gold falling inside the top-k window."""
    gold = ["g"]
    # Gold at position 5 (rank 5) → hit@5 = 1, hit@1 = 0.
    retrieved = ["a", "b", "c", "d", "g"]
    assert hit_at_k(retrieved, gold, 1) == 0.0
    assert hit_at_k(retrieved, gold, 5) == 1.0
    # Gold beyond top-1 / top-5 windows
    assert hit_at_k(["x", "y", "z", "w", "v", "g"], gold, 5) == 0.0


# ---------------------------------------------------------------------------
# 2. EvalResults JSON round-trip
# ---------------------------------------------------------------------------


def _toy_results() -> EvalResults:
    """Build a pre-populated EvalResults so round-trip exercises every field."""
    qr = QueryResult(
        query="favorite drink",
        category="preference",
        gold_memory_ids=["mem_a"],
        retrieved_memory_ids=["mem_a", "mem_b"],
        metrics={"precision_at_k": 0.5, "recall_at_k": 1.0, "mrr": 1.0,
                 "hit_at_1": 1.0, "hit_at_5": 1.0},
        latency_ms=12.34,
        time_to_first_relevant_ms=12.34,
    )
    mode = ModeResults(mode="hybrid", top_k=10, n_cases=1, queries=[qr])
    mode.finalize()
    return EvalResults(
        suite_name="toy",
        top_k=10,
        modes=[mode],
        meta={"who": "tests", "n_cases": 1},
    )


def test_eval_results_json_roundtrip(tmp_path):
    """to_json → from_json must preserve every aggregate AND every per-query row."""
    original = _toy_results()
    out_path = tmp_path / "results.json"
    original.to_json(out_path)
    loaded = EvalResults.from_json(out_path)

    assert loaded.suite_name == original.suite_name
    assert loaded.top_k == original.top_k
    assert loaded.meta == original.meta
    assert len(loaded.modes) == 1

    lo = loaded.modes[0]
    og = original.modes[0]
    assert lo.mode == og.mode
    assert lo.precision_at_k == pytest.approx(og.precision_at_k)
    assert lo.recall_at_k == pytest.approx(og.recall_at_k)
    assert lo.mrr == pytest.approx(og.mrr)
    assert lo.hit_at_1 == pytest.approx(og.hit_at_1)
    assert lo.latency_p50_ms == pytest.approx(og.latency_p50_ms)
    # Per-query rows must round-trip 1:1.
    assert len(lo.queries) == 1
    qr_l, qr_o = lo.queries[0], og.queries[0]
    assert qr_l.query == qr_o.query
    assert qr_l.category == qr_o.category
    assert qr_l.gold_memory_ids == qr_o.gold_memory_ids
    assert qr_l.retrieved_memory_ids == qr_o.retrieved_memory_ids
    assert qr_l.metrics == qr_o.metrics
    assert qr_l.latency_ms == pytest.approx(qr_o.latency_ms)
    assert qr_l.time_to_first_relevant_ms == pytest.approx(qr_o.time_to_first_relevant_ms)


def test_print_table_emits_one_line_per_mode():
    """print_table must write one row per mode, plus header + separator."""
    results = _toy_results()
    # Add a second mode so we can verify multi-row layout.
    extra = ModeResults(mode="bm25", top_k=10, n_cases=1, queries=results.modes[0].queries)
    extra.finalize()
    results.modes.append(extra)

    buf = io.StringIO()
    results.print_table(file=buf)
    text = buf.getvalue()
    assert "toy" in text                      # title carries suite name
    assert "hybrid" in text
    assert "bm25" in text
    # 4 fixed lines (title + header + sep + ≥2 mode rows)
    assert text.count("\n") >= 4


# ---------------------------------------------------------------------------
# 3. Synthetic suite + run_eval
# ---------------------------------------------------------------------------


def test_synthetic_suite_seeds_30_memorias_and_10_cases(tmp_db):
    """Sanity: build() seeds exactly 30 memorias + emits 10 cases, with the
    expected category distribution."""
    from memoirs.evals.suites.synthetic_basic import build
    suite = build(tmp_db)
    assert suite.name == "synthetic_basic"
    assert len(suite.cases) == 10
    n_mem = tmp_db.conn.execute(
        "SELECT COUNT(*) AS c FROM memories WHERE archived_at IS NULL"
    ).fetchone()["c"]
    assert n_mem == 30
    # Multi-hop pair must have memory_links wired (P1-3 contract).
    n_links = tmp_db.conn.execute(
        "SELECT COUNT(*) AS c FROM memory_links "
        "WHERE source_memory_id LIKE 'mem_mh_%'"
    ).fetchone()["c"]
    assert n_links >= 4  # 2 pairs × 2 directions

    cats = sorted(c.category for c in suite.cases)
    assert cats.count("single-hop") == 4
    assert cats.count("multi-hop") == 3
    assert cats.count("temporal") == 2
    assert cats.count("preference") == 1


def test_run_eval_reports_correct_shape(tmp_db):
    """run_eval returns one ModeResults per requested mode, every metric set."""
    from memoirs.evals.suites.synthetic_basic import build
    suite = build(tmp_db)
    results = run_eval(
        tmp_db, suite, top_k=10,
        retrieval_modes=("hybrid", "bm25"),
    )
    assert results.suite_name == "synthetic_basic"
    assert results.top_k == 10
    assert len(results.modes) == 2
    for m in results.modes:
        assert m.n_cases == 10
        assert len(m.queries) == 10
        # Every metric is in [0, 1] when populated.
        assert 0.0 <= m.precision_at_k <= 1.0
        assert 0.0 <= m.recall_at_k <= 1.0
        assert 0.0 <= m.mrr <= 1.0
        assert m.latency_p50_ms >= 0.0
        assert m.latency_p95_ms >= m.latency_p50_ms
        # By-category breakdown covers all 4 buckets.
        cats = set(m.by_category.keys())
        assert {"single-hop", "multi-hop", "temporal", "preference"} <= cats


def test_bm25_finds_single_hop_keyword_matches(tmp_db):
    """BM25 must hit at least 1 single-hop case — keywords are unique enough.

    This is a weak smoke test (we don't pin specific recall numbers since
    tokenizer behavior may shift) but it catches the 'BM25 returns nothing'
    regression class.
    """
    from memoirs.evals.suites.synthetic_basic import build
    suite = build(tmp_db)
    results = run_eval(
        tmp_db, suite, top_k=10, retrieval_modes=("bm25",),
    )
    bm25 = results.modes[0]
    sh = bm25.by_category.get("single-hop", {})
    assert sh.get("hit_at_1", 0) > 0 or sh.get("recall_at_k", 0) > 0


# ---------------------------------------------------------------------------
# 4. Performance budget
# ---------------------------------------------------------------------------


def test_run_eval_three_modes_under_30s(tmp_db):
    """Acceptance: 3 modes × 10 queries × 30 memorias < 30 seconds.

    Embeddings may load a sentence-transformers model lazily on first call;
    we still expect the total wall-clock to fit comfortably under the
    30-second budget called out in the spec, even on a slow CI machine.
    """
    from memoirs.evals.suites.synthetic_basic import build
    suite = build(tmp_db)
    t0 = time.perf_counter()
    results = run_eval(
        tmp_db, suite, top_k=10, retrieval_modes=("hybrid", "dense", "bm25"),
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, f"run_eval took {elapsed:.1f}s (budget: 30s)"
    assert len(results.modes) == 3


# ---------------------------------------------------------------------------
# 5. LongMemEval adapter — soft skip on missing dataset
# ---------------------------------------------------------------------------


def test_longmemeval_adapter_returns_skip_when_missing(tmp_path):
    """If the JSONL doesn't exist, the adapter must return (None, info)
    with a useful 'reason' — never raise."""
    from memoirs.evals.longmemeval_adapter import load_longmemeval, is_available

    missing = tmp_path / "definitely-not-here.jsonl"
    assert is_available(missing) is False
    suite, info = load_longmemeval(missing)
    assert suite is None
    assert "reason" in info
    assert "github.com/xiaowu0162/LongMemEval" in info.get("hint", "")


def test_longmemeval_adapter_parses_minimal_jsonl(tmp_path):
    """A minimal 2-record JSONL should parse into 2 EvalCases with proper
    category mapping."""
    from memoirs.evals.longmemeval_adapter import load_longmemeval

    p = tmp_path / "tiny.jsonl"
    p.write_text(
        json.dumps({
            "question_id": "Q1",
            "question": "When did Alice mention her birthday?",
            "question_type": "single-session-user",
            "answer_session_ids": ["s_alice_42"],
        }) + "\n" +
        json.dumps({
            "question_id": "Q2",
            "question": "Has the project deadline changed?",
            "question_type": "knowledge-update",
            "evidence_ids": ["m_old", "m_new"],
            "as_of": "2024-12-01T00:00:00Z",
        }) + "\n",
        encoding="utf-8",
    )
    suite, info = load_longmemeval(p)
    assert suite is not None
    assert len(suite.cases) == 2
    assert info["n_cases"] == 2
    cats = {c.category for c in suite.cases}
    assert cats == {"single-hop", "temporal"}
    # The temporal case carries `as_of`.
    temporal = [c for c in suite.cases if c.category == "temporal"][0]
    assert temporal.as_of and "2024-12-01" in temporal.as_of


# ---------------------------------------------------------------------------
# 6. CLI smoke test
# ---------------------------------------------------------------------------


def test_cli_eval_synthetic_runs_and_prints_table(capsys):
    """`memoirs eval --suite synthetic_basic` returns 0 and prints
    a table with the three default mode rows."""
    from memoirs.cli import main

    rc = main(["eval", "--suite", "synthetic_basic",
               "--modes", "hybrid,bm25,dense", "--top-k", "10"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Eval results: synthetic_basic" in out
    for mode in ("hybrid", "bm25", "dense"):
        assert mode in out


def test_cli_eval_save_writes_json_roundtrippable(tmp_path, capsys):
    """`--save` must produce a JSON file that EvalResults.from_json reads back."""
    from memoirs.cli import main

    out_json = tmp_path / "results.json"
    rc = main(["eval", "--suite", "synthetic_basic",
               "--modes", "bm25", "--top-k", "5",
               "--save", str(out_json)])
    assert rc == 0
    assert out_json.exists()
    loaded = EvalResults.from_json(out_json)
    assert loaded.suite_name == "synthetic_basic"
    assert loaded.top_k == 5
    assert len(loaded.modes) == 1
    assert loaded.modes[0].mode == "bm25"
