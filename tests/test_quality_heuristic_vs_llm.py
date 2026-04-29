"""Tests for `scripts/quality_heuristic_vs_llm.py`.

Covers: heuristic baselines, LLM mock plumbing, metric computation, and the
CLI entrypoint on a reduced dataset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import quality_heuristic_vs_llm as qhl  # noqa: E402


# ----------------------------------------------------------------------
# Heuristic side — sanity on a couple of canonical cases
# ----------------------------------------------------------------------


def test_heuristic_consolidate_exact_dup_returns_update():
    res = qhl._heuristic_consolidate(
        "user prefers dark mode in editors", "preference",
        [("user prefers dark mode in editors", "preference", 1.0)],
    )
    assert res == "UPDATE"


def test_heuristic_consolidate_semantic_same_type_returns_merge():
    res = qhl._heuristic_consolidate(
        "user really prefers Python for scripting", "preference",
        [("user prefers Python for scripts", "preference", 0.91)],
    )
    assert res == "MERGE"


def test_heuristic_consolidate_semantic_diff_type_returns_contradiction():
    res = qhl._heuristic_consolidate(
        "user prefers Python over Ruby", "preference",
        [("user uses Python at work daily", "fact", 0.92)],
    )
    assert res == "CONTRADICTION"


def test_heuristic_consolidate_unrelated_returns_add():
    res = qhl._heuristic_consolidate(
        "user owns a vintage typewriter from 1960s", "fact",
        [("user prefers Python for scripting", "preference", 0.4)],
    )
    assert res == "ADD"


def test_heuristic_consolidate_empty_returns_ignore():
    assert qhl._heuristic_consolidate("", "fact", []) == "IGNORE"
    assert qhl._heuristic_consolidate("   ", "fact", []) == "IGNORE"


# ----------------------------------------------------------------------
# Metric calculation
# ----------------------------------------------------------------------


def test_binary_stats_records_correctly():
    s = qhl.BinaryStats()
    qhl._record(s, True, True)   # tp
    qhl._record(s, True, True)   # tp
    qhl._record(s, False, False)  # tn
    qhl._record(s, True, False)  # fp
    qhl._record(s, False, True)  # fn
    assert s.tp == 2 and s.tn == 1 and s.fp == 1 and s.fn == 1
    assert s.total == 5
    assert s.accuracy == pytest.approx(3 / 5)
    assert s.fpr == pytest.approx(1 / 2)  # fp / (fp + tn) = 1/2
    assert s.fnr == pytest.approx(1 / 3)  # fn / (fn + tp) = 1/3


def test_ent_metrics_perfect_match():
    m = qhl._ent_metrics({"a", "b"}, {"a", "b"})
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


def test_ent_metrics_partial():
    m = qhl._ent_metrics({"a", "b", "c"}, {"a", "b", "d"})
    # tp=2, fp=1, fn=1
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 1
    assert m["precision"] == pytest.approx(2 / 3, rel=1e-3)
    assert m["recall"] == pytest.approx(2 / 3, rel=1e-3)


def test_ent_metrics_both_empty_is_perfect():
    m = qhl._ent_metrics(set(), set())
    assert m["f1"] == 1.0


# ----------------------------------------------------------------------
# LLM mock plumbing
# ----------------------------------------------------------------------


def test_llm_judge_expire_yes_path():
    calls: list[str] = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        return "YES — clearly contradictory"

    pred, raw = qhl._llm_judge_expire(
        "user lives in Paris", "user lives in Buenos Aires", llm_call=llm,
    )
    assert pred is True
    assert calls and "lives in Paris" in calls[0]


def test_llm_judge_expire_no_path():
    pred, _ = qhl._llm_judge_expire(
        "x", "y", llm_call=lambda p: "NO\nreason: unrelated",
    )
    assert pred is False


def test_llm_judge_expire_garbage_defaults_to_no():
    pred, _ = qhl._llm_judge_expire(
        "x", "y", llm_call=lambda p: "I don't know lol",
    )
    assert pred is False


def test_llm_consolidate_judge_picks_first_token():
    pred = qhl._llm_consolidate_judge(
        "user prefers Python over Ruby", "preference",
        [("user uses Python at work daily", "fact", 0.92)],
        llm_call=lambda p: "MERGE — same intent",
    )
    assert pred == "MERGE"


def test_llm_consolidate_judge_empty_short_circuits():
    pred = qhl._llm_consolidate_judge(
        "  ", "fact", [], llm_call=lambda p: "ADD",
    )
    assert pred == "IGNORE"


def test_llm_judge_entities_parses_csv():
    out = qhl._llm_judge_entities(
        "Memoirs uses SQLite",
        llm_call=lambda p: "memoirs, sqlite, llama.cpp",
    )
    assert out == {"memoirs", "sqlite", "llama.cpp"}


# ----------------------------------------------------------------------
# Full-loop integration on a tiny dataset (no real LLM)
# ----------------------------------------------------------------------


def test_run_layer_a_with_mock_llm():
    cases_expire = qhl.EXPIRE_CASES[:3]
    cases_archive = qhl.ARCHIVE_CASES[:3]
    # Mock LLM that always says NO — we just want to exercise the plumbing.
    out = qhl.run_layer_a(
        cases_expire, cases_archive, llm_call=lambda p: "NO",
    )
    assert out["expire"]["n"] == 3
    assert out["archive"]["n"] == 3
    assert out["expire"]["llm"] is not None
    assert "accuracy" in out["expire"]["heuristic"]
    assert out["expire"]["agreement_rate"] is not None


def test_run_layer_b_with_mock_llm():
    cases = qhl.CONSOLIDATE_CASES[:3]
    out = qhl.run_layer_b(cases, llm_call=lambda p: "ADD")
    assert out["n"] == 3
    assert out["llm"] is not None
    assert "accuracy" in out["heuristic"]


def test_run_layer_c_with_mock_llm():
    cases = qhl.ENTITY_CASES[:3]
    out = qhl.run_layer_c(cases, llm_call=lambda p: "memoirs, sqlite")
    assert out["n"] == 3
    assert out["llm"] is not None
    assert "f1_mean" in out["heuristic"]


def test_run_benchmark_no_llm_path():
    report = qhl.run_benchmark(
        layers={"A", "B", "C"}, n_per_layer=2, llm_call=None,
    )
    assert report["llm_used"] is False
    assert "layer_a" in report and "layer_b" in report and "layer_c" in report
    rec = report["recommendation"]
    assert "heuristic only" in rec  # all layers should report heuristic-only


def test_cli_runs_on_smoke_dataset(tmp_path, monkeypatch):
    out = tmp_path / "report.json"
    rc = qhl.main([
        "--smoke", "--no-llm", "--quiet",
        "--out", str(out),
    ])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["llm_used"] is False
    assert data["n_per_layer"] == 10
    assert "layer_a" in data and "layer_b" in data and "layer_c" in data


def test_cli_single_layer(tmp_path):
    out = tmp_path / "r.json"
    rc = qhl.main([
        "--smoke", "--no-llm", "--quiet",
        "--layer", "C", "--out", str(out),
    ])
    assert rc == 0
    data = json.loads(out.read_text())
    assert "layer_c" in data
    assert "layer_a" not in data and "layer_b" not in data


def test_recommendation_handles_no_llm():
    rep = {
        "layer_a": {"sections": {
            "expire": {"heuristic": {"accuracy": 0.8, "latency": {"p95_ms": 0.1}},
                       "llm": None}}},
    }
    rec = qhl._recommendation(rep)
    assert "heuristic only" in rec
