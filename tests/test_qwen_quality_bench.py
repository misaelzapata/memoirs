"""Tests for `scripts/bench_qwen_quality.py`.

These tests exercise the bench harness without loading a real LLM:
- importability + CLI dry-run
- helper utilities (heuristics, percentile, contradiction pair builder)
- end-to-end run with a mock LLM against a tiny synthetic DB
- DB-snapshot safety (live DB never mutates)
- JSON report shape

The Qwen GGUF is not required to pass these tests.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Locate scripts/bench_qwen_quality.py and import it as a module.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BENCH_PATH = _REPO_ROOT / "scripts" / "bench_qwen_quality.py"


def _load_bench_module():
    spec = importlib.util.spec_from_file_location("bench_qwen_quality", _BENCH_PATH)
    assert spec and spec.loader, "could not load bench_qwen_quality module spec"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_qwen_quality"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bench():
    return _load_bench_module()


# ----------------------------------------------------------------------
# Synthetic DB fixture
# ----------------------------------------------------------------------


@pytest.fixture
def synthetic_db(tmp_path: Path) -> Path:
    """Build a minimal Memoirs-shaped SQLite DB without invoking migrations.

    Only the tables/columns the bench actually queries are needed. Keeping
    this self-contained avoids tying the test to migration drift.
    """
    db_path = tmp_path / "synth.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY,
            message_count INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT '2026-01-01',
            created_at TEXT
        );
        CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE memory_candidates (
            id TEXT PRIMARY KEY,
            conversation_id TEXT,
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            importance INTEGER NOT NULL DEFAULT 3,
            confidence REAL NOT NULL DEFAULT 0.5,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL DEFAULT '2026-01-01'
        );
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            importance INTEGER NOT NULL DEFAULT 3,
            confidence REAL NOT NULL DEFAULT 0.5,
            score REAL NOT NULL DEFAULT 0.0,
            archived_at TEXT
        );
        """
    )
    # Seed: 3 conversations, 2 long enough for the bench.
    conn.execute("INSERT INTO conversations VALUES (?, ?, ?, ?)",
                 ("conv_long_a", 25, "2026-01-15", "2026-01-01"))
    conn.execute("INSERT INTO conversations VALUES (?, ?, ?, ?)",
                 ("conv_long_b", 22, "2026-01-16", "2026-01-02"))
    conn.execute("INSERT INTO conversations VALUES (?, ?, ?, ?)",
                 ("conv_short_c", 4, "2026-01-17", "2026-01-03"))

    msgs = []
    proper_nouns = ["Memoirs", "SQLite", "Qwen", "Python"]
    for cid, count in (("conv_long_a", 25), ("conv_long_b", 22), ("conv_short_c", 4)):
        for i in range(count):
            role = "user" if i % 2 == 0 else "assistant"
            content = (f"Message {i} for {cid}: discussing {proper_nouns[i % len(proper_nouns)]} "
                       f"and refactor of the engine; this is meaningful content #{i}.")
            msgs.append((f"{cid}_msg_{i}", cid, role, content, i, 1))
    conn.executemany(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)", msgs
    )

    # Memories.
    conn.executemany(
        "INSERT INTO memories(id, type, content, importance, confidence, score) VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("mem_1", "project", "working on memoirs (local-first memory engine)", 5, 0.95, 0.8),
            ("mem_2", "preference", "prefers Python over Go for prototyping", 4, 0.9, 0.6),
            ("mem_3", "decision", "uses sqlite-vec for embeddings", 4, 0.85, 0.7),
            ("mem_4", "fact", "Qwen 2.5 3B is the curator backend", 4, 0.9, 0.65),
        ],
    )
    conn.commit()
    conn.close()
    return db_path


# ----------------------------------------------------------------------
# 1. Importability + CLI dry-run
# ----------------------------------------------------------------------


def test_module_importable(bench):
    """The bench module loads without errors."""
    assert hasattr(bench, "run_bench")
    assert hasattr(bench, "build_arg_parser")
    assert hasattr(bench, "MockLLM")
    assert callable(bench.main)


def test_cli_dry_run_no_llm(bench, synthetic_db, tmp_path, capsys):
    """`--dry-run` must not call any LLM and must produce a stub report."""
    out_path = tmp_path / "report.json"
    rc = bench.main([
        "--db", str(synthetic_db),
        "--dry-run",
        "--out", str(out_path),
    ])
    assert rc == 0
    assert out_path.exists()

    report = json.loads(out_path.read_text())
    assert report["mode"] == "dry-run"
    for stage in ("extract", "consolidate", "contradictions", "summaries"):
        assert report[stage]["n"] == 0
        assert report[stage]["skipped"] == "dry-run"

    # And the captured stdout must include the table.
    captured = capsys.readouterr().out
    assert "Qwen Quality Bench" in captured


# ----------------------------------------------------------------------
# 2. Helper utilities
# ----------------------------------------------------------------------


def test_useful_content_heuristic(bench):
    # Useful: has proper nouns, reasonable length, no fluff.
    assert bench._is_useful_content("user prefers Python over Go for prototyping")
    # Too short.
    assert not bench._is_useful_content("yes")
    # Fluffy.
    assert not bench._is_useful_content(
        "the user is talking about something important here in summary")
    # No proper noun.
    assert not bench._is_useful_content(
        "the system stores things and uses default values when needed for items")
    # Too long: 400+ chars.
    long_text = "Memoirs is great. " + "blah " * 100
    assert not bench._is_useful_content(long_text)


def test_percentile_edge_cases(bench):
    assert bench._percentile([], 50) == 0.0
    assert bench._percentile([42.0], 95) == 42.0
    assert bench._percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0


def test_build_contradiction_pairs(bench):
    pairs = bench.build_contradiction_pairs(5, 3)
    assert len(pairs) == 8
    contra = [p for p in pairs if p["expected_contradictory"]]
    non = [p for p in pairs if not p["expected_contradictory"]]
    assert len(contra) == 5
    assert len(non) == 3
    # All pairs must have non-empty a/b.
    for p in pairs:
        assert isinstance(p["a"], str) and len(p["a"]) > 5
        assert isinstance(p["b"], str) and len(p["b"]) > 5


def test_entity_overlap(bench):
    corpus = "We talked about Memoirs and Python. Qwen replaces Gemma in the curator."
    summary = "Memoirs uses Qwen as the curator. The Python project is active."
    n = bench._entity_overlap(summary, corpus)
    # Memoirs, Python, Qwen all appear in both.
    assert n >= 3


def test_fluff_detector(bench):
    assert bench._looks_fluffy("In summary, the user is working on stuff.")
    assert bench._looks_fluffy("This conversation discusses several topics.")
    assert not bench._looks_fluffy(
        "memoirs migrated to Qwen 2.5 3B as the curator backend"
    )


# ----------------------------------------------------------------------
# 3. DB snapshot safety
# ----------------------------------------------------------------------


def test_copy_db_readonly_does_not_mutate_source(bench, synthetic_db, tmp_path):
    src_size = synthetic_db.stat().st_size
    src_mtime = synthetic_db.stat().st_mtime

    dst = tmp_path / "snap.sqlite"
    out = bench.copy_db_readonly(synthetic_db, dst)
    assert out == dst
    assert dst.exists()

    # Mutate the snapshot and confirm source is untouched.
    conn = sqlite3.connect(str(dst))
    conn.execute("DELETE FROM memories")
    conn.commit()
    conn.close()

    assert synthetic_db.stat().st_size == src_size
    # mtime can wobble on some FSes; just ensure reads still return seeded data.
    conn2 = sqlite3.connect(str(synthetic_db))
    n = conn2.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    conn2.close()
    assert n == 4  # all original memories still there


# ----------------------------------------------------------------------
# 4. End-to-end with MockLLM
# ----------------------------------------------------------------------


def test_run_bench_with_mock_llm(bench, synthetic_db, tmp_path, monkeypatch):
    """Drive every stage with a deterministic MockLLM and confirm metrics shape."""
    # Make sure the curator code uses the synthetic DB and the mock path.
    monkeypatch.setenv("MEMOIRS_DB", str(synthetic_db))

    args = bench.build_arg_parser().parse_args([
        "--db", str(synthetic_db),
        "--mock",
        "--n-extract", "2",
        "--n-consolidate", "3",
        "--n-conflict", "3",
        "--no-conflict", "2",
        "--n-summary", "2",
        "--out", str(tmp_path / "r.json"),
    ])
    report = bench.run_bench(args)

    # Top-level shape.
    assert report["mode"] == "mock"
    assert report["backend"]
    assert report["timestamp"]
    for stage in ("extract", "consolidate", "contradictions", "summaries"):
        assert stage in report
        s = report[stage]
        assert "n" in s
        # latencies are tracked for any non-skip stage
        if not s.get("skipped"):
            assert "latency_p50_ms" in s
            assert "latency_p95_ms" in s

    # Consolidate uses MERGE-by-default mock; on synthetic candidates the
    # heuristic returns MERGE -> agreement should be high.
    co = report["consolidate"]
    assert co["n"] >= 1
    assert "agreement_with_heuristic" in co

    # Contradictions: mock says everything is contradictory -> precision = TP/(TP+FP).
    ct = report["contradictions"]
    assert ct["n"] == 5  # 3 contra + 2 non
    # With everything-contradictory mock: TP=3 (all contra hit), FP=2 (non flagged).
    assert ct["true_positives"] == 3
    assert ct["false_positives"] == 2
    assert ct["true_negatives"] == 0
    assert ct["false_negatives"] == 0

    # Summaries: mock returns a non-empty string -> json_valid_total > 0.
    sm = report["summaries"]
    assert sm["n"] == 2


def test_metrics_with_known_responses(bench, synthetic_db):
    """run_consolidate with a tightly-controlled MockLLM produces expected metrics."""
    # Force MERGE on every call. The synthesized candidates may include the
    # empty-suffix variant, which exact-matches an existing memory and thus
    # the heuristic returns UPDATE — so agreement won't be 100%, but
    # near-dup-correctness (Qwen action in {MERGE, UPDATE}) WILL be 100%.
    mock = bench.MockLLM(
        default_response='{"action":"MERGE","target_id":"mem_1","reason":"dup"}'
    )
    snapshot = synthetic_db
    stats = bench.run_consolidate(snapshot, n=4, llm_override=mock)
    d = stats.to_dict()
    assert d["n"] == 4
    # Near-dup correctness is the canonical metric here: MERGE counts as correct.
    assert d["near_dup_correctness_rate"] == pytest.approx(1.0)
    # All calls produced parseable JSON.
    assert d["json_valid_total"] == 4
    assert d.get("parse_errors", 0) == 0
    # Action distribution: only MERGE.
    assert d["actions_distribution"] == {"MERGE": 4}
    # Agreement is bounded but typically high.
    assert 0.5 <= d["agreement_with_heuristic"] <= 1.0


def test_render_table_contains_all_stages(bench, synthetic_db, tmp_path):
    args = bench.build_arg_parser().parse_args([
        "--db", str(synthetic_db),
        "--mock",
        "--n-extract", "1",
        "--n-consolidate", "2",
        "--n-conflict", "1",
        "--no-conflict", "1",
        "--n-summary", "1",
        "--out", str(tmp_path / "r.json"),
    ])
    report = bench.run_bench(args)
    table = bench.render_table(report)
    for stage in ("extract", "consolidate", "contradictions", "summaries"):
        assert stage in table
    assert "Qwen Quality Bench" in table


def test_skip_stages(bench, synthetic_db, tmp_path):
    args = bench.build_arg_parser().parse_args([
        "--db", str(synthetic_db),
        "--mock",
        "--skip-stages", "extract,summaries",
        "--n-extract", "5",
        "--n-consolidate", "1",
        "--n-conflict", "1",
        "--no-conflict", "1",
        "--n-summary", "5",
        "--out", str(tmp_path / "r.json"),
    ])
    report = bench.run_bench(args)
    assert report["extract"].get("skipped") is True
    assert report["summaries"].get("skipped") is True
    # Non-skipped stages still ran.
    assert report["consolidate"]["n"] >= 1
    assert report["contradictions"]["n"] == 2
