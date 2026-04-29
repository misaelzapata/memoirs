"""Tests for scripts/audit_data_utility.py — Phase 5C data utility audit."""
from __future__ import annotations

import array
import importlib.util
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# The audit script lives under scripts/ (not a package), so we import it
# manually via importlib to keep the script file self-contained without
# polluting sys.path globally.

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_AUDIT_PATH = _SCRIPTS_DIR / "audit_data_utility.py"


def _load_audit_module():
    import sys
    spec = importlib.util.spec_from_file_location("audit_data_utility", _AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass annotation resolution can find the module.
    sys.modules["audit_data_utility"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def audit():
    return _load_audit_module()


# ---------------------------------------------------------------------------
# Helpers — build a synthetic DB the audit script can read.
# ---------------------------------------------------------------------------


def _pack_vec(vec: list[float]) -> bytes:
    return array.array("f", vec).tobytes()


def _seed_synthetic_db(db_path: Path, *, with_rejected: bool = False) -> dict:
    """Create a DB with a small fixture corpus the audit can chew on.

    Returns a dict describing what was inserted (counts, ids, etc.) so tests
    can assert against it without re-deriving expectations.
    """
    from memoirs.db import MemoirsDB

    db = MemoirsDB(db_path)
    db.init()

    now = datetime(2026, 4, 27, tzinfo=timezone.utc)
    fresh = now.isoformat()
    stale_ts = (now - timedelta(days=45)).isoformat()
    very_stale = (now - timedelta(days=120)).isoformat()
    recent = (now - timedelta(days=3)).isoformat()

    # Five memories with controlled embeddings:
    # - mem_a / mem_b are near-duplicates (>0.95 cosine)
    # - mem_c is orthogonal to both
    # - mem_stale is old + unused
    # - mem_short is noise (too short, no proper noun, generic)
    fixtures = [
        # Useful, specific, fresh, recent access
        {
            "id": "mem_a",
            "type": "decision",
            "content": "We decided to migrate the Memoirs vector store to LanceDB for performance.",
            "importance": 4,
            "score": 0.7,
            "usage_count": 3,
            "last_accessed_at": fresh,
            "created_at": fresh,
            "vec": [1.0, 0.0, 0.0],
        },
        # Useful but extremely similar to mem_a (the duplicate that survived)
        {
            "id": "mem_b",
            "type": "decision",
            "content": "We decided to migrate the Memoirs vector store to LanceDB to improve performance.",
            "importance": 4,
            "score": 0.65,
            "usage_count": 2,
            "last_accessed_at": fresh,
            "created_at": fresh,
            "vec": [0.99, 0.05, 0.0],
        },
        # Useful, distinct
        {
            "id": "mem_c",
            "type": "preference",
            "content": "Misael prefers Python for backend services rather than Go.",
            "importance": 4,
            "score": 0.6,
            "usage_count": 1,
            "last_accessed_at": recent,
            "created_at": recent,
            "vec": [0.0, 1.0, 0.0],
        },
        # Stale: old + zero usage
        {
            "id": "mem_stale",
            "type": "fact",
            "content": "The Firecracker VM is left paused so the caller can resume it later on demand.",
            "importance": 2,
            "score": 0.4,
            "usage_count": 0,
            "last_accessed_at": None,
            "created_at": stale_ts,
            "vec": [0.0, 0.0, 1.0],
        },
        # Noise: too short, generic, low importance
        {
            "id": "mem_noise",
            "type": "fact",
            "content": "the user said ok",
            "importance": 1,
            "score": 0.2,
            "usage_count": 0,
            "last_accessed_at": None,
            "created_at": very_stale,
            "vec": [0.5, 0.5, 0.0],
        },
    ]

    for f in fixtures:
        db.conn.execute(
            "INSERT INTO memories (id, type, content, content_hash, importance, "
            "confidence, score, usage_count, last_accessed_at, user_signal, "
            "valid_from, metadata_json, created_at, updated_at) "
            "VALUES (?, ?, ?, 'h_'||?, ?, 0.7, ?, ?, ?, 0, ?, '{}', ?, ?)",
            (
                f["id"], f["type"], f["content"], f["id"], f["importance"],
                f["score"], f["usage_count"], f["last_accessed_at"],
                f["created_at"], f["created_at"], f["created_at"],
            ),
        )
        db.conn.execute(
            "INSERT INTO memory_embeddings (memory_id, dim, embedding, model, created_at) "
            "VALUES (?, ?, ?, 'test-model', ?)",
            (f["id"], len(f["vec"]), _pack_vec(f["vec"]), f["created_at"]),
        )

    if with_rejected:
        # One rejected candidate that the heuristic would call USEFUL → false negative.
        db.conn.execute(
            "INSERT INTO memory_candidates (id, type, content, importance, confidence, "
            "status, rejection_reason, raw_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 0.7, 'rejected', 'extractor noise', '{}', ?, ?)",
            (
                "cand_useful",
                "decision",
                "Misael decided to ship the Phase 5C audit before merging the GAP branch.",
                4,
                fresh,
                fresh,
            ),
        )
        # And a clearly noisy rejected candidate.
        db.conn.execute(
            "INSERT INTO memory_candidates (id, type, content, importance, confidence, "
            "status, rejection_reason, raw_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 0.4, 'rejected', 'too short', '{}', ?, ?)",
            ("cand_noise", "fact", "ok", 1, fresh, fresh),
        )

    db.conn.commit()
    db.close()
    return {"fixtures": fixtures}


# ---------------------------------------------------------------------------
# 1. Heuristic correctness
# ---------------------------------------------------------------------------


def test_heuristic_classifies_specific_as_useful(audit):
    v = audit.heuristic_classify(
        "Misael decided to migrate Memoirs from sqlite-vec to LanceDB.",
        mem_type="decision",
        importance=4,
    )
    assert v.label == "useful", v


def test_heuristic_classifies_generic_as_noise(audit):
    v = audit.heuristic_classify(
        "the user said ok",
        mem_type="fact",
        importance=1,
    )
    assert v.label == "noise", v
    assert any("filler" in r or "len" in r or "proper" in r for r in v.reasons)


def test_heuristic_borderline_when_two_signals_match(audit):
    # No proper noun, no filler, length OK, importance OK for non-strong type
    v = audit.heuristic_classify(
        "we typically run the test suite via pytest before merging changes.",
        mem_type="fact",
        importance=2,
    )
    # This satisfies: shape_ok, no_filler, type_importance_ok (fact is not strong) -> useful
    assert v.label in {"useful", "borderline"}


def test_heuristic_strong_type_low_importance_penalised(audit):
    v = audit.heuristic_classify(
        "Misael prefers Python rather than Go for backend services.",
        mem_type="preference",
        importance=1,  # below threshold for strong type
    )
    # Even with proper noun + length + no filler, the strong-type importance
    # penalty knocks us down — but still has 3/4 signals so "useful".
    assert v.type_importance_ok is False


# ---------------------------------------------------------------------------
# 2. Duplicate detection
# ---------------------------------------------------------------------------


def test_find_duplicate_pairs_detects_high_similarity(audit):
    rows = [
        {"id": "a", "type": "fact", "content": "alpha", "vec": [1.0, 0.0, 0.0]},
        {"id": "b", "type": "fact", "content": "alpha-clone", "vec": [0.99, 0.05, 0.0]},
        {"id": "c", "type": "fact", "content": "beta", "vec": [0.0, 1.0, 0.0]},
        {"id": "d", "type": "fact", "content": "gamma", "vec": [0.0, 0.0, 1.0]},
        {"id": "e", "type": "fact", "content": "delta", "vec": [0.5, 0.5, 0.0]},
    ]
    pairs = audit.find_duplicate_pairs(rows, threshold=0.9)
    assert len(pairs) == 1
    p = pairs[0]
    assert {p["a_id"], p["b_id"]} == {"a", "b"}
    assert p["sim"] > 0.9


def test_cosine_zero_vector_returns_zero(audit):
    assert audit.cosine([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# 3. Stale calculation
# ---------------------------------------------------------------------------


def test_stale_counts_match_expected(tmp_path, audit):
    db_path = tmp_path / "synth.sqlite"
    _seed_synthetic_db(db_path)
    conn = audit.open_ro(db_path)
    try:
        stale = audit.compute_stale(conn)
    finally:
        conn.close()
    # mem_stale (45d, uc=0) and mem_noise (120d, uc=0) should both count.
    assert stale["total"] == 2
    assert stale["active_total"] == 5
    assert stale["rate"] == pytest.approx(2 / 5, abs=1e-3)
    assert "fact" in stale["by_type"]
    assert stale["by_type"]["fact"] == 2


# ---------------------------------------------------------------------------
# 4. CLI smoke + JSON shape
# ---------------------------------------------------------------------------


def test_cli_smoke_writes_valid_report(tmp_path, audit):
    db_path = tmp_path / "synth.sqlite"
    out_path = tmp_path / "report.json"
    _seed_synthetic_db(db_path, with_rejected=True)

    rc = audit.main(
        [
            "--db", str(db_path),
            "--out", str(out_path),
            "--sample", "5",
            "--rejected-sample", "5",
            "--dup-threshold", "0.9",
            "--dup-subset", "0",
            "--seed", "1",
            "--quiet",
        ]
    )
    assert rc == 0
    assert out_path.exists()
    report = json.loads(out_path.read_text())
    # Required top-level fields per the GAP spec.
    for key in [
        "timestamp", "corpus_size", "sample", "qwen_judge",
        "false_negatives", "duplicates", "stale", "distributions",
        "top_memories", "bottom_memories",
    ]:
        assert key in report, f"missing {key}"
    # Corpus size matches our 5 fixtures.
    assert report["corpus_size"] == 5
    # Sample summary shape.
    assert report["sample"]["n_evaluated"] == 5
    assert report["sample"]["useful"] + report["sample"]["noise"] + report["sample"]["borderline"] == 5
    # Duplicates: mem_a / mem_b should be flagged at 0.9.
    assert report["duplicates"]["n_pairs_above_threshold"] == 1
    # False negatives: cand_useful (decision, importance=4) is heuristically useful.
    fn = report["false_negatives"]
    assert fn["available"] is True
    assert fn["n_evaluated"] == 2
    assert fn["n_useful_rejected"] == 1
    # Stale rate matches.
    assert report["stale"]["total"] == 2
    # Qwen judge defaults to disabled.
    assert report["qwen_judge"]["enabled"] is False


def test_cli_missing_db_returns_nonzero(tmp_path, audit):
    out_path = tmp_path / "out.json"
    rc = audit.main(
        ["--db", str(tmp_path / "nope.sqlite"), "--out", str(out_path), "--quiet"]
    )
    assert rc != 0


# ---------------------------------------------------------------------------
# 5. Read-only safety: production DB cannot be mutated.
# ---------------------------------------------------------------------------


def test_open_ro_refuses_writes(tmp_path, audit):
    db_path = tmp_path / "synth.sqlite"
    _seed_synthetic_db(db_path)
    conn = audit.open_ro(db_path)
    try:
        with pytest.raises(Exception):
            conn.execute("DELETE FROM memories")
    finally:
        conn.close()
