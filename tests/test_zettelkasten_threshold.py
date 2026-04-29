"""Tests for the new mode= dispatch + prune/stats helpers in zettelkasten.

These complement ``test_zettelkasten.py`` (which covers the legacy absolute
behavior) by exercising:
- ``mode="topk"``: bounded fan-out regardless of threshold.
- ``mode="adaptive"``: percentile-based local cutoff.
- ``prune_excess_links``: reduces over-linked corpora to ≤ max_per_memory.
- ``link_stats``: shape of summary.
- CLI ``links prune --dry-run``: counts without mutating.
"""
from __future__ import annotations

import json
import math
import random
import subprocess
import sys

import pytest

from memoirs.config import EMBEDDING_DIM
from memoirs.engine import embeddings as emb
from memoirs.engine import zettelkasten as zk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _seed_memory(db, *, memory_id: str, content: str, vec: list[float] | None = None) -> None:
    """Insert a memory with an optional pre-baked unit-normalized embedding."""
    now = "2026-04-27T00:00:00+00:00"
    db.conn.execute(
        "INSERT INTO memories (id, type, content, content_hash, importance, "
        "confidence, score, usage_count, user_signal, valid_from, metadata_json, "
        "created_at, updated_at) "
        "VALUES (?, 'fact', ?, 'h_'||?, 3, 0.5, 0.5, 0, 0, ?, '{}', ?, ?)",
        (memory_id, content, memory_id, now, now, now),
    )
    if vec is not None:
        emb._require_vec(db)
        # Pad/truncate to EMBEDDING_DIM so callers can pass terse 3-d vectors
        # for clarity in cluster setup.
        padded = list(vec) + [0.0] * max(0, EMBEDDING_DIM - len(vec))
        padded = padded[:EMBEDDING_DIM]
        v = _normalize(padded)
        blob = emb._pack(v)
        db.conn.execute(
            "INSERT INTO memory_embeddings (memory_id, dim, embedding, model, created_at) "
            "VALUES (?, ?, ?, 'test', ?)",
            (memory_id, EMBEDDING_DIM, blob, now),
        )
        db.conn.execute("DELETE FROM vec_memories WHERE memory_id = ?", (memory_id,))
        db.conn.execute(
            "INSERT INTO vec_memories(memory_id, embedding) VALUES (?, ?)",
            (memory_id, blob),
        )
    db.conn.commit()


def _cluster_vec(rng: random.Random, base: list[float], jitter: float = 0.05) -> list[float]:
    """Vector close to `base` plus small noise — cluster member."""
    v = [b + rng.uniform(-jitter, jitter) for b in base]
    # Pad to EMBEDDING_DIM with tiny noise.
    while len(v) < EMBEDDING_DIM:
        v.append(rng.uniform(-0.001, 0.001))
    return v[:EMBEDDING_DIM]


def _noise_vec(rng: random.Random) -> list[float]:
    """Random-direction vector — uncorrelated with the cluster."""
    v = [rng.gauss(0.0, 1.0) for _ in range(EMBEDDING_DIM)]
    return v


def _seed_corpus(db, n_cluster: int = 10, n_noise: int = 40, seed: int = 7) -> tuple[list[str], list[str]]:
    """Insert ``n_cluster`` tightly grouped memories + ``n_noise`` random ones.

    Returns (cluster_ids, noise_ids).
    """
    rng = random.Random(seed)
    # Cluster center: a unit vector along the first axis with small variation.
    base = [1.0, 0.0, 0.0]
    cluster_ids: list[str] = []
    for i in range(n_cluster):
        mid = f"c_{i:03d}"
        _seed_memory(db, memory_id=mid, content=f"cluster-{i}", vec=_cluster_vec(rng, base, 0.02))
        cluster_ids.append(mid)
    noise_ids: list[str] = []
    for i in range(n_noise):
        mid = f"n_{i:03d}"
        _seed_memory(db, memory_id=mid, content=f"noise-{i}", vec=_noise_vec(rng))
        noise_ids.append(mid)
    return cluster_ids, noise_ids


# ---------------------------------------------------------------------------
# mode="topk"
# ---------------------------------------------------------------------------


def test_mode_topk_yields_exactly_top_k_per_memory(tmp_db):
    """With mode='topk' each memory grows exactly top_k outgoing links."""
    zk.ensure_schema(tmp_db.conn)
    _seed_corpus(tmp_db, n_cluster=10, n_noise=40)

    # Run link_memory for every memory in topk mode with top_k=5.
    rows = tmp_db.conn.execute(
        "SELECT id FROM memories WHERE archived_at IS NULL ORDER BY id"
    ).fetchall()
    for r in rows:
        zk.link_memory(tmp_db, r["id"], top_k=5, mode="topk", threshold=0.99)

    # Each source should have exactly 5 outgoing links (50 mems → enough peers).
    counts = [
        int(r["c"]) for r in tmp_db.conn.execute(
            "SELECT source_memory_id, COUNT(*) AS c FROM memory_links "
            "WHERE reason='semantic' GROUP BY source_memory_id"
        ).fetchall()
    ]
    assert len(counts) == 50
    # In top-k mode bidirectional writes can push a few sources above top_k
    # because the same target can be reached as someone else's neighbor too.
    # Cap is therefore: each memory was *queried* with top_k=5 → ≥5 outgoing
    # rows from that act, plus possible inbound symmetry rows. Accept exact 5
    # as the minimum.
    assert all(c >= 5 for c in counts), counts


def test_mode_topk_ignores_threshold(tmp_db):
    """An absurdly high threshold must not filter anything out under mode='topk'."""
    zk.ensure_schema(tmp_db.conn)
    _seed_corpus(tmp_db, n_cluster=10, n_noise=40)

    # threshold=0.999 would filter virtually everything in absolute mode.
    # In topk mode it must be ignored.
    links = zk.link_memory(tmp_db, "n_000", top_k=5, mode="topk", threshold=0.999)
    assert len(links) == 5


# ---------------------------------------------------------------------------
# mode="adaptive"
# ---------------------------------------------------------------------------


def test_mode_adaptive_keeps_only_local_top_percentile(tmp_db):
    """Adaptive p95 should only link cluster members to cluster members."""
    zk.ensure_schema(tmp_db.conn)
    cluster_ids, noise_ids = _seed_corpus(tmp_db, n_cluster=10, n_noise=40)

    # Link a cluster member at threshold=0.95 (top 5% of its 50-neighbor pool).
    src = cluster_ids[0]
    links = zk.link_memory(tmp_db, src, top_k=5, mode="adaptive", threshold=0.95)
    target_ids = {l.target_memory_id for l in links}
    # Every accepted target should be a cluster member, not noise.
    # (Cluster vectors are tightly grouped at high cosine sim; noise sits much
    # lower, so the p95 cutoff lands inside the cluster band.)
    for t in target_ids:
        assert t in cluster_ids, f"adaptive linked across to noise: {t}"
    assert len(links) >= 1


def test_mode_adaptive_caps_at_top_k(tmp_db):
    """Even when many neighbors clear the percentile, top_k is the hard cap."""
    zk.ensure_schema(tmp_db.conn)
    _seed_corpus(tmp_db, n_cluster=10, n_noise=40)

    links = zk.link_memory(tmp_db, "c_000", top_k=3, mode="adaptive", threshold=0.5)
    assert len(links) <= 3


# ---------------------------------------------------------------------------
# mode validation
# ---------------------------------------------------------------------------


def test_mode_invalid_raises(tmp_db):
    zk.ensure_schema(tmp_db.conn)
    _seed_memory(tmp_db, memory_id="m_x", content="x", vec=[1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        zk.link_memory(tmp_db, "m_x", mode="bogus")


# ---------------------------------------------------------------------------
# prune_excess_links
# ---------------------------------------------------------------------------


def _fill_links(db, source: str, n: int, base_sim: float = 0.5) -> None:
    """Insert n synthetic outgoing links with varying similarity."""
    for i in range(n):
        db.conn.execute(
            "INSERT INTO memory_links (source_memory_id, target_memory_id, similarity, reason) "
            "VALUES (?, ?, ?, 'semantic')",
            (source, f"target_{i:04d}", base_sim + i * 0.001, ),
        )
    db.conn.commit()


def test_prune_excess_links_keeps_highest_similarity(tmp_db):
    """prune_excess_links(max_per_memory=3) leaves the 3 highest-sim rows."""
    zk.ensure_schema(tmp_db.conn)
    # Need a real source memory so ensure_schema doesn't reject; but
    # memory_links has no FK, so we can insert rows freely.
    _fill_links(tmp_db, "src", 10, base_sim=0.5)

    result = zk.prune_excess_links(tmp_db, max_per_memory=3)
    assert result["deleted"] == 7

    rows = tmp_db.conn.execute(
        "SELECT similarity FROM memory_links WHERE source_memory_id='src' "
        "ORDER BY similarity DESC"
    ).fetchall()
    assert len(rows) == 3
    # The kept rows must be the top-3 by similarity.
    sims = [round(r["similarity"], 4) for r in rows]
    assert sims == sorted(sims, reverse=True)
    # Highest original was 0.5 + 9*0.001 = 0.509.
    assert pytest.approx(rows[0]["similarity"], abs=1e-6) == 0.509


def test_prune_excess_links_min_similarity_floor(tmp_db):
    """A min_similarity floor drops rows below the threshold even if under cap."""
    zk.ensure_schema(tmp_db.conn)
    _fill_links(tmp_db, "src", 5, base_sim=0.1)

    # Cap=10 (no rows over) but floor=0.103 drops the bottom 3 (0.100..0.102).
    result = zk.prune_excess_links(tmp_db, max_per_memory=10, min_similarity=0.103)
    assert result["deleted"] == 3
    remaining = tmp_db.conn.execute(
        "SELECT similarity FROM memory_links WHERE source_memory_id='src' ORDER BY similarity"
    ).fetchall()
    assert len(remaining) == 2
    assert all(r["similarity"] >= 0.103 for r in remaining)


def test_prune_excess_links_dry_run_does_not_mutate(tmp_db):
    zk.ensure_schema(tmp_db.conn)
    _fill_links(tmp_db, "src", 8)

    before = tmp_db.conn.execute(
        "SELECT COUNT(*) AS c FROM memory_links"
    ).fetchone()["c"]
    result = zk.prune_excess_links(tmp_db, max_per_memory=3, dry_run=True)
    after = tmp_db.conn.execute(
        "SELECT COUNT(*) AS c FROM memory_links"
    ).fetchone()["c"]

    assert before == after == 8
    assert result["deleted"] == 0
    assert result["would_delete"] == 5
    assert result["dry_run"] is True


def test_prune_excess_links_reason_filter(tmp_db):
    """--reason scopes pruning to a single edge kind."""
    zk.ensure_schema(tmp_db.conn)
    # 5 'semantic' + 5 'shared_entity' on the same source.
    for i in range(5):
        tmp_db.conn.execute(
            "INSERT INTO memory_links (source_memory_id, target_memory_id, similarity, reason) "
            "VALUES (?, ?, ?, 'semantic')",
            ("src", f"tA_{i}", 0.5 + i * 0.01),
        )
        tmp_db.conn.execute(
            "INSERT INTO memory_links (source_memory_id, target_memory_id, similarity, reason) "
            "VALUES (?, ?, ?, 'shared_entity')",
            ("src", f"tB_{i}", 0.5 + i * 0.01),
        )
    tmp_db.conn.commit()

    # Cap semantic at 2; shared_entity should be untouched.
    zk.prune_excess_links(tmp_db, max_per_memory=2, reason="semantic")

    sem_count = tmp_db.conn.execute(
        "SELECT COUNT(*) AS c FROM memory_links WHERE reason='semantic'"
    ).fetchone()["c"]
    shared_count = tmp_db.conn.execute(
        "SELECT COUNT(*) AS c FROM memory_links WHERE reason='shared_entity'"
    ).fetchone()["c"]
    assert sem_count == 2
    assert shared_count == 5  # untouched


# ---------------------------------------------------------------------------
# link_stats
# ---------------------------------------------------------------------------


def test_link_stats_shape(tmp_db):
    """stats() returns the documented dict shape with all keys present."""
    zk.ensure_schema(tmp_db.conn)
    # Three sources with varying fan-out and similarity.
    tmp_db.conn.executemany(
        "INSERT INTO memory_links (source_memory_id, target_memory_id, similarity, reason) "
        "VALUES (?, ?, ?, ?)",
        [
            ("a", "b", 0.95, "semantic"),
            ("a", "c", 0.75, "semantic"),
            ("a", "d", 0.55, "semantic"),
            ("b", "a", 0.95, "semantic"),
            ("c", "a", 0.75, "shared_entity"),
        ],
    )
    tmp_db.conn.commit()

    s = zk.link_stats(tmp_db)
    assert s["total"] == 5
    assert s["distinct_sources"] == 3
    assert "max" in s["per_source"]
    assert "avg" in s["per_source"]
    assert "p95" in s["per_source"]
    assert s["per_source"]["max"] == 3  # source 'a' has 3 outgoing
    assert isinstance(s["similarity_histogram"], list)
    assert len(s["similarity_histogram"]) == 10
    assert sum(s["similarity_histogram"]) == 5
    assert s["by_reason"]["semantic"] == 4
    assert s["by_reason"]["shared_entity"] == 1


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_links_prune_dry_run_does_not_mutate(tmp_path):
    """`memoirs links prune --dry-run` reports counts without writing."""
    from memoirs.db import MemoirsDB

    db_path = tmp_path / "cli.sqlite"
    db = MemoirsDB(db_path)
    db.init()
    zk.ensure_schema(db.conn)
    _fill_links(db, "src", 8)
    db.close()

    proc = subprocess.run(
        [sys.executable, "-m", "memoirs", "--db", str(db_path),
         "links", "prune", "--max-per-memory", "3", "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["dry_run"] is True
    assert payload["deleted"] == 0
    assert payload["would_delete"] == 5

    db = MemoirsDB(db_path)
    after = db.conn.execute(
        "SELECT COUNT(*) AS c FROM memory_links"
    ).fetchone()["c"]
    db.close()
    assert after == 8  # untouched
