"""Hybrid retrieval — BM25 (FTS5) + dense (sqlite-vec) fused via RRF.

Combines lexical and semantic signals so queries with rare terms (model
names, tool names, identifiers) still surface even when their embedding
neighborhood is noisy, and queries with no token overlap still benefit
from semantic search.

Pipeline:

    query ─┬→ bm25_search   (FTS5 MATCH + bm25())  ─┐
           │                                        ├→ rrf_fuse → top_k
           └→ dense_search  (vec0 KNN, cosine)     ─┘

Reciprocal Rank Fusion (Cormack et al., 2009): for each ranking, the
contribution of an item at rank r is 1/(k+r). Summed over rankings, the
fused score has a stable scale across heterogeneous scorers (BM25 is
unbounded, cosine ∈ [-1,1]) — no normalization needed.

Schema:

    CREATE VIRTUAL TABLE memories_fts USING fts5(
        memory_id UNINDEXED,
        content,
        tokenize = 'unicode61 remove_diacritics 2'
    );

Triggers on `memories` keep `memories_fts` in sync. `ensure_fts_schema`
backfills any pre-existing rows on first call (idempotent).

Note: this module imports `embeddings` lazily inside `dense_search` so the
zero-dep mode stays usable for BM25-only callers.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from typing import Iterable, Sequence

from ..db import MemoirsDB, utc_now


log = logging.getLogger("memoirs.hybrid")


# ---------------------------------------------------------------------------
# Schema management
#
# TODO: depends on P0-1 migrations system — once `MemoirsDB.init()` runs the
# discovered migrations from `memoirs/migrations/`, the inline
# `ensure_fts_schema()` becomes a no-op safety net. Until then it carries the
# FTS5 schema for any fresh / pre-existing DB that hasn't applied migration
# 003 explicitly.
# ---------------------------------------------------------------------------


_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory_id UNINDEXED,
    content,
    tokenize = 'unicode61 remove_diacritics 2'
);

-- Triggers keep memories_fts mirrored to memories.content.
-- Archived rows are excluded: we delete on archive, re-insert on un-archive.
CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories
WHEN NEW.archived_at IS NULL
BEGIN
    INSERT INTO memories_fts(memory_id, content) VALUES (NEW.id, NEW.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories
BEGIN
    DELETE FROM memories_fts WHERE memory_id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE OF content, archived_at ON memories
BEGIN
    DELETE FROM memories_fts WHERE memory_id = OLD.id;
    INSERT INTO memories_fts(memory_id, content)
        SELECT NEW.id, NEW.content WHERE NEW.archived_at IS NULL;
END;
"""


def ensure_fts_schema(conn: sqlite3.Connection) -> bool:
    """Create the FTS5 virtual table + sync triggers if absent.

    Returns True iff this call performed first-time setup (and triggered a
    backfill). Idempotent: subsequent calls are cheap no-ops.
    """
    existed = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories_fts'"
    ).fetchone()
    conn.executescript(_FTS_SCHEMA)
    conn.commit()
    if existed:
        return False
    # First-time: backfill any pre-existing memories rows.
    inserted = rebuild_fts_index(conn)
    if inserted:
        log.info("ensure_fts_schema: backfilled %d memories into memories_fts", inserted)
    return True


def rebuild_fts_index(conn: sqlite3.Connection) -> int:
    """Truncate + repopulate `memories_fts` from `memories` (non-archived).

    Safe to call against very large corpora: a single `INSERT … SELECT` is
    handled in one transaction. Returns rows inserted.
    """
    # Make sure the virtual table exists before we try to populate it.
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5("
        "memory_id UNINDEXED, content, tokenize='unicode61 remove_diacritics 2')"
    )
    with conn:
        conn.execute("DELETE FROM memories_fts")
        cur = conn.execute(
            "INSERT INTO memories_fts(memory_id, content) "
            "SELECT id, content FROM memories WHERE archived_at IS NULL"
        )
        return int(cur.rowcount or 0)


# ---------------------------------------------------------------------------
# FTS5 query escaping
# ---------------------------------------------------------------------------

# Characters that have special meaning inside an FTS5 MATCH expression.
# We sanitize free-form user queries by quoting tokens.
_FTS_RESERVED = set('"():*+-^')


def _fts_quote(query: str) -> str:
    """Convert a free-form query into a safe FTS5 MATCH expression.

    Strategy: split on whitespace, drop tokens that contain only reserved
    characters, double-quote the rest, join with OR so partial matches still
    rank (FTS5 defaults to AND, which is too strict for RAG-style retrieval
    — a single missing token would zero out otherwise relevant docs).

    BM25 already penalizes thin matches; the OR keeps recall high and
    lets the scorer rank by how many query tokens land. Empty queries
    return ``""`` (caller must short-circuit).

    Examples:
        "hybrid retrieval"      → '"hybrid" OR "retrieval"'
        "C++ vs Rust"           → '"C" OR "vs" OR "Rust"' (++ stripped)
        "spawn(*) error"        → '"spawn" OR "error"'
    """
    tokens: list[str] = []
    for raw in query.split():
        cleaned = "".join(ch for ch in raw if ch not in _FTS_RESERVED).strip()
        if not cleaned:
            continue
        # Double any embedded quote (FTS5 phrase escaping).
        tokens.append('"' + cleaned.replace('"', '""') + '"')
    return " OR ".join(tokens)


# ---------------------------------------------------------------------------
# BM25 (lexical) — FTS5
# ---------------------------------------------------------------------------


def bm25_search(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 20,
    *,
    as_of: str | None = None,
) -> list[tuple[str, float]]:
    """Lexical search over `memories_fts`. Returns [(memory_id, bm25_score)].

    SQLite's `bm25()` returns LOWER == better. We negate it so callers can
    treat higher = better consistently; the raw value is otherwise opaque.
    Filters by temporal validity / archive state to mirror dense_search().
    """
    ensure_fts_schema(conn)
    fts_q = _fts_quote(query)
    if not fts_q:
        return []
    ts = as_of or utc_now()
    if as_of is None:
        sql = """
            SELECT m.id AS memory_id, -bm25(memories_fts) AS score
            FROM memories_fts
            JOIN memories m ON m.id = memories_fts.memory_id
            WHERE memories_fts MATCH ?
              AND m.archived_at IS NULL
              AND (m.valid_to IS NULL OR m.valid_to >= ?)
            ORDER BY bm25(memories_fts) ASC
            LIMIT ?
        """
        params = (fts_q, ts, top_k)
    else:
        sql = """
            SELECT m.id AS memory_id, -bm25(memories_fts) AS score
            FROM memories_fts
            JOIN memories m ON m.id = memories_fts.memory_id
            WHERE memories_fts MATCH ?
              AND (m.valid_from IS NULL OR m.valid_from <= ?)
              AND (m.valid_to   IS NULL OR m.valid_to   >  ?)
              AND (m.archived_at IS NULL OR m.archived_at > ?)
            ORDER BY bm25(memories_fts) ASC
            LIMIT ?
        """
        params = (fts_q, ts, ts, ts, top_k)
    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as e:
        log.warning("bm25_search: FTS query rejected (%s) for q=%r", e, fts_q)
        return []
    return [(r[0] if not hasattr(r, "keys") else r["memory_id"], float(r[1] if not hasattr(r, "keys") else r["score"])) for r in rows]


# ---------------------------------------------------------------------------
# Dense (semantic) — wraps embeddings.search_similar_memories
# ---------------------------------------------------------------------------


def dense_search(
    db: MemoirsDB,
    query: str,
    top_k: int = 20,
    *,
    as_of: str | None = None,
) -> list[tuple[str, float]]:
    """Cosine-similarity ANN over `vec_memories`. Returns [(memory_id, sim)].

    Thin wrapper over `embeddings.search_similar_memories`; kept here so
    `hybrid_search` doesn't have to reach into another module's internals
    and so callers can mock it cleanly in tests.
    """
    from . import embeddings as emb  # lazy: keep BM25 path zero-dep
    rows = emb.search_similar_memories(db, query, top_k=top_k, as_of=as_of)
    return [(m["id"], float(m.get("similarity", 0.0))) for m in rows]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def rrf_fuse(
    rankings: Sequence[Sequence[tuple[str, float]]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Combine N rankings via RRF: score(d) = Σ_r 1/(k + rank_r(d)).

    `k` defaults to 60 (Cormack et al.); larger k flattens the rank curve so
    later positions still contribute. The output is sorted descending by
    fused score.

    Items absent from a ranking simply contribute 0 from that ranking — no
    penalty, no bonus. Ties in a single ranking share the same rank index
    (the order they arrived in).
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (memory_id, _score) in enumerate(ranking, start=1):
            fused[memory_id] = fused.get(memory_id, 0.0) + 1.0 / (k + rank)
    out = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    if top_k is not None:
        out = out[:top_k]
    return out


# ---------------------------------------------------------------------------
# Hybrid search (the public entry point)
# ---------------------------------------------------------------------------


def hybrid_search(
    db: MemoirsDB,
    query: str,
    *,
    top_k: int = 20,
    bm25_k: int | None = None,
    dense_k: int | None = None,
    rrf_k: int = 60,
    as_of: str | None = None,
    with_embeddings: bool = True,
) -> list[dict]:
    """Run BM25 + dense in parallel and fuse via RRF.

    Returns a list of dicts: `{id, score, bm25_rank, dense_rank, bm25_score,
    dense_score}` — `*_rank` is None when the doc is missing from that
    ranking. `score` is the RRF fused value. The ordering is
    score-descending.

    `with_embeddings=False` skips dense entirely (handy when sqlite-vec /
    sentence-transformers are unavailable and the caller still wants
    something better than a hard failure).
    """
    bm25_top = bm25_k or max(top_k * 2, 20)
    dense_top = dense_k or max(top_k * 2, 20)

    bm25_results = bm25_search(db.conn, query, top_k=bm25_top, as_of=as_of)
    dense_results: list[tuple[str, float]] = []
    if with_embeddings:
        try:
            dense_results = dense_search(db, query, top_k=dense_top, as_of=as_of)
        except Exception as e:
            # Embeddings may legitimately be unavailable (no model, no extension).
            # We fall through to BM25-only rather than raise.
            log.warning("hybrid_search: dense search unavailable (%s) — BM25-only", e)
            dense_results = []

    bm25_index = {mid: (rank, score) for rank, (mid, score) in enumerate(bm25_results, start=1)}
    dense_index = {mid: (rank, score) for rank, (mid, score) in enumerate(dense_results, start=1)}

    fused = rrf_fuse([bm25_results, dense_results], k=rrf_k, top_k=top_k)

    out: list[dict] = []
    for mid, fscore in fused:
        b_rank, b_score = bm25_index.get(mid, (None, None))
        d_rank, d_score = dense_index.get(mid, (None, None))
        out.append({
            "id": mid,
            "score": round(float(fscore), 6),
            "bm25_rank": b_rank,
            "dense_rank": d_rank,
            "bm25_score": b_score,
            "dense_score": d_score,
        })
    return out


# ---------------------------------------------------------------------------
# Convenience: hydrate IDs into full memory rows (matching dense_search shape)
# ---------------------------------------------------------------------------


def hydrate_memories(
    db: MemoirsDB,
    fused: Iterable[dict],
    *,
    as_of: str | None = None,
) -> list[dict]:
    """Fetch full memory rows for the IDs in `fused`, preserving order.

    Returned dicts mirror `embeddings.search_similar_memories` so the rest of
    the engine (conflict detection, _compress_context, …) keeps working
    transparently. We use the BM25 score in `similarity` slot when dense is
    absent — only as a relevance proxy for downstream sort fallbacks.
    """
    fused_list = list(fused)
    if not fused_list:
        return []
    ids = [f["id"] for f in fused_list]
    placeholders = ",".join("?" * len(ids))
    ts = as_of or utc_now()
    if as_of is None:
        sql = f"""
            SELECT id, type, content, importance, confidence, score,
                   usage_count, last_used_at, valid_from, valid_to, archived_at
            FROM memories
            WHERE id IN ({placeholders})
              AND archived_at IS NULL
              AND (valid_to IS NULL OR valid_to >= ?)
        """
        rows = db.conn.execute(sql, (*ids, ts)).fetchall()
    else:
        sql = f"""
            SELECT id, type, content, importance, confidence, score,
                   usage_count, last_used_at, valid_from, valid_to, archived_at
            FROM memories
            WHERE id IN ({placeholders})
              AND (valid_from IS NULL OR valid_from <= ?)
              AND (valid_to IS NULL OR valid_to > ?)
              AND (archived_at IS NULL OR archived_at > ?)
        """
        rows = db.conn.execute(sql, (*ids, ts, ts, ts)).fetchall()
    by_id = {r["id"]: dict(r) for r in rows}
    out: list[dict] = []
    for f in fused_list:
        row = by_id.get(f["id"])
        if row is None:
            continue
        # Surface fused score as `similarity` so the downstream pipeline,
        # which uses `m.get('similarity')`, sees a meaningful value.
        # Use dense_score if present (true cosine), otherwise the RRF score.
        if f.get("dense_score") is not None:
            row["similarity"] = round(float(f["dense_score"]), 4)
        else:
            row["similarity"] = round(float(f["score"]), 4)
        row["fused_score"] = f["score"]
        row["bm25_rank"] = f.get("bm25_rank")
        row["dense_rank"] = f.get("dense_rank")
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Lightweight microbenchmark helper (used by tests)
# ---------------------------------------------------------------------------


def benchmark_query(
    db: MemoirsDB,
    query: str,
    *,
    top_k: int = 10,
    repeat: int = 3,
) -> dict:
    """Time hybrid_search vs dense_search vs bm25_search. Returns ms each."""
    def _time(fn) -> float:
        # Warm once, then median of `repeat` runs.
        fn()
        samples = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1000.0)
        samples.sort()
        return samples[len(samples) // 2]

    return {
        "hybrid_ms": _time(lambda: hybrid_search(db, query, top_k=top_k)),
        "dense_ms": _time(lambda: dense_search(db, query, top_k=top_k)),
        "bm25_ms": _time(lambda: bm25_search(db.conn, query, top_k=top_k)),
    }
