"""A-MEM Zettelkasten linking — memory↔memory bidirectional graph.

Implements GAP P1-3: at insert time (or via backfill), each memory grows a
small set of bidirectional links to the most relevant existing memories.
Links carry a `reason` so we can mix semantic neighbors with structural ones
(shared entities, shared tags, temporal proximity) without losing provenance.

Design choices
--------------
- Top-k semantic neighbors are computed against the **stored** embedding of
  the source memory (from `memory_embeddings`). We avoid re-encoding the text
  unless no embedding exists — that keeps `link_memory` cheap (<50ms for ~1k
  memorias on commodity hardware, dominated by the vec0 ANN scan).
- Similarity = `1 - distance² / 2` for unit-normalized vectors (the same
  cosine approximation used by `embeddings.search_similar_memories`).
- Bidirectional edges are stored explicitly (A→B and B→A) so neighbor SQL
  stays a single ``WHERE source = ?`` lookup, no UNION required.
- The UNIQUE constraint on `(source, target, reason)` makes the linker
  idempotent — re-linking the same pair upserts the similarity rather than
  duplicating rows.
- ``ensure_schema`` is a fallback: if migration 002 has been applied this is
  a no-op; if the DB pre-dates the migration system, it bootstraps the table
  so the module is fully standalone.

Reasons currently produced
--------------------------
- ``semantic``      — top-k nearest neighbor by cosine
- ``shared_entity`` — both memories link to the same entity in
                      ``memory_entities`` (Layer 3 KG overlap)

Future reasons (not implemented yet but reserved by the schema):
- ``shared_tag``    — explicit user/tag overlap
- ``temporal``      — created within the same window
"""
from __future__ import annotations

import logging
import math
import os
import sqlite3
import struct
from dataclasses import dataclass, asdict
from typing import Iterable

from . import embeddings as emb


log = logging.getLogger("memoirs.zettelkasten")


# --------------------------------------------------------------------------
# Public dataclasses
# --------------------------------------------------------------------------


@dataclass
class Link:
    source_memory_id: str
    target_memory_id: str
    similarity: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MemoryRef:
    """A memory reachable from a starting node, with traversal metadata."""

    memory_id: str
    type: str
    content: str
    depth: int
    similarity: float       # similarity along the path that brought us here
    reason: str             # reason of the edge that brought us here

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Schema bootstrap (fallback when migration system isn't run)
# --------------------------------------------------------------------------


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_memory_id TEXT NOT NULL,
    target_memory_id TEXT NOT NULL,
    similarity REAL NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source_memory_id, target_memory_id, reason)
);
CREATE INDEX IF NOT EXISTS idx_memory_links_source ON memory_links(source_memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_links_target ON memory_links(target_memory_id);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create `memory_links` if it doesn't exist.

    Safe to call repeatedly. Acts as a fallback for code paths that don't
    run the formal migration system (e.g. tests using ``MemoirsDB.init()``,
    or workspaces created before migration 002 was added).
    """
    conn.executescript(_SCHEMA_SQL)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _is_enabled() -> bool:
    """Honor ``MEMOIRS_ZETTELKASTEN`` env (default on)."""
    val = os.environ.get("MEMOIRS_ZETTELKASTEN", "on").strip().lower()
    return val not in {"0", "off", "false", "no"}


def _distance_to_similarity(distance: float) -> float:
    """Map vec0 L2 distance on unit-normalized vectors to cosine similarity.

    The same approximation `1 - d²/2` used elsewhere in the codebase. Clamped
    to [0, 1] so floating-point drift doesn't yield negatives.
    """
    return max(0.0, min(1.0, 1.0 - (distance ** 2) / 2.0))


def _fetch_stored_embedding(db, memory_id: str) -> bytes | None:
    row = db.conn.execute(
        "SELECT embedding FROM memory_embeddings WHERE memory_id = ?",
        (memory_id,),
    ).fetchone()
    return bytes(row["embedding"]) if row else None


def _fetch_memory_content(db, memory_id: str) -> tuple[str | None, str | None]:
    """Return (type, content) or (None, None) if missing/archived."""
    row = db.conn.execute(
        "SELECT type, content FROM memories WHERE id = ? AND archived_at IS NULL",
        (memory_id,),
    ).fetchone()
    if row is None:
        return None, None
    return row["type"], row["content"]


def _write_links(db, pairs: Iterable[tuple[str, str, float, str]]) -> int:
    """Insert links, ignoring duplicates (UNIQUE constraint)."""
    n = 0
    cursor = db.conn.cursor()
    for source, target, sim, reason in pairs:
        if source == target:
            continue
        # ON CONFLICT DO UPDATE: refresh similarity if the same pair was
        # already linked under the same reason. This keeps the latest
        # estimate without duplicating rows.
        cursor.execute(
            """
            INSERT INTO memory_links (source_memory_id, target_memory_id, similarity, reason)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(source_memory_id, target_memory_id, reason) DO UPDATE SET
                similarity = excluded.similarity
            """,
            (source, target, float(sim), reason),
        )
        n += 1
    db.conn.commit()
    return n


# --------------------------------------------------------------------------
# Linkers
# --------------------------------------------------------------------------


_VALID_MODES = {"absolute", "topk", "adaptive", "zscore"}

# Pool size used for adaptive/zscore: how many neighbors we sample to estimate
# the local similarity distribution. 50 is a sane default (bigger than top_k by
# an order of magnitude, small enough that vec0 returns it in microseconds).
_ADAPTIVE_POOL_SIZE = 50


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile from a pre-sorted list. q in [0, 1].

    Mirrors numpy.percentile with method='linear' so callers get the same
    cutoff numpy would compute, without pulling numpy into engine code.
    """
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    q = max(0.0, min(1.0, q))
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def link_memory(
    db,
    memory_id: str,
    *,
    top_k: int = 5,
    threshold: float = 0.55,
    mode: str = "topk",
    reason: str = "semantic",
) -> list[Link]:
    """Create bidirectional `semantic` links to relevant nearest memories.

    Modes
    -----
    - ``"absolute"`` — legacy behavior. Take the top_k nearest neighbors that
      pass ``similarity >= threshold``. With dense embeddings (e.g.
      all-MiniLM-L6-v2) where the corpus mean similarity sits around 0.5–0.7,
      a fixed threshold of 0.55 lets through nearly everything, producing
      ~corpus_size links per memory. Kept for backward compat.
    - ``"topk"`` (default) — always take the top_k nearest neighbors. The
      threshold is ignored. This is the right default: we don't pretend to
      know what "similar enough" means in absolute terms; we just guarantee
      a small, bounded fan-out per memory.
    - ``"adaptive"`` — sample the top _ADAPTIVE_POOL_SIZE neighbors, compute
      the percentile defined by ``threshold`` over that local distribution,
      and keep up to top_k neighbors above that cutoff. Example: with
      ``threshold=0.95`` you get the top 5% of *this* memory's neighborhood,
      capped at ``top_k``. Robust to corpora with very different similarity
      ranges.
    - ``"zscore"`` — sample the top _ADAPTIVE_POOL_SIZE neighbors, compute
      mean and stdev of similarities, and keep neighbors whose similarity is
      above ``mean + threshold * stdev``. Default ``threshold=1.5`` (roughly
      the top decile under a normal distribution). Useful when the local
      neighborhood has a clear "stand-out cluster" tail.

    Returns the list of links written (one entry per pair, regardless of
    direction — bidirectionality is an internal storage detail).
    """
    ensure_schema(db.conn)

    if mode not in _VALID_MODES:
        raise ValueError(
            f"link_memory: unknown mode={mode!r}; expected one of {sorted(_VALID_MODES)}"
        )

    # Verify the source memory exists and is active.
    src_type, _ = _fetch_memory_content(db, memory_id)
    if src_type is None:
        log.debug("link_memory: source %s missing/archived, skipping", memory_id[:16])
        return []

    # Get the embedding. If no stored embedding exists, fall back to encoding
    # the content. This keeps the function callable from contexts where the
    # embedding pipeline hasn't run yet (e.g. unit tests).
    blob = _fetch_stored_embedding(db, memory_id)
    if blob is None:
        # Fall back to live encode (loads sentence-transformers — slow first time).
        try:
            row = db.conn.execute(
                "SELECT content FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
            if row is None:
                return []
            vec = emb.embed_text(row["content"])
            blob = emb._pack(vec)
        except emb.EmbeddingsUnavailable:
            log.debug("link_memory: no embedding and no embedder; skipping")
            return []

    # Make sure vec0 is loaded (no-op if already done).
    try:
        emb._require_vec(db)
    except emb.EmbeddingsUnavailable:
        log.warning("link_memory: sqlite-vec unavailable, skipping")
        return []

    # For adaptive/zscore we need a wider sample so we can estimate the local
    # similarity distribution. For absolute/topk we only need top_k+1.
    if mode in ("adaptive", "zscore"):
        fetch_k = max(_ADAPTIVE_POOL_SIZE, top_k + 1)
    else:
        fetch_k = top_k + 1

    # vec0 KNN. We over-fetch so we can drop the source itself reliably and
    # (in adaptive modes) compute distribution stats.
    rows = db.conn.execute(
        """
        SELECT v.memory_id AS mid, v.distance, m.archived_at
        FROM vec_memories v
        JOIN memories m ON m.id = v.memory_id
        WHERE v.embedding MATCH ?
          AND m.archived_at IS NULL
          AND k = ?
        ORDER BY v.distance
        """,
        (blob, fetch_k),
    ).fetchall()

    # Build (target, sim) candidates excluding self.
    candidates: list[tuple[str, float]] = []
    for r in rows:
        target = r["mid"]
        if target == memory_id:
            continue
        sim = _distance_to_similarity(float(r["distance"]))
        candidates.append((target, sim))

    # Determine the dynamic cutoff based on `mode`.
    cutoff: float
    if mode == "absolute":
        cutoff = threshold
    elif mode == "topk":
        # No similarity gate; we'll just slice top_k.
        cutoff = -1.0
    elif mode == "adaptive":
        sims_sorted = sorted((s for _, s in candidates))
        # `threshold` is a percentile in [0, 1]. e.g. 0.95 -> top 5%.
        cutoff = _percentile(sims_sorted, threshold) if sims_sorted else 0.0
    elif mode == "zscore":
        sims = [s for _, s in candidates]
        if len(sims) >= 2:
            mean = sum(sims) / len(sims)
            var = sum((s - mean) ** 2 for s in sims) / (len(sims) - 1)
            stdev = math.sqrt(var)
            cutoff = mean + threshold * stdev
        else:
            cutoff = sims[0] if sims else 0.0
    else:  # pragma: no cover -- guarded above
        cutoff = threshold

    pairs: list[tuple[str, str, float, str]] = []
    accepted: list[Link] = []
    for target, sim in candidates:
        if sim < cutoff:
            continue
        pairs.append((memory_id, target, sim, reason))
        pairs.append((target, memory_id, sim, reason))
        accepted.append(Link(memory_id, target, round(sim, 4), reason))
        if len(accepted) >= top_k:
            break

    _write_links(db, pairs)
    if accepted:
        log.debug(
            "link_memory: %s -> %d neighbors (mode=%s, reason=%s, cutoff=%.3f)",
            memory_id[:16], len(accepted), mode, reason, cutoff,
        )
    return accepted


def link_by_shared_entities(db, memory_id: str) -> list[Link]:
    """Link the memory to peers that share at least one entity in `memory_entities`.

    Similarity is set to the Jaccard overlap of the two memories' entity
    sets — gives a meaningful weight without needing embeddings.
    """
    ensure_schema(db.conn)

    # Collect this memory's entities. Bail out fast if it has none.
    src_entities = {
        r["entity_id"]
        for r in db.conn.execute(
            "SELECT entity_id FROM memory_entities WHERE memory_id = ?",
            (memory_id,),
        ).fetchall()
    }
    if not src_entities:
        return []

    placeholders = ",".join("?" * len(src_entities))
    # Find every other memory that touches any of those entities.
    rows = db.conn.execute(
        f"""
        SELECT DISTINCT me.memory_id AS mid
        FROM memory_entities me
        JOIN memories m ON m.id = me.memory_id
        WHERE me.entity_id IN ({placeholders})
          AND me.memory_id != ?
          AND m.archived_at IS NULL
        """,
        (*src_entities, memory_id),
    ).fetchall()

    pairs: list[tuple[str, str, float, str]] = []
    accepted: list[Link] = []
    for r in rows:
        target = r["mid"]
        target_entities = {
            x["entity_id"]
            for x in db.conn.execute(
                "SELECT entity_id FROM memory_entities WHERE memory_id = ?",
                (target,),
            ).fetchall()
        }
        if not target_entities:
            continue
        union = src_entities | target_entities
        inter = src_entities & target_entities
        # Guard against zero-division (impossible given checks above, but
        # cheap insurance).
        jaccard = (len(inter) / len(union)) if union else 0.0
        pairs.append((memory_id, target, jaccard, "shared_entity"))
        pairs.append((target, memory_id, jaccard, "shared_entity"))
        accepted.append(Link(memory_id, target, round(jaccard, 4), "shared_entity"))

    _write_links(db, pairs)
    return accepted


# --------------------------------------------------------------------------
# Traversal
# --------------------------------------------------------------------------


def get_neighbors(
    db,
    memory_id: str,
    *,
    max_depth: int = 1,
    min_similarity: float = 0.5,
    reasons: tuple[str, ...] | None = None,
) -> list[MemoryRef]:
    """Return memories reachable from `memory_id` within ``max_depth`` hops.

    Traversal is breadth-first via a recursive CTE:
      depth 1 = direct neighbors,
      depth 2 = neighbors of neighbors, etc.

    Each returned memory is the *closest* path found (smallest depth), with
    the similarity and reason of the edge that introduced it. Edges with
    ``similarity < min_similarity`` are pruned at every hop. ``reasons`` lets
    callers restrict to a subset of edge kinds (e.g. only ``semantic``).
    """
    ensure_schema(db.conn)

    if max_depth < 1:
        return []

    reason_filter = ""
    params: list = [memory_id]
    if reasons:
        placeholders = ",".join("?" * len(reasons))
        reason_filter = f"AND ml.reason IN ({placeholders})"
        params.extend(reasons)
    params.append(min_similarity)
    params.append(max_depth)

    # Recursive CTE walks the graph. We prune by depth + similarity at each
    # step. The DISTINCT in the outer SELECT plus the MIN(depth) GROUP BY
    # collapses multiple paths to the same memory into the shortest one.
    sql = f"""
    WITH RECURSIVE walk(memory_id, depth, similarity, reason, path) AS (
        SELECT ml.target_memory_id, 1, ml.similarity, ml.reason,
               ',' || ? || ',' || ml.target_memory_id || ','
        FROM memory_links ml
        WHERE ml.source_memory_id = ?
          {reason_filter}
          AND ml.similarity >= ?
        UNION ALL
        SELECT ml.target_memory_id, w.depth + 1, ml.similarity, ml.reason,
               w.path || ml.target_memory_id || ','
        FROM memory_links ml
        JOIN walk w ON w.memory_id = ml.source_memory_id
        WHERE w.depth < ?
          AND ml.similarity >= ?
          {reason_filter}
          AND instr(w.path, ',' || ml.target_memory_id || ',') = 0
    )
    SELECT walk.memory_id, MIN(walk.depth) AS depth,
           MAX(walk.similarity) AS similarity, walk.reason,
           m.type, m.content
    FROM walk
    JOIN memories m ON m.id = walk.memory_id
    WHERE m.archived_at IS NULL
      AND walk.memory_id != ?
    GROUP BY walk.memory_id
    ORDER BY depth, similarity DESC
    """
    # The recursive CTE needs `min_similarity` and `reason_filter` repeated for
    # the recursive arm. We rebuild the parameter sequence explicitly here.
    cte_params: list = [
        memory_id,            # ',' || ? || ',' (path seed in anchor SELECT)
        memory_id,            # WHERE ml.source_memory_id = ?
    ]
    if reasons:
        cte_params.extend(reasons)        # anchor reason filter
    cte_params.append(min_similarity)     # anchor similarity gate
    cte_params.append(max_depth)          # recursive depth gate
    cte_params.append(min_similarity)     # recursive similarity gate
    if reasons:
        cte_params.extend(reasons)        # recursive reason filter
    cte_params.append(memory_id)          # outer WHERE filter (exclude self)

    rows = db.conn.execute(sql, cte_params).fetchall()
    return [
        MemoryRef(
            memory_id=r["memory_id"],
            type=r["type"],
            content=r["content"],
            depth=int(r["depth"]),
            similarity=round(float(r["similarity"]), 4),
            reason=r["reason"] or "",
        )
        for r in rows
    ]


# --------------------------------------------------------------------------
# Backfill
# --------------------------------------------------------------------------


def recompute_links(
    db,
    *,
    batch_size: int = 100,
    top_k: int = 5,
    threshold: float = 0.55,
    mode: str = "topk",
    include_shared_entities: bool = True,
) -> dict:
    """Rebuild semantic links across the corpus in batches.

    Streams active memories ordered by id, calls :func:`link_memory` for
    each. Returns a summary of how many memories were processed and how many
    links were written. Idempotent — the UNIQUE constraint absorbs repeats.

    ``mode`` is forwarded to :func:`link_memory`. Default is ``"topk"`` —
    see that function's docstring for the trade-offs.
    """
    ensure_schema(db.conn)

    total_links = 0
    total_processed = 0
    last_id = ""
    while True:
        rows = db.conn.execute(
            """
            SELECT id FROM memories
            WHERE archived_at IS NULL AND id > ?
            ORDER BY id
            LIMIT ?
            """,
            (last_id, batch_size),
        ).fetchall()
        if not rows:
            break
        for r in rows:
            mid = r["id"]
            try:
                links = link_memory(
                    db, mid, top_k=top_k, threshold=threshold, mode=mode,
                )
                total_links += len(links)
                if include_shared_entities:
                    links2 = link_by_shared_entities(db, mid)
                    total_links += len(links2)
                total_processed += 1
            except Exception:
                log.exception("recompute_links: failed for memory %s", mid[:16])
            last_id = mid
    return {"processed": total_processed, "links_written": total_links}


# --------------------------------------------------------------------------
# Pruning + stats (defensive backfill for over-linked corpora)
# --------------------------------------------------------------------------


def prune_excess_links(
    db,
    *,
    max_per_memory: int = 10,
    min_similarity: float | None = None,
    reason: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Trim ``memory_links`` so each source has at most ``max_per_memory`` rows.

    Strategy:
    - For each ``source_memory_id`` (optionally restricted by ``reason``),
      keep the rows with the **highest** ``similarity`` and delete the rest
      until the source has at most ``max_per_memory`` outgoing links.
    - If ``min_similarity`` is set, also delete every row below that floor
      (regardless of per-source counts).
    - With ``dry_run=True``, no rows are deleted; the count of rows that
      *would* be deleted is returned in ``would_delete``.

    Returns a dict with ``{"deleted": N, "would_delete": N, "scanned": M}``.
    Note: this only operates on the side of an edge stored under the given
    ``source_memory_id``. Bidirectional links are stored as two rows (A→B
    and B→A) — to keep the graph symmetric you typically want to either:
      - prune both directions (default: pass ``reason=None``), or
      - run this separately per direction.
    Because each row carries its own ``similarity`` and they are equal at
    write time, the two directions usually survive or perish together.
    """
    ensure_schema(db.conn)

    where_extras: list[str] = []
    params: list = []
    if reason is not None:
        where_extras.append("reason = ?")
        params.append(reason)

    where_clause = ""
    if where_extras:
        where_clause = "WHERE " + " AND ".join(where_extras)

    scanned = db.conn.execute(
        f"SELECT COUNT(*) AS c FROM memory_links {where_clause}", params,
    ).fetchone()["c"]

    # Collect ids to delete using a window function: rank rows per source by
    # similarity DESC, mark those past `max_per_memory` for deletion.
    rank_sql = f"""
    SELECT id FROM (
        SELECT id,
               ROW_NUMBER() OVER (
                   PARTITION BY source_memory_id
                   ORDER BY similarity DESC, id ASC
               ) AS rk
        FROM memory_links
        {where_clause}
    )
    WHERE rk > ?
    """
    excess_ids = [
        row["id"]
        for row in db.conn.execute(rank_sql, (*params, max_per_memory)).fetchall()
    ]

    floor_ids: list[int] = []
    if min_similarity is not None:
        floor_sql = f"SELECT id FROM memory_links {where_clause}"
        if where_extras:
            floor_sql += " AND similarity < ?"
        else:
            floor_sql += " WHERE similarity < ?"
        floor_ids = [
            row["id"]
            for row in db.conn.execute(
                floor_sql, (*params, min_similarity)
            ).fetchall()
        ]

    to_delete = set(excess_ids) | set(floor_ids)

    if dry_run:
        return {
            "scanned": int(scanned),
            "deleted": 0,
            "would_delete": len(to_delete),
            "max_per_memory": max_per_memory,
            "min_similarity": min_similarity,
            "reason": reason,
            "dry_run": True,
        }

    if to_delete:
        # Chunk DELETE to keep parameter count under SQLite's limit (~999).
        ids = list(to_delete)
        cur = db.conn.cursor()
        chunk = 500
        for i in range(0, len(ids), chunk):
            batch = ids[i : i + chunk]
            placeholders = ",".join("?" * len(batch))
            cur.execute(
                f"DELETE FROM memory_links WHERE id IN ({placeholders})",
                batch,
            )
        db.conn.commit()

    return {
        "scanned": int(scanned),
        "deleted": len(to_delete),
        "would_delete": len(to_delete),
        "max_per_memory": max_per_memory,
        "min_similarity": min_similarity,
        "reason": reason,
        "dry_run": False,
    }


def link_stats(db) -> dict:
    """Summarize the ``memory_links`` table.

    Returns:
      - ``total``: total link rows.
      - ``distinct_sources``: number of distinct ``source_memory_id`` values.
      - ``per_source``: ``{"max", "avg", "p95"}`` of outgoing links per source.
      - ``similarity_histogram``: list of 10 bucket counts for sim ∈ [0, 1].
        Bucket i covers ``[i/10, (i+1)/10)`` (last bucket is closed on both ends).
      - ``by_reason``: ``{reason: count}`` aggregated.
    """
    ensure_schema(db.conn)

    total = db.conn.execute("SELECT COUNT(*) AS c FROM memory_links").fetchone()["c"]
    distinct_sources = db.conn.execute(
        "SELECT COUNT(DISTINCT source_memory_id) AS c FROM memory_links"
    ).fetchone()["c"]

    # Per-source counts.
    per_source_rows = db.conn.execute(
        "SELECT source_memory_id, COUNT(*) AS c FROM memory_links GROUP BY source_memory_id"
    ).fetchall()
    counts = sorted(int(r["c"]) for r in per_source_rows)
    if counts:
        max_per = counts[-1]
        avg_per = sum(counts) / len(counts)
        p95 = _percentile([float(c) for c in counts], 0.95)
    else:
        max_per = 0
        avg_per = 0.0
        p95 = 0.0

    # Histogram (10 equal-width buckets between 0 and 1).
    buckets = [0] * 10
    for r in db.conn.execute("SELECT similarity FROM memory_links").fetchall():
        sim = float(r["similarity"])
        sim = max(0.0, min(1.0, sim))
        idx = int(sim * 10)
        if idx >= 10:
            idx = 9  # collapse 1.0 into the last bucket
        buckets[idx] += 1

    by_reason = {}
    for r in db.conn.execute(
        "SELECT COALESCE(reason, '<none>') AS reason, COUNT(*) AS c "
        "FROM memory_links GROUP BY reason"
    ).fetchall():
        by_reason[r["reason"]] = int(r["c"])

    return {
        "total": int(total),
        "distinct_sources": int(distinct_sources),
        "per_source": {
            "max": int(max_per),
            "avg": round(float(avg_per), 2),
            "p95": round(float(p95), 2),
        },
        "similarity_histogram": buckets,
        "by_reason": by_reason,
    }
