"""Personalized PageRank multi-hop retrieval (HippoRAG 2-inspired).

Implements GAP P1-2: walk the joint memory↔entity↔memory_link graph from
seed entities (extracted from the user query) and rank memories by the
stationary distribution of a Personalized PageRank random walker that
teleports back to the seeds with probability ``1-alpha``.

Why PPR
-------
- Traditional dense / BM25 retrieval ranks each document independently.
  Multi-hop questions ("which decision linked memoirs to HippoRAG?") need
  *evidence chaining* — bridging multiple memories via shared entities.
- PPR gives every node a score proportional to the expected fraction of
  steps a random walker spends there, biased toward the seeds. Nodes that
  are reachable from many seeds via short, high-weight paths float to the
  top — exactly the multi-hop signal we want.

Graph topology
--------------
Nodes:
  * ``mem:<id>``  — every active memory
  * ``ent:<id>``  — every entity referenced by ≥1 active memory

Edges (undirected, normalized to a column-stochastic transition matrix):
  * ``memory ↔ entity``  — weight 1.0 (from ``memory_entities``)
  * ``entity ↔ entity``  — weight 1.0 (from ``relationships``)
  * ``memory ↔ memory``  — weight = ``memory_links.similarity``
                          (from A-MEM Zettelkasten, P1-3)

Public API
----------
- :func:`extract_seed_entities` — query → list of entity ids
- :func:`build_graph`           — DB → cached :class:`GraphView`
- :func:`personalized_pagerank` — pure-Python iterative PPR
- :func:`graph_search`          — full pipeline (seeds + PPR + memory filter)
- :func:`hybrid_graph_search`   — RRF-fused with hybrid_search

Caching
-------
``build_graph`` keeps a process-local TTL cache keyed by the DB path. The
cache invalidates when (a) ``MEMOIRS_GRAPH_TTL`` seconds have passed
(default 300 = 5 min) **or** (b) the active-memory count changes between
calls — which catches the common ingest scenario without a full rebuild
on every query.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable

from ..db import MemoirsDB, utc_now


log = logging.getLogger("memoirs.graph_retrieval")


# ---------------------------------------------------------------------------
# Constants & tuning knobs
# ---------------------------------------------------------------------------

# Default damping factor (HippoRAG 2 uses 0.5 — heavier teleport keeps the
# walker close to the seeds, which is the right behavior for tightly-scoped
# multi-hop queries on a small graph).
DEFAULT_ALPHA = 0.5

# Iterative solver settings. 20 iterations is enough to converge well below
# 1e-4 on graphs we realistically build (≤ ~50k nodes); larger graphs hit
# the early-exit on ``tol`` before reaching ``max_iter``.
DEFAULT_MAX_ITER = 20
DEFAULT_TOL = 1e-4

# Cache TTL — env-overridable for tests / long-running daemons.
def _cache_ttl_seconds() -> int:
    raw = os.environ.get("MEMOIRS_GRAPH_TTL", "300").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 300


# ---------------------------------------------------------------------------
# Graph data model
# ---------------------------------------------------------------------------


# Node ids carry their kind as a prefix so the same string namespace can hold
# both memories and entities without collisions, and downstream filters can
# project to just one kind in O(deg).
_MEM_PREFIX = "mem:"
_ENT_PREFIX = "ent:"


def _mem_node(memory_id: str) -> str:
    return _MEM_PREFIX + memory_id


def _ent_node(entity_id: str) -> str:
    return _ENT_PREFIX + entity_id


def _is_memory_node(node: str) -> bool:
    return node.startswith(_MEM_PREFIX)


def _strip_prefix(node: str) -> str:
    if node.startswith(_MEM_PREFIX):
        return node[len(_MEM_PREFIX):]
    if node.startswith(_ENT_PREFIX):
        return node[len(_ENT_PREFIX):]
    return node


@dataclass
class GraphView:
    """Frozen, in-memory view of the joint memory/entity graph.

    ``adjacency[u]`` maps each neighbor ``v`` to the **outgoing** edge weight
    (already row-normalized so each row sums to 1.0 — the PPR loop can multiply
    directly without re-normalizing on every iteration).

    ``raw_adjacency[u]`` keeps the un-normalized weights — useful for tests
    and sanity checks without re-walking the DB.
    """

    adjacency: dict[str, dict[str, float]]
    raw_adjacency: dict[str, dict[str, float]]
    memory_count: int
    entity_count: int
    built_at: float = field(default_factory=time.time)

    @property
    def num_nodes(self) -> int:
        return len(self.adjacency)

    def memory_nodes(self) -> list[str]:
        return [n for n in self.adjacency if _is_memory_node(n)]


def _row_normalize(adj: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Convert raw weighted adjacency to a row-stochastic transition matrix.

    Dangling nodes (no outgoing edges) keep an empty row — the PPR loop
    handles them by redistributing their mass to the teleport vector,
    matching the standard "personalization restart" formulation.
    """
    out: dict[str, dict[str, float]] = {}
    for u, neighbors in adj.items():
        total = sum(neighbors.values())
        if total <= 0:
            out[u] = {}
            continue
        out[u] = {v: w / total for v, w in neighbors.items()}
    return out


# ---------------------------------------------------------------------------
# Graph build (with TTL cache)
# ---------------------------------------------------------------------------


# Process-local cache. Keyed by the resolved DB path so multiple DBs in the
# same process don't clobber each other.
_GRAPH_CACHE: dict[str, tuple[GraphView, int, float]] = {}
_CACHE_LOCK = threading.Lock()


def _db_cache_key(db: MemoirsDB) -> str:
    try:
        return str(db.path.resolve())
    except Exception:
        return str(db.path)


def _active_memory_count(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM memories WHERE archived_at IS NULL"
    ).fetchone()
    return int(row["c"]) if row is not None else 0


def invalidate_graph_cache(db: MemoirsDB | None = None) -> None:
    """Drop the cached graph view (for tests, manual refresh).

    With ``db=None`` the entire cache is cleared.
    """
    with _CACHE_LOCK:
        if db is None:
            _GRAPH_CACHE.clear()
        else:
            _GRAPH_CACHE.pop(_db_cache_key(db), None)


def build_graph(db: MemoirsDB, *, force_rebuild: bool = False) -> GraphView:
    """Build (or fetch from cache) the joint memory↔entity graph.

    The graph is cached per-DB-path with a TTL governed by
    ``MEMOIRS_GRAPH_TTL`` (default 300s). The cache is also invalidated
    whenever the active memory count changes — that catches the common
    "ingested some events, then queried" pattern without an explicit
    refresh call.

    Pass ``force_rebuild=True`` to bypass the cache (used by tests and the
    ``MEMOIRS_GRAPH_TTL=0`` debug mode).
    """
    key = _db_cache_key(db)
    ttl = _cache_ttl_seconds()
    now = time.time()
    current_mem_count = _active_memory_count(db.conn)

    if not force_rebuild:
        with _CACHE_LOCK:
            cached = _GRAPH_CACHE.get(key)
            if cached is not None:
                view, cached_count, built_at = cached
                fresh = (now - built_at) < ttl
                if fresh and cached_count == current_mem_count:
                    return view

    view = _build_graph_uncached(db)
    with _CACHE_LOCK:
        _GRAPH_CACHE[key] = (view, current_mem_count, now)
    return view


def _build_graph_uncached(db: MemoirsDB) -> GraphView:
    """Materialize the in-memory adjacency from the SQL tables.

    Three SELECTs, one pass each: ``memory_entities``, ``relationships``,
    ``memory_links``. We only include active memories — archived rows would
    poison the random walk with stale evidence.

    Edge accumulation rules:
      * memory↔entity: weight = 1.0 (presence is binary)
      * entity↔entity: weight = ``confidence`` from ``relationships``
                       (capped at 1.0; defaults to 1.0 if missing)
      * memory↔memory: weight = ``memory_links.similarity``; if multiple
                       links exist between the same pair (semantic + shared_entity)
                       we take the MAX — already-strong edges shouldn't be
                       penalized for having multiple justifications.
    """
    raw: dict[str, dict[str, float]] = {}

    def _add_edge(u: str, v: str, w: float) -> None:
        if u == v or w <= 0:
            return
        raw.setdefault(u, {})
        raw[u][v] = max(raw[u].get(v, 0.0), w)
        # Symmetric — PPR is undirected by construction.
        raw.setdefault(v, {})
        raw[v][u] = max(raw[v].get(u, 0.0), w)

    # 1. memory ↔ entity (only active memories) ------------------------------
    rows = db.conn.execute(
        """
        SELECT me.memory_id AS memory_id, me.entity_id AS entity_id
        FROM memory_entities me
        JOIN memories m ON m.id = me.memory_id
        WHERE m.archived_at IS NULL
        """
    ).fetchall()
    memory_ids: set[str] = set()
    entity_ids: set[str] = set()
    for r in rows:
        m, e = r["memory_id"], r["entity_id"]
        memory_ids.add(m)
        entity_ids.add(e)
        _add_edge(_mem_node(m), _ent_node(e), 1.0)

    # 2. entity ↔ entity ------------------------------------------------------
    rel_rows = db.conn.execute(
        "SELECT source_entity_id AS s, target_entity_id AS t, confidence FROM relationships"
    ).fetchall()
    for r in rel_rows:
        s, t = r["s"], r["t"]
        if s == t:
            continue
        try:
            w = float(r["confidence"]) if r["confidence"] is not None else 1.0
        except (TypeError, ValueError):
            w = 1.0
        # Clamp to (0, 1] — guard against bad data.
        w = max(0.0, min(1.0, w))
        if w <= 0.0:
            continue
        entity_ids.add(s)
        entity_ids.add(t)
        _add_edge(_ent_node(s), _ent_node(t), w)

    # 3. memory ↔ memory (Zettelkasten links) --------------------------------
    # ``memory_links`` may not exist on very old DBs — guard with a soft check.
    has_links = db.conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_links'"
    ).fetchone()
    if has_links:
        link_rows = db.conn.execute(
            """
            SELECT ml.source_memory_id AS s, ml.target_memory_id AS t,
                   ml.similarity AS sim
            FROM memory_links ml
            JOIN memories ms ON ms.id = ml.source_memory_id
            JOIN memories mt ON mt.id = ml.target_memory_id
            WHERE ms.archived_at IS NULL AND mt.archived_at IS NULL
            """
        ).fetchall()
        for r in link_rows:
            s, t = r["s"], r["t"]
            if s == t:
                continue
            try:
                w = float(r["sim"])
            except (TypeError, ValueError):
                continue
            if w <= 0.0:
                continue
            memory_ids.add(s)
            memory_ids.add(t)
            _add_edge(_mem_node(s), _mem_node(t), w)

    # Make sure isolated memories that have entity links still appear as nodes
    # in the adjacency map (they already do via _add_edge above, but harmless
    # to keep the invariant explicit).
    for m in memory_ids:
        raw.setdefault(_mem_node(m), {})
    for e in entity_ids:
        raw.setdefault(_ent_node(e), {})

    normalized = _row_normalize(raw)
    return GraphView(
        adjacency=normalized,
        raw_adjacency=raw,
        memory_count=len(memory_ids),
        entity_count=len(entity_ids),
    )


# ---------------------------------------------------------------------------
# Seed extraction (query → entity ids)
# ---------------------------------------------------------------------------


# Word-grabbing pattern: unicode-friendly, hyphen-aware. We keep it cheap
# and language-agnostic — the heavy lifting is the LIKE match against
# the entities table, which already stores normalized names.
_WORD_RE = re.compile(r"\b[\w\-]+\b", re.UNICODE)


def _query_tokens(query: str) -> list[str]:
    """Lowercased word tokens for fuzzy entity matching."""
    if not query:
        return []
    return [t.lower() for t in _WORD_RE.findall(query) if len(t) >= 2]


def _spacy_entity_strings(query: str) -> list[str]:
    """Best-effort spaCy NER over the query. Returns lowercased entity texts.

    Silently returns ``[]`` when spaCy is not installed or no model is
    available — we don't want NER's optional-dep status to break retrieval.
    """
    try:
        from . import extract_spacy as _sp  # local import; spaCy is optional
    except Exception:
        return []
    if not _sp.is_available():
        return []
    try:
        nlp = _sp._pick_pipeline(query)
    except Exception:
        return []
    if nlp is None:
        return []
    try:
        doc = nlp(query)
    except Exception:
        return []
    return [ent.text.strip().lower() for ent in doc.ents if ent.text.strip()]


def extract_seed_entities(db: MemoirsDB, query: str) -> list[str]:
    """Map a free-form query to entity ids in the ``entities`` table.

    Strategy:
      1. Run spaCy NER on the query if a model is available — use those
         spans as the candidate strings.
      2. Otherwise (or as a supplement) tokenize the query and keep
         non-trivial words (≥3 chars).
      3. Match each candidate against ``entities.normalized_name`` using a
         case-insensitive substring search (``LIKE %candidate%``). The
         normalized-name index keeps the lookup fast even at 10k+ entities.
      4. De-duplicate while preserving first-seen order.

    Returns entity ids; an empty list signals "graph search not applicable".
    """
    if not query or not query.strip():
        return []

    # Build the candidate list: NER spans first (more precise), then tokens.
    candidates: list[str] = []
    seen_cand: set[str] = set()
    for span in _spacy_entity_strings(query):
        s = span.strip().lower()
        if s and s not in seen_cand:
            candidates.append(s)
            seen_cand.add(s)
    for tok in _query_tokens(query):
        if len(tok) < 3:
            continue
        if tok in seen_cand:
            continue
        candidates.append(tok)
        seen_cand.add(tok)

    if not candidates:
        return []

    # Match candidates against entities.normalized_name. We prefer the
    # normalized column because it lowercases + strips diacritics at insert
    # time (see ``ingesters/entities.py`` historically).
    seen_ids: set[str] = set()
    out: list[str] = []
    for cand in candidates:
        # `LIKE` is case-insensitive in SQLite by default for ASCII text
        # via the COLLATE NOCASE on the column-less compare, but we lower()
        # both sides explicitly to keep behavior consistent across locales.
        pattern = f"%{cand}%"
        rows = db.conn.execute(
            """
            SELECT id FROM entities
            WHERE LOWER(normalized_name) LIKE ?
               OR LOWER(name) LIKE ?
            LIMIT 25
            """,
            (pattern, pattern),
        ).fetchall()
        for r in rows:
            eid = r["id"]
            if eid in seen_ids:
                continue
            seen_ids.add(eid)
            out.append(eid)
    return out


# ---------------------------------------------------------------------------
# Personalized PageRank
# ---------------------------------------------------------------------------


def personalized_pagerank(
    graph: GraphView,
    seeds: dict[str, float],
    *,
    alpha: float = DEFAULT_ALPHA,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
) -> dict[str, float]:
    """Iterative Personalized PageRank.

    Update rule (column-stochastic, with restart):

        r ← alpha · Mᵀ · r + (1 − alpha) · p

    where ``M`` is the row-normalized adjacency, ``r`` is the rank vector,
    and ``p`` is the personalization (teleport) vector built from ``seeds``.
    Dangling-node mass redistributes to ``p`` to preserve the L1 norm.

    Convergence: the L1 difference between successive ``r`` is checked each
    iteration; we early-exit at ``tol``. ``max_iter`` is a hard ceiling.

    Falls back to ``networkx.pagerank(personalization=...)`` when networkx is
    importable — it's typically 2-3× faster on graphs of a few thousand
    nodes and we don't need to re-implement what's already mature.
    """
    if not graph.adjacency:
        return {}
    if not seeds:
        return {}

    # Normalize the personalization vector. Negative or zero weights are
    # dropped so callers can't accidentally craft a zero-norm vector.
    seed_total = sum(max(0.0, w) for w in seeds.values())
    if seed_total <= 0.0:
        return {}

    # Restrict the teleport set to seeds that actually exist in the graph —
    # an unknown seed would silently bleed mass forever otherwise.
    p: dict[str, float] = {
        node: max(0.0, w) / seed_total
        for node, w in seeds.items()
        if node in graph.adjacency and w > 0.0
    }
    p_total = sum(p.values())
    if p_total <= 0.0:
        return {}
    if p_total < 0.999:
        # Re-normalize after dropping unknown seeds.
        p = {n: w / p_total for n, w in p.items()}

    # Try networkx first (drop-in, well-tested implementation). We pass the
    # raw (un-normalized) adjacency so networkx's own normalization runs.
    try:
        import networkx as nx  # type: ignore
    except ImportError:
        nx = None

    if nx is not None:
        try:
            G = nx.DiGraph()
            for u, neighbors in graph.raw_adjacency.items():
                if not neighbors:
                    G.add_node(u)
                    continue
                for v, w in neighbors.items():
                    G.add_edge(u, v, weight=w)
            # networkx uses (1-alpha) as teleport probability, same as us.
            ranks = nx.pagerank(
                G,
                alpha=alpha,
                personalization=p,
                max_iter=max_iter,
                tol=tol,
                weight="weight",
                nstart=p,  # warm-start at the seeds → fewer iterations
            )
            return dict(ranks)
        except Exception as exc:  # pragma: no cover - rare, falls through to pure-py
            log.debug("networkx PPR failed (%s) — using pure-Python fallback", exc)

    # Pure-Python iterative solver -------------------------------------------------
    nodes = list(graph.adjacency.keys())
    # Initial distribution = personalization vector (warm start).
    r: dict[str, float] = {n: p.get(n, 0.0) for n in nodes}

    # Pre-bind for speed in the inner loop.
    adj = graph.adjacency

    for _ in range(max_iter):
        # Aggregate dangling mass — nodes with no outgoing edges.
        dangling = 0.0
        for u in nodes:
            if not adj[u]:
                dangling += r[u]

        # Compute new rank vector.
        new_r: dict[str, float] = {n: 0.0 for n in nodes}
        for u, neighbors in adj.items():
            if not neighbors:
                continue
            ru = r[u]
            if ru == 0.0:
                continue
            for v, w in neighbors.items():
                # `w` is already normalized so neighbors sum to 1 per row.
                new_r[v] += alpha * ru * w

        # Add the teleport (and dangling redistribution) over the seeds.
        teleport_mass = (1.0 - alpha) + alpha * dangling
        for n, pn in p.items():
            new_r[n] += teleport_mass * pn

        # Convergence check (L1).
        diff = 0.0
        for n in nodes:
            diff += abs(new_r[n] - r[n])
        r = new_r
        if diff < tol:
            break

    return r


# ---------------------------------------------------------------------------
# graph_search — the public retrieval entry point
# ---------------------------------------------------------------------------


def graph_search(
    db: MemoirsDB,
    query: str,
    *,
    top_k: int = 10,
    alpha: float = DEFAULT_ALPHA,
    seed_weight: float = 1.0,
) -> list[tuple[str, float]]:
    """End-to-end PPR retrieval.

    Returns ``[(memory_id, ppr_score), ...]`` sorted descending by score.
    Only memory nodes are returned — entity scores are an internal
    intermediate.

    Workflow:
      1. ``extract_seed_entities`` from the query.
      2. Build (or fetch cached) :class:`GraphView`.
      3. Run :func:`personalized_pagerank` with seeds = those entity nodes,
         each weighted by ``seed_weight`` (default 1.0 → uniform).
      4. Project to memory nodes only and slice top-K.

    Empty seeds → ``[]`` (the caller should fall back to dense / hybrid).
    """
    seed_entity_ids = extract_seed_entities(db, query)
    if not seed_entity_ids:
        log.debug("graph_search: no seed entities matched for query %r", query)
        return []

    graph = build_graph(db)
    if graph.num_nodes == 0:
        return []

    # Build the personalization dict from the seed entity ids — but only
    # those that actually exist as nodes in the graph (an entity may have
    # zero linked active memories and zero relationships → not in graph).
    seeds: dict[str, float] = {}
    for eid in seed_entity_ids:
        node = _ent_node(eid)
        if node in graph.adjacency:
            seeds[node] = seed_weight
    if not seeds:
        log.debug("graph_search: seed entities %s are not present in the graph", seed_entity_ids[:3])
        return []

    ranks = personalized_pagerank(graph, seeds, alpha=alpha)
    if not ranks:
        return []

    # Project to memory nodes and exclude near-zero noise.
    mem_scores: list[tuple[str, float]] = []
    for node, score in ranks.items():
        if not _is_memory_node(node):
            continue
        if score <= 0.0:
            continue
        mem_scores.append((_strip_prefix(node), float(score)))
    mem_scores.sort(key=lambda kv: kv[1], reverse=True)
    return mem_scores[:top_k]


# ---------------------------------------------------------------------------
# hybrid_graph_search — fuse PPR with hybrid via RRF
# ---------------------------------------------------------------------------


def hybrid_graph_search(
    db: MemoirsDB,
    query: str,
    *,
    top_k: int = 10,
    alpha: float = DEFAULT_ALPHA,
    rrf_k: int = 60,
    as_of: str | None = None,
) -> list[tuple[str, float]]:
    """Combine the existing hybrid (BM25+dense) ranking with PPR via RRF.

    PPR contributes the multi-hop / structural signal; hybrid contributes
    the surface-level relevance. RRF makes them comparable without any
    score normalization.

    Returns ``[(memory_id, fused_score), ...]`` sorted descending.
    """
    # Local import: keeps cold-start overhead off the pure-graph path.
    from . import hybrid_retrieval as hr

    # We oversample each leg by 2× — RRF reorders, so giving it more
    # candidates helps recall without much cost.
    over_k = max(top_k * 2, 20)
    try:
        hybrid_results = hr.hybrid_search(db, query, top_k=over_k, as_of=as_of)
    except Exception as exc:
        log.warning("hybrid_graph_search: hybrid leg failed (%s) — graph-only", exc)
        hybrid_results = []
    hybrid_pairs = [(r["id"], float(r.get("score", 0.0))) for r in hybrid_results]

    graph_pairs = graph_search(db, query, top_k=over_k, alpha=alpha)

    if not hybrid_pairs and not graph_pairs:
        return []

    fused: dict[str, float] = {}
    for ranking in (hybrid_pairs, graph_pairs):
        for rank, (mid, _score) in enumerate(ranking, start=1):
            fused[mid] = fused.get(mid, 0.0) + 1.0 / (rrf_k + rank)
    out = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return out[:top_k]


# ---------------------------------------------------------------------------
# Hydration helper — convert (id, score) pairs into full memory rows
# ---------------------------------------------------------------------------


def hydrate_memories(
    db: MemoirsDB,
    pairs: Iterable[tuple[str, float]],
    *,
    as_of: str | None = None,
) -> list[dict]:
    """Fetch full memory rows for ``pairs`` while preserving order.

    Mirrors :func:`hybrid_retrieval.hydrate_memories` so the engine's
    downstream pipeline (conflict detection, scoring, compression) doesn't
    need to special-case the graph mode.
    """
    pair_list = list(pairs)
    if not pair_list:
        return []
    ids = [p[0] for p in pair_list]
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
    for mid, score in pair_list:
        row = by_id.get(mid)
        if row is None:
            continue
        # Surface PPR score in the slot the engine reads as a relevance
        # proxy (`similarity`). The original `score` column (curation score)
        # stays untouched.
        row["similarity"] = round(float(score), 6)
        row["graph_score"] = float(score)
        out.append(row)
    return out
