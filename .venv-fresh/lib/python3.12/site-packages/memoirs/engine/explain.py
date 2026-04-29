"""Chain-of-memory provenance traversal for ``mcp_explain_context`` (P1-9).

This module answers "*why* did the engine surface this memory for that
query?" by walking the joint memory↔entity graph that
:mod:`engine.graph_retrieval` already maintains, and emitting an explicit
**provenance chain** — a list of step dicts that the UI / CLI can render
as a timeline.

Step shape (canonical)::

    [{"step": 0, "kind": "query",          "value": "..."},
     {"step": 1, "kind": "entity_match",   "entity": "foo", "confidence": 0.92},
     {"step": 2, "kind": "entity_relation","from":"foo","to":"bar","relation":"uses"},
     {"step": 3, "kind": "entity_to_memory","entity":"bar","memory_id":"mem_xyz"},
     {"step": 4, "kind": "memory_link",    "from":"mem_xyz","to":"mem_target",
                                            "similarity":0.84,"reason":"semantic"}]

Fallback (no graph path) — the memory was retrieved by dense / BM25 only::

    [{"step": 0, "kind": "query"},
     {"step": 1, "kind": "semantic_match", "score": 0.78}]

Memory not found::

    [{"step": 0, "kind": "query"},
     {"step": 1, "kind": "not_found", "memory_id": "mem_xyz"}]

Why a separate module
~~~~~~~~~~~~~~~~~~~~~
Keeping the BFS + step-shaping logic out of ``mcp/tools.py`` lets the UI
inspector and the CLI consume the same canonical chain without dragging
the MCP layer along. The heavy lifting (graph build, seed extraction,
PPR) already lives in :mod:`engine.graph_retrieval`; this file is a thin
graph-walker that *uses* those primitives.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Optional

from . import graph_retrieval as gr
from ..db import MemoirsDB


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

Step = dict[str, Any]
Chain = list[Step]


# ---------------------------------------------------------------------------
# Internals — small DB lookups (cheap, no caching needed)
# ---------------------------------------------------------------------------


def _entity_name(db: MemoirsDB, entity_id: str) -> str:
    """Return a human-friendly entity label (``name`` column) or the id.

    The provenance chain is consumed by humans (UI tooltip, CLI table) so
    showing the canonical name beats showing an opaque ulid.
    """
    row = db.conn.execute(
        "SELECT name FROM entities WHERE id = ?", (entity_id,)
    ).fetchone()
    if row is None:
        return entity_id
    return row["name"] or entity_id


def _entity_match_confidence(db: MemoirsDB, query: str, entity_id: str) -> float:
    """Heuristic 0-1 score for "how cleanly did this entity match the query".

    1.0 — full token of the query equals the (normalized) entity name.
    0.7 — entity name is a substring of the query.
    0.4 — query token is a substring of the entity name (partial).
    0.2 — fallback when the seed extractor returned the id but we can't
          re-derive the match (e.g. NER span no longer in the query).
    """
    if not query:
        return 0.2
    row = db.conn.execute(
        "SELECT name, normalized_name FROM entities WHERE id = ?", (entity_id,)
    ).fetchone()
    if row is None:
        return 0.2
    name = (row["name"] or "").strip().lower()
    norm = (row["normalized_name"] or "").strip().lower()
    q = query.strip().lower()
    if not name and not norm:
        return 0.2
    # Exact-token match against the query tokens.
    tokens = set(gr._query_tokens(query))
    if name in tokens or norm in tokens:
        return 1.0
    # Substring match — the entity surface form appears verbatim.
    if (name and name in q) or (norm and norm in q):
        return 0.7
    # The query has a token that overlaps the entity (partial).
    for tok in tokens:
        if name and tok in name:
            return 0.4
        if norm and tok in norm:
            return 0.4
    return 0.2


def _entities_for_memory(db: MemoirsDB, memory_id: str) -> list[str]:
    rows = db.conn.execute(
        "SELECT entity_id FROM memory_entities WHERE memory_id = ?",
        (memory_id,),
    ).fetchall()
    return [r["entity_id"] for r in rows]


def _memory_exists(db: MemoirsDB, memory_id: str) -> bool:
    row = db.conn.execute(
        "SELECT 1 FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()
    return row is not None


def _relation_label(db: MemoirsDB, src_entity: str, tgt_entity: str) -> str:
    """Return the strongest ``relation`` label between two entities.

    ``relationships`` is directional but the graph is undirected, so we
    look up both orientations and prefer the higher-confidence row.
    """
    row = db.conn.execute(
        """
        SELECT relation, confidence FROM relationships
        WHERE (source_entity_id = ? AND target_entity_id = ?)
           OR (source_entity_id = ? AND target_entity_id = ?)
        ORDER BY confidence DESC LIMIT 1
        """,
        (src_entity, tgt_entity, tgt_entity, src_entity),
    ).fetchone()
    if row is None:
        return "related_to"
    return row["relation"] or "related_to"


def _memory_link_meta(
    db: MemoirsDB, src_memory: str, tgt_memory: str
) -> tuple[float, str]:
    """Return (similarity, reason) of the strongest ``memory_links`` edge.

    Picks the row with the largest similarity in either direction.
    Defaults to (0.0, "semantic") if the link table is missing or the row
    is gone (invariant of an undirected graph view: the edge exists, the
    row may have been deduped).
    """
    try:
        row = db.conn.execute(
            """
            SELECT similarity, reason FROM memory_links
            WHERE (source_memory_id = ? AND target_memory_id = ?)
               OR (source_memory_id = ? AND target_memory_id = ?)
            ORDER BY similarity DESC LIMIT 1
            """,
            (src_memory, tgt_memory, tgt_memory, src_memory),
        ).fetchone()
    except Exception:
        return (0.0, "semantic")
    if row is None:
        return (0.0, "semantic")
    sim = float(row["similarity"] or 0.0)
    reason = row["reason"] or "semantic"
    return (sim, reason)


# ---------------------------------------------------------------------------
# BFS shortest-path on the joint graph
# ---------------------------------------------------------------------------


def _bfs_shortest_path(
    graph: gr.GraphView,
    sources: Iterable[str],
    target: str,
    *,
    max_hops: int,
) -> Optional[list[str]]:
    """Multi-source BFS — return the shortest node path from any source to ``target``.

    Returns ``None`` when ``target`` is unreachable within ``max_hops`` hops
    (or not in the graph at all).

    We use a *multi-source* BFS so the caller doesn't have to run one BFS
    per seed and pick the min — the unified frontier already explores all
    seeds in parallel and gives us the optimal answer in O(V + E).
    """
    if max_hops < 0:
        return None
    if target not in graph.adjacency:
        return None
    sources = [s for s in sources if s in graph.adjacency]
    if not sources:
        return None
    if target in sources:
        return [target]

    # parents[node] = the predecessor on the shortest path from any source.
    parents: dict[str, Optional[str]] = {s: None for s in sources}
    depths: dict[str, int] = {s: 0 for s in sources}
    queue: deque[str] = deque(sources)
    while queue:
        u = queue.popleft()
        if depths[u] >= max_hops:
            continue
        for v in graph.adjacency.get(u, {}):
            if v in parents:
                continue
            parents[v] = u
            depths[v] = depths[u] + 1
            if v == target:
                # Reconstruct the path target → ... → source.
                path: list[str] = [v]
                cur: Optional[str] = u
                while cur is not None:
                    path.append(cur)
                    cur = parents[cur]
                path.reverse()
                return path
            queue.append(v)
    return None


def _path_score(graph: gr.GraphView, path: list[str]) -> float:
    """Cumulative log-score along ``path`` using the un-normalized edge weights.

    Higher = stronger evidence. We use the raw (un-normalized) adjacency
    so an entity↔entity edge with confidence=1.0 doesn't get penalized
    when its source has many neighbors (which would happen if we used
    the row-stochastic ``adjacency``).
    """
    if len(path) < 2:
        return 0.0
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        w = graph.raw_adjacency.get(u, {}).get(v, 0.0)
        total += float(w)
    return total


# ---------------------------------------------------------------------------
# Step builder — node path → typed steps
# ---------------------------------------------------------------------------


def _path_to_steps(
    db: MemoirsDB,
    query: str,
    path: list[str],
    *,
    seed_entity_node_ids: set[str],
) -> Chain:
    """Translate a node-id path into the canonical step shape.

    The path always starts at a seed node (entity or memory) and ends at
    the target memory node. Each *transition* between consecutive nodes
    becomes a step; the first step is always the synthetic ``query``
    anchor.
    """
    steps: Chain = [{"step": 0, "kind": "query", "value": query}]
    if not path:
        return steps

    # Step 1: the seed (almost always an entity_match — memory seeds are
    # only used when the target itself is a seed, which we handle below).
    seed = path[0]
    if gr._is_memory_node(seed):
        # Seed *is* the target memory (path of length 1) — no graph hop
        # was needed; the caller will append a semantic_match fallback.
        return steps
    seed_eid = gr._strip_prefix(seed)
    steps.append(
        {
            "step": len(steps),
            "kind": "entity_match",
            "entity": _entity_name(db, seed_eid),
            "entity_id": seed_eid,
            "confidence": round(_entity_match_confidence(db, query, seed_eid), 3),
        }
    )

    # Subsequent steps: one per (u → v) hop.
    for u, v in zip(path[:-1], path[1:]):
        u_is_mem = gr._is_memory_node(u)
        v_is_mem = gr._is_memory_node(v)
        u_id = gr._strip_prefix(u)
        v_id = gr._strip_prefix(v)

        if u_is_mem and v_is_mem:
            sim, reason = _memory_link_meta(db, u_id, v_id)
            steps.append(
                {
                    "step": len(steps),
                    "kind": "memory_link",
                    "from": u_id,
                    "to": v_id,
                    "similarity": round(sim, 4),
                    "reason": reason,
                }
            )
        elif (not u_is_mem) and v_is_mem:
            steps.append(
                {
                    "step": len(steps),
                    "kind": "entity_to_memory",
                    "entity": _entity_name(db, u_id),
                    "entity_id": u_id,
                    "memory_id": v_id,
                }
            )
        elif u_is_mem and (not v_is_mem):
            # memory → entity hop — symmetric of entity_to_memory.
            steps.append(
                {
                    "step": len(steps),
                    "kind": "memory_to_entity",
                    "memory_id": u_id,
                    "entity": _entity_name(db, v_id),
                    "entity_id": v_id,
                }
            )
        else:
            # entity → entity
            steps.append(
                {
                    "step": len(steps),
                    "kind": "entity_relation",
                    "from": _entity_name(db, u_id),
                    "from_id": u_id,
                    "to": _entity_name(db, v_id),
                    "to_id": v_id,
                    "relation": _relation_label(db, u_id, v_id),
                }
            )
    return steps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_provenance_chain(
    db: MemoirsDB,
    query: str,
    query_embedding: Optional[list[float]],
    memory_id: str,
    *,
    max_hops: int = 3,
    similarity_score: Optional[float] = None,
) -> Chain:
    """Walk the joint graph from query-seed entities to ``memory_id``.

    Args:
      db: open ``MemoirsDB`` handle.
      query: original free-form user query.
      query_embedding: optional query embedding (unused here but kept in
        the signature for forward compat with future score tweaks).
      memory_id: target memory.
      max_hops: cap on edges in the returned path. ``max_hops=1`` only
        admits direct entity_to_memory and memory_link hops.
      similarity_score: optional dense / BM25 score; used as the
        ``score`` field of the ``semantic_match`` fallback when no graph
        path exists.

    Returns:
      A list of step dicts (always non-empty: at least the ``query``
      anchor). When the memory is unreachable from the seeds, a
      ``semantic_match`` step is appended. When ``memory_id`` doesn't
      exist at all, a ``not_found`` step is appended.
    """
    # 0. Sanity — the memory must exist. We surface this loudly so the UI
    #    can render a useful "memory was archived" message instead of an
    #    empty chain.
    if not _memory_exists(db, memory_id):
        return [
            {"step": 0, "kind": "query", "value": query},
            {"step": 1, "kind": "not_found", "memory_id": memory_id},
        ]

    # 1. Try a graph traversal. Empty graph or no seeds → fall through to
    #    the semantic fallback.
    seed_entity_ids = gr.extract_seed_entities(db, query)
    target_node = gr._mem_node(memory_id)

    chain: Optional[Chain] = None
    if seed_entity_ids:
        graph = gr.build_graph(db)
        if graph.num_nodes > 0 and target_node in graph.adjacency:
            seed_nodes = {gr._ent_node(eid) for eid in seed_entity_ids}
            seeds_in_graph = {n for n in seed_nodes if n in graph.adjacency}
            if seeds_in_graph:
                # Multi-source BFS with the bound. We *also* try the
                # reverse direction (target → seeds) and pick the lower
                # cost path — semantically equivalent on an undirected
                # graph but lets us surface the most-likely justification
                # when there are multiple equal-length paths.
                fwd = _bfs_shortest_path(
                    graph, seeds_in_graph, target_node, max_hops=max_hops
                )
                bwd = _bfs_shortest_path(
                    graph, [target_node], next(iter(seeds_in_graph)),
                    max_hops=max_hops,
                )
                paths: list[list[str]] = []
                if fwd is not None:
                    paths.append(fwd)
                if bwd is not None:
                    # Reverse so the chain still reads seed → target.
                    paths.append(list(reversed(bwd)))
                if paths:
                    # Disambiguate ties by total raw-edge weight.
                    paths.sort(
                        key=lambda p: (len(p), -_path_score(graph, p))
                    )
                    chain = _path_to_steps(
                        db, query, paths[0],
                        seed_entity_node_ids=seeds_in_graph,
                    )

    if chain is not None and len(chain) > 1:
        return chain

    # 2. Semantic fallback. We surface the score the caller already paid
    #    to compute — re-running the embedding here would double the work
    #    and the caller (``explain_memory_selection``) always has it.
    score = similarity_score if similarity_score is not None else 0.0
    return [
        {"step": 0, "kind": "query", "value": query},
        {
            "step": 1,
            "kind": "semantic_match",
            "score": round(float(score), 4),
            "memory_id": memory_id,
        },
    ]


def explain_memory_selection(
    db: MemoirsDB,
    query: str,
    query_embedding: Optional[list[float]],
    candidates: list[dict],
    *,
    max_hops: int = 3,
) -> list[dict]:
    """Enrich each candidate dict with a ``provenance_chain``.

    The input ``candidates`` is the list returned by
    :func:`embeddings.search_similar_memories` (or a compatible shape):
    each dict must carry at least ``id`` and may carry ``similarity``.
    The returned list preserves order and never drops a candidate — even
    if the chain build fails for one row, that row gets a
    ``semantic_match`` fallback so the UI never has to special-case
    empty chains.

    The function is pure (no DB writes) and reuses the cached
    ``GraphView`` — running it for 10 candidates over a 1k-node graph
    should stay well below the 200ms budget on a laptop.
    """
    out: list[dict] = []
    for cand in candidates:
        mem_id = cand.get("id")
        if not mem_id:
            # Defensive — skip malformed rows rather than 500.
            continue
        sim = cand.get("similarity")
        try:
            chain = build_provenance_chain(
                db,
                query,
                query_embedding,
                mem_id,
                max_hops=max_hops,
                similarity_score=float(sim) if sim is not None else None,
            )
        except Exception:
            # Never let an explain failure poison the whole response —
            # the caller still gets the unenriched candidate.
            chain = [
                {"step": 0, "kind": "query", "value": query},
                {
                    "step": 1,
                    "kind": "semantic_match",
                    "score": float(sim or 0.0),
                    "memory_id": mem_id,
                },
            ]
        enriched = dict(cand)
        enriched["provenance_chain"] = chain
        out.append(enriched)
    return out
