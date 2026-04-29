"""RAPTOR-style hierarchical summary tree (P1-6).

Implements `Sarthi et al., RAPTOR (ICLR 2024) <https://arxiv.org/abs/2401.18059>`_
for the memoirs corpus: cluster the active memories by embedding similarity,
summarize each cluster into a parent node, then recurse on the new layer
until only a handful of nodes remain. Retrieval can then descend the tree
(or query *all* levels at once) to pick the granularity best matching the
user's question — broad summaries for broad queries, leaf memories for
specific ones.

The module is intentionally self-contained and side-effect-free against
``memory_engine.py`` / ``hybrid_retrieval.py`` (those are edited in parallel
by other agents). Wire-up via ``raptor_search`` is exposed at module level
so the parent retrieval pipeline can call it explicitly later.

Design choices
--------------
- **Clustering**: K-means via scikit-learn when available (chosen because it
  handles dense unit-normalized vectors well and we already require
  ``embeddings`` extra anyway). Fallback: a deterministic greedy
  similarity-threshold clustering keyed by cosine, so the suite stays green
  on minimal installs.
- **Summarization**: Gemma when ``llm`` is supplied (we reuse
  ``gemma._wrap_prompt`` and ``gemma._count_tokens`` for token-budget safety).
  Fallback: header concatenation + TF-IDF keyword extraction — deterministic
  and good enough for retrieval routing.
- **Persistence**: every node has a ``parent_id`` (NULL for the root) so we
  can walk subtrees with a single recursive CTE. ``summary_node_members``
  stores both ``memory`` (level-0 leaves) and ``summary`` members (interior
  edges) in one table to keep tree ops uniform.
- **Idempotence**: ``build_raptor_tree(..., rebuild=False)`` is a no-op when
  a tree already exists for the given scope. Pass ``rebuild=True`` to
  ``delete_subtree`` and rebuild from scratch.
"""
from __future__ import annotations

import logging
import math
import re
import struct
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..core.ids import stable_id, utc_now
from ..db import MemoirsDB
from . import embeddings as emb


log = logging.getLogger("memoirs.raptor")


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ClusterMember:
    """A single member of a candidate cluster (level-N node)."""

    node_id: str           # memory id (level=0) or summary_node id (level≥1)
    kind: str              # "memory" | "summary"
    embedding: list[float]
    content: str
    level: int = 0

    @property
    def member_kind(self) -> str:
        return self.kind


@dataclass
class Cluster:
    """A K-means / threshold cluster ready to be summarized."""

    members: list[ClusterMember] = field(default_factory=list)
    centroid: list[float] = field(default_factory=list)

    def __len__(self) -> int:  # convenience
        return len(self.members)


@dataclass
class SummaryNode:
    """In-memory mirror of a row in ``summary_nodes``."""

    id: str
    level: int
    content: str
    embedding: list[float]
    child_count: int
    parent_id: str | None
    scope_kind: str | None
    scope_id: str | None
    members: list[tuple[str, str, float]] = field(default_factory=list)
    # ``members`` triples: (member_kind, member_id, similarity)


@dataclass
class SummaryTree:
    """Stats / handle for a built tree."""

    scope_kind: str
    scope_id: str | None
    leaf_count: int
    levels: list[tuple[int, int]]  # (level, count)
    root_id: str | None


# ---------------------------------------------------------------------------
# Schema bootstrap (fallback when migration system isn't run)
# ---------------------------------------------------------------------------


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS summary_nodes (
    id TEXT PRIMARY KEY,
    level INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB,
    child_count INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    parent_id TEXT REFERENCES summary_nodes(id) ON DELETE SET NULL,
    scope_kind TEXT,
    scope_id TEXT
);
CREATE TABLE IF NOT EXISTS summary_node_members (
    node_id TEXT NOT NULL,
    member_kind TEXT NOT NULL,
    member_id TEXT NOT NULL,
    similarity REAL,
    PRIMARY KEY (node_id, member_kind, member_id)
);
CREATE INDEX IF NOT EXISTS idx_summary_nodes_parent
    ON summary_nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_summary_nodes_level
    ON summary_nodes(level);
CREATE INDEX IF NOT EXISTS idx_summary_nodes_scope
    ON summary_nodes(scope_kind, scope_id);
CREATE INDEX IF NOT EXISTS idx_summary_node_members_member
    ON summary_node_members(member_kind, member_id);
"""


def ensure_schema(db: MemoirsDB) -> None:
    """Create tables if not already present (fallback for legacy DBs)."""
    db.conn.executescript(_SCHEMA_SQL)


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity assuming neither vector is the zero vector."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    out = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            out[i] += v[i]
    n = float(len(vectors))
    return [x / n for x in out]


def _unpack_embedding(blob: bytes | None, dim: int) -> list[float] | None:
    if blob is None:
        return None
    if len(blob) != dim * 4:
        return None
    return list(struct.unpack(f"{dim}f", blob))


def _pack_embedding(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


# ---------------------------------------------------------------------------
# Member loading per level / scope
# ---------------------------------------------------------------------------


def _scope_clause(scope_kind: str, scope_id: str | None) -> tuple[str, list[Any]]:
    """Return ``(extra_sql, params)`` to filter memories by scope.

    The mapping from scope to memory selection:

    - ``global``       — every active memory.
    - ``project``      — memories that touch an entity of type=project with
                          ``normalized_name = scope_id``.
    - ``conversation`` — memories whose source candidate row points at the
                          given conversation_id.
    """
    if scope_kind == "global" or scope_id is None:
        return "", []
    if scope_kind == "project":
        return (
            " AND m.id IN ("
            "   SELECT me.memory_id FROM memory_entities me"
            "   JOIN entities e ON e.id = me.entity_id"
            "   WHERE e.type = 'project' AND e.normalized_name = ?"
            " )",
            [scope_id.lower()],
        )
    if scope_kind == "conversation":
        return (
            " AND m.id IN ("
            "   SELECT promoted_memory_id FROM memory_candidates"
            "   WHERE conversation_id = ? AND promoted_memory_id IS NOT NULL"
            " )",
            [scope_id],
        )
    # Unknown scope kind → behave like global, but log loudly so callers fix.
    log.warning("raptor: unknown scope_kind=%r — treating as global", scope_kind)
    return "", []


def _load_level_members(
    db: MemoirsDB,
    *,
    level: int,
    scope_kind: str,
    scope_id: str | None,
) -> list[ClusterMember]:
    """Pull every member at the requested level, scoped accordingly.

    Level 0 reads from ``memories`` + ``memory_embeddings``. Level ≥ 1 reads
    from ``summary_nodes`` matching ``(level, scope_kind, scope_id)``.
    """
    members: list[ClusterMember] = []
    if level == 0:
        from ..config import EMBEDDING_DIM
        scope_sql, scope_params = _scope_clause(scope_kind, scope_id)
        rows = db.conn.execute(
            f"""
            SELECT m.id, m.content, me.embedding
            FROM memories m
            JOIN memory_embeddings me ON me.memory_id = m.id
            WHERE m.archived_at IS NULL{scope_sql}
            """,
            scope_params,
        ).fetchall()
        for r in rows:
            vec = _unpack_embedding(bytes(r["embedding"]), EMBEDDING_DIM)
            if vec is None:
                continue
            members.append(
                ClusterMember(
                    node_id=r["id"],
                    kind="memory",
                    embedding=vec,
                    content=r["content"],
                    level=0,
                )
            )
        return members

    # level >= 1 — interior nodes
    rows = db.conn.execute(
        """
        SELECT id, content, embedding, level
        FROM summary_nodes
        WHERE level = ?
          AND COALESCE(scope_kind, 'global') = COALESCE(?, scope_kind, 'global')
          AND ((scope_id IS NULL AND ? IS NULL) OR scope_id = ?)
          AND parent_id IS NULL
        """,
        (level, scope_kind, scope_id, scope_id),
    ).fetchall()
    from ..config import EMBEDDING_DIM
    for r in rows:
        emb_blob = r["embedding"]
        vec = _unpack_embedding(bytes(emb_blob) if emb_blob is not None else None,
                                EMBEDDING_DIM)
        if vec is None:
            continue
        members.append(
            ClusterMember(
                node_id=r["id"],
                kind="summary",
                embedding=vec,
                content=r["content"],
                level=int(r["level"]),
            )
        )
    return members


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _have_sklearn() -> bool:
    try:
        import sklearn.cluster  # noqa: F401
        return True
    except ImportError:
        return False


def _kmeans_cluster(
    members: list[ClusterMember],
    *,
    k_per_cluster: int,
) -> list[Cluster]:
    """K-means clustering using scikit-learn. Picks K = max(2, n // k_per_cluster)."""
    from sklearn.cluster import KMeans
    n = len(members)
    if n < 2:
        return [Cluster(members=list(members),
                        centroid=members[0].embedding if members else [])]
    k = max(2, min(n // max(1, k_per_cluster), n - 1))
    # Cap K so we never produce singleton-only clusters when corpus is tiny.
    k = max(1, k)
    vectors = [m.embedding for m in members]
    # n_init='auto' → silence sklearn>=1.4 deprecation while remaining
    # compatible with older versions that ignore the kw.
    try:
        model = KMeans(n_clusters=k, n_init="auto", random_state=42)
    except TypeError:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(vectors)
    clusters_by_label: dict[int, list[ClusterMember]] = {}
    for label, mem in zip(labels, members):
        clusters_by_label.setdefault(int(label), []).append(mem)
    out: list[Cluster] = []
    for lbl, mems in clusters_by_label.items():
        out.append(Cluster(members=mems,
                           centroid=_centroid([m.embedding for m in mems])))
    return out


def _greedy_cluster(
    members: list[ClusterMember],
    *,
    threshold: float = 0.6,
) -> list[Cluster]:
    """Deterministic fallback: greedy similarity-threshold clustering.

    Scan members in order; each one either joins the highest-similarity
    existing cluster (if cosine ≥ ``threshold``) or seeds a new cluster.
    Centroids are recomputed incrementally so later assignments see the
    cluster's running mean.
    """
    clusters: list[Cluster] = []
    for mem in members:
        best_idx = -1
        best_sim = -1.0
        for idx, c in enumerate(clusters):
            sim = _cosine(c.centroid, mem.embedding)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        if best_idx >= 0 and best_sim >= threshold:
            clusters[best_idx].members.append(mem)
            clusters[best_idx].centroid = _centroid(
                [m.embedding for m in clusters[best_idx].members]
            )
        else:
            clusters.append(Cluster(members=[mem], centroid=list(mem.embedding)))
    return clusters


def cluster_memories(
    db: MemoirsDB,
    *,
    scope_kind: str = "global",
    scope_id: str | None = None,
    level: int = 0,
    k_per_cluster: int = 8,
) -> list[Cluster]:
    """Cluster the active members at ``level`` into RAPTOR groups.

    Returns one ``Cluster`` per grouping. Clusters with fewer than 2 members
    are still returned but the build loop will *not* summarize them — they
    pass through to the next level unchanged so that orphan nodes still get
    a chance to merge once the corpus shrinks.
    """
    members = _load_level_members(
        db, level=level, scope_kind=scope_kind, scope_id=scope_id
    )
    if not members:
        return []
    if _have_sklearn():
        return _kmeans_cluster(members, k_per_cluster=k_per_cluster)
    return _greedy_cluster(members)


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


_TFIDF_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "up", "down",
    "and", "or", "but", "not", "no", "if", "then", "else", "when", "where",
    "why", "how", "what", "who", "which", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their",
    "do", "does", "did", "have", "has", "had", "can", "could", "should",
    "would", "will", "shall", "may", "might", "must", "as", "than", "into",
    "about", "over", "after", "before", "very", "more", "most", "such",
})


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _heuristic_summarize(cluster: Cluster, *, max_chars: int = 320) -> str:
    """Deterministic fallback summary built from headers + top TF-IDF keywords.

    1. Take the first 60 chars of each member as a header.
    2. Compute term frequencies across the cluster (excluding stop-words).
    3. Emit "<top-3 keywords>: <header1> | <header2> | …" capped at
       ``max_chars``. The output is deterministic so tests that don't have
       Gemma still see the same value across runs.
    """
    headers: list[str] = []
    seen_headers: set[str] = set()
    counts: Counter[str] = Counter()
    df: Counter[str] = Counter()
    for m in cluster.members:
        text = (m.content or "").strip().splitlines()
        first_line = text[0] if text else ""
        head = first_line[:60].strip() or (m.content or "")[:60].strip()
        if head and head not in seen_headers:
            seen_headers.add(head)
            headers.append(head)
        toks = [t for t in _tokenize(m.content) if t not in _TFIDF_STOPWORDS]
        counts.update(toks)
        df.update(set(toks))
    n_docs = max(1, len(cluster.members))
    # Approx TF-IDF: tf * log(1 + n_docs / df)
    scored: list[tuple[str, float]] = []
    for term, tf in counts.items():
        idf = math.log(1.0 + n_docs / max(1, df[term]))
        scored.append((term, tf * idf))
    scored.sort(key=lambda kv: kv[1], reverse=True)
    top_terms = [t for t, _ in scored[:3]]
    head_blob = " | ".join(headers[:6])
    if top_terms:
        prefix = ", ".join(top_terms)
        body = f"[{prefix}] {head_blob}"
    else:
        body = head_blob or "(empty cluster)"
    if len(body) > max_chars:
        body = body[: max_chars - 1] + "…"
    return body


def _gemma_summarize(cluster: Cluster, *, llm: Any, max_chars: int = 320) -> str:
    """Run Gemma with token-budget safety to summarize a cluster."""
    from . import gemma as gm

    headers = [(m.content or "").strip() for m in cluster.members]
    n = len(headers)
    body = "\n- ".join(h for h in headers if h)
    convo = (
        f"Summarize these {n} memories in 1-2 sentences capturing the core fact:\n"
        f"- {body}"
    )
    # Compose prompt under model context.
    try:
        budget = gm._content_token_budget(llm)
        # If body too long, truncate by tokens via gemma's chunker (we still
        # want a single summary call).
        tokens = gm._count_tokens(llm, convo)
        if tokens > budget:
            # Trim head-of-list until under budget; preserves the lead-in.
            while tokens > budget and headers:
                headers.pop()
                body = "\n- ".join(h for h in headers if h)
                convo = (
                    f"Summarize these {len(headers)} memories in 1-2 sentences "
                    f"capturing the core fact:\n- {body}"
                )
                tokens = gm._count_tokens(llm, convo)
        prompt = gm._wrap_prompt(convo)
    except Exception:
        # If Gemma helpers misbehave, bail out to heuristic.
        log.warning("raptor: Gemma helper failed; falling back to heuristic")
        return _heuristic_summarize(cluster, max_chars=max_chars)

    try:
        out = llm.create_completion(
            prompt=prompt,
            max_tokens=200,
            temperature=0.2,
            stop=["<end_of_turn>", "\n\n\n"],
        )
        text = (out["choices"][0]["text"] or "").strip()
    except Exception as e:
        log.warning("raptor: Gemma completion failed (%s); falling back", e)
        return _heuristic_summarize(cluster, max_chars=max_chars)
    if not text:
        return _heuristic_summarize(cluster, max_chars=max_chars)
    return text[:max_chars]


def summarize_cluster(
    cluster: Cluster,
    *,
    llm: Any | None = None,
    max_chars: int = 320,
) -> str:
    """Produce a single-string summary for the cluster (no DB writes).

    The persisted node is created by :func:`build_raptor_tree`. Splitting the
    "content" generation step out keeps the function trivially mockable in
    tests.
    """
    if not cluster.members:
        return ""
    if llm is not None:
        return _gemma_summarize(cluster, llm=llm, max_chars=max_chars)
    return _heuristic_summarize(cluster, max_chars=max_chars)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _persist_summary_node(
    db: MemoirsDB,
    *,
    cluster: Cluster,
    level: int,
    summary_text: str,
    scope_kind: str,
    scope_id: str | None,
    parent_id: str | None = None,
    embed_summary: bool = True,
) -> SummaryNode:
    """Insert a row into ``summary_nodes`` + members, return the SummaryNode."""
    ensure_schema(db)
    node_id = stable_id(
        "sum",
        scope_kind,
        scope_id or "",
        level,
        summary_text[:200],
        ",".join(sorted(m.node_id for m in cluster.members))[:512],
    )
    # Prefer summary embedding via embed_text if available; fall back to
    # cluster centroid (cheap, deterministic, sufficient for retrieval).
    summary_vec: list[float] = []
    if embed_summary:
        try:
            summary_vec = emb.embed_text(summary_text or "summary")
        except emb.EmbeddingsUnavailable:
            summary_vec = list(cluster.centroid) if cluster.centroid else []
        except Exception as e:  # pragma: no cover -- defensive
            log.warning("raptor: embed_text failed (%s); using centroid", e)
            summary_vec = list(cluster.centroid) if cluster.centroid else []
    if not summary_vec:
        summary_vec = list(cluster.centroid) if cluster.centroid else []

    blob = _pack_embedding(summary_vec) if summary_vec else None
    db.conn.execute(
        """
        INSERT OR REPLACE INTO summary_nodes
            (id, level, content, embedding, child_count, created_at,
             parent_id, scope_kind, scope_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            node_id,
            level,
            summary_text,
            blob,
            len(cluster.members),
            utc_now(),
            parent_id,
            scope_kind,
            scope_id,
        ),
    )
    members_triples: list[tuple[str, str, float]] = []
    for m in cluster.members:
        sim = _cosine(cluster.centroid, m.embedding) if cluster.centroid else 0.0
        member_kind = m.kind
        db.conn.execute(
            """
            INSERT OR REPLACE INTO summary_node_members
                (node_id, member_kind, member_id, similarity)
            VALUES (?, ?, ?, ?)
            """,
            (node_id, member_kind, m.node_id, float(sim)),
        )
        members_triples.append((member_kind, m.node_id, float(sim)))
        # Update child summary's parent_id (only when child IS a summary).
        if m.kind == "summary":
            db.conn.execute(
                "UPDATE summary_nodes SET parent_id = ? WHERE id = ?",
                (node_id, m.node_id),
            )
    db.conn.commit()
    return SummaryNode(
        id=node_id,
        level=level,
        content=summary_text,
        embedding=summary_vec,
        child_count=len(cluster.members),
        parent_id=parent_id,
        scope_kind=scope_kind,
        scope_id=scope_id,
        members=members_triples,
    )


# ---------------------------------------------------------------------------
# Tree builder
# ---------------------------------------------------------------------------


def _scope_has_existing_tree(
    db: MemoirsDB, scope_kind: str, scope_id: str | None
) -> bool:
    row = db.conn.execute(
        """
        SELECT COUNT(*) AS c FROM summary_nodes
        WHERE COALESCE(scope_kind, 'global') = COALESCE(?, scope_kind, 'global')
          AND ((scope_id IS NULL AND ? IS NULL) OR scope_id = ?)
        """,
        (scope_kind, scope_id, scope_id),
    ).fetchone()
    return int(row["c"]) > 0 if row else False


def build_raptor_tree(
    db: MemoirsDB,
    *,
    scope_kind: str = "global",
    scope_id: str | None = None,
    max_levels: int = 4,
    k_per_cluster: int = 8,
    min_top_nodes: int = 2,
    llm: Any | None = None,
    rebuild: bool = False,
) -> SummaryTree:
    """Build the full RAPTOR tree for the given scope.

    Loop: cluster current level → summarize each cluster of size >= 2 →
    those summaries become the inputs of the next level. Loop terminates
    when (a) only ``min_top_nodes`` or fewer nodes remain or (b)
    ``max_levels`` is reached.

    Idempotence: when a tree already exists for the scope and ``rebuild``
    is False, the existing root is returned without rewriting any row.
    With ``rebuild=True`` the existing subtree is wiped first.
    """
    ensure_schema(db)

    if rebuild:
        # Best-effort cleanup of all nodes for the scope before rebuilding.
        delete_subtree_by_scope(db, scope_kind=scope_kind, scope_id=scope_id)
    elif _scope_has_existing_tree(db, scope_kind, scope_id):
        return _stats_for_existing_tree(db, scope_kind, scope_id)

    # Snapshot of leaves for stats.
    leaves = _load_level_members(
        db, level=0, scope_kind=scope_kind, scope_id=scope_id
    )
    leaf_count = len(leaves)
    levels: list[tuple[int, int]] = [(0, leaf_count)]
    if leaf_count == 0:
        return SummaryTree(
            scope_kind=scope_kind,
            scope_id=scope_id,
            leaf_count=0,
            levels=levels,
            root_id=None,
        )

    current_level = 0
    last_node_id: str | None = None
    while current_level < max_levels:
        clusters = cluster_memories(
            db,
            scope_kind=scope_kind,
            scope_id=scope_id,
            level=current_level,
            k_per_cluster=k_per_cluster,
        )
        if not clusters:
            break
        # Filter out singletons — they pass through to the next level via
        # being already in summary_nodes (level≥1) or by being absorbed in
        # subsequent rounds. For level 0 leaves with no cluster mate, we
        # promote them as solo summaries (so the tree always covers all
        # leaves — RAPTOR §3.2).
        next_level = current_level + 1
        produced_count = 0
        for c in clusters:
            if len(c) < 2:
                # Promote singleton: persist as a summary node so the next
                # round can re-cluster summaries.
                summary_text = (
                    c.members[0].content if c.members else "(empty)"
                )[:320]
                node = _persist_summary_node(
                    db,
                    cluster=c,
                    level=next_level,
                    summary_text=summary_text,
                    scope_kind=scope_kind,
                    scope_id=scope_id,
                )
                last_node_id = node.id
                produced_count += 1
                continue
            summary_text = summarize_cluster(c, llm=llm)
            node = _persist_summary_node(
                db,
                cluster=c,
                level=next_level,
                summary_text=summary_text,
                scope_kind=scope_kind,
                scope_id=scope_id,
            )
            last_node_id = node.id
            produced_count += 1
        levels.append((next_level, produced_count))
        if produced_count <= min_top_nodes:
            break
        current_level = next_level

    return SummaryTree(
        scope_kind=scope_kind,
        scope_id=scope_id,
        leaf_count=leaf_count,
        levels=levels,
        root_id=last_node_id,
    )


def _stats_for_existing_tree(
    db: MemoirsDB, scope_kind: str, scope_id: str | None
) -> SummaryTree:
    """Compute the SummaryTree summary for a scope without modifying it."""
    leaf_count = db.conn.execute(
        f"""
        SELECT COUNT(DISTINCT m.id) AS c
        FROM memories m
        JOIN memory_embeddings me ON me.memory_id = m.id
        WHERE m.archived_at IS NULL
        """
    ).fetchone()["c"]
    levels_rows = db.conn.execute(
        """
        SELECT level, COUNT(*) AS c
        FROM summary_nodes
        WHERE COALESCE(scope_kind, 'global') = COALESCE(?, scope_kind, 'global')
          AND ((scope_id IS NULL AND ? IS NULL) OR scope_id = ?)
        GROUP BY level
        ORDER BY level
        """,
        (scope_kind, scope_id, scope_id),
    ).fetchall()
    levels = [(0, int(leaf_count))] + [
        (int(r["level"]), int(r["c"])) for r in levels_rows
    ]
    root_row = db.conn.execute(
        """
        SELECT id FROM summary_nodes
        WHERE parent_id IS NULL
          AND COALESCE(scope_kind, 'global') = COALESCE(?, scope_kind, 'global')
          AND ((scope_id IS NULL AND ? IS NULL) OR scope_id = ?)
        ORDER BY level DESC
        LIMIT 1
        """,
        (scope_kind, scope_id, scope_id),
    ).fetchone()
    return SummaryTree(
        scope_kind=scope_kind,
        scope_id=scope_id,
        leaf_count=int(leaf_count),
        levels=levels,
        root_id=root_row["id"] if root_row else None,
    )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieve_raptor(
    db: MemoirsDB,
    query_embedding: list[float],
    *,
    top_k: int = 10,
    scope_kind: str = "global",
    scope_id: str | None = None,
    prefer_high_level: bool = False,
) -> list[tuple[str, float, int, list[str]]]:
    """Tree-descent retrieval over the RAPTOR tree.

    Returns up to ``top_k`` tuples of ``(memory_id, score, level, path)``
    where ``path`` is the list of summary_node ids walked from the root.
    Scores are cosine similarities; with ``prefer_high_level=True`` we add
    a small ``+0.05 * level`` bonus so broad queries surface broad summaries.

    Strategy
    --------
    Pure RAPTOR-style "all nodes" KNN: iterate every (memory + summary_node)
    embedding visible in the scope, score by cosine, return top_k. Path is
    derived from ``parent_id`` for summary nodes; for memories we walk
    ``summary_node_members`` upward.
    """
    ensure_schema(db)
    from ..config import EMBEDDING_DIM

    # Score memories
    scope_sql, scope_params = _scope_clause(scope_kind, scope_id)
    rows_mem = db.conn.execute(
        f"""
        SELECT m.id AS id, me.embedding AS emb
        FROM memories m
        JOIN memory_embeddings me ON me.memory_id = m.id
        WHERE m.archived_at IS NULL{scope_sql}
        """,
        scope_params,
    ).fetchall()

    scored: list[tuple[str, float, int, str]] = []
    # (member_id, score, level, kind)
    for r in rows_mem:
        vec = _unpack_embedding(bytes(r["emb"]), EMBEDDING_DIM)
        if vec is None:
            continue
        sim = _cosine(query_embedding, vec)
        scored.append((r["id"], sim, 0, "memory"))

    # Score summaries
    rows_sum = db.conn.execute(
        """
        SELECT id, level, embedding
        FROM summary_nodes
        WHERE COALESCE(scope_kind, 'global') = COALESCE(?, scope_kind, 'global')
          AND ((scope_id IS NULL AND ? IS NULL) OR scope_id = ?)
        """,
        (scope_kind, scope_id, scope_id),
    ).fetchall()
    for r in rows_sum:
        emb_blob = r["embedding"]
        vec = _unpack_embedding(bytes(emb_blob) if emb_blob is not None else None,
                                EMBEDDING_DIM)
        if vec is None:
            continue
        sim = _cosine(query_embedding, vec)
        if prefer_high_level:
            sim = sim + 0.05 * int(r["level"])
        scored.append((r["id"], sim, int(r["level"]), "summary"))

    scored.sort(key=lambda t: t[1], reverse=True)

    # Build leaf-only result list. For interior matches, we walk DOWN to
    # collect their member memories (RAPTOR-style: a hit on a summary
    # implies its leaves are relevant). We also annotate the path via
    # parent_id for any node, providing context.
    seen_mem: set[str] = set()
    out: list[tuple[str, float, int, list[str]]] = []
    for member_id, score, level, kind in scored:
        if len(out) >= top_k:
            break
        if kind == "memory":
            if member_id in seen_mem:
                continue
            path = _ancestor_path_for_memory(db, member_id)
            seen_mem.add(member_id)
            out.append((member_id, float(score), level, path))
            continue
        # summary: collect its leaves
        leaves = _collect_leaf_memories(db, member_id)
        for mem_id in leaves:
            if len(out) >= top_k:
                break
            if mem_id in seen_mem:
                continue
            path = _ancestor_path_from_summary(db, member_id)
            seen_mem.add(mem_id)
            out.append((mem_id, float(score), level, path))
    return out


def _ancestor_path_for_memory(db: MemoirsDB, memory_id: str) -> list[str]:
    """Walk up via ``summary_node_members`` then ``parent_id`` to root."""
    row = db.conn.execute(
        "SELECT node_id FROM summary_node_members "
        "WHERE member_kind = 'memory' AND member_id = ? LIMIT 1",
        (memory_id,),
    ).fetchone()
    if not row:
        return []
    return _ancestor_path_from_summary(db, row["node_id"])


def _ancestor_path_from_summary(db: MemoirsDB, node_id: str) -> list[str]:
    path: list[str] = []
    current = node_id
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        path.append(current)
        row = db.conn.execute(
            "SELECT parent_id FROM summary_nodes WHERE id = ?", (current,)
        ).fetchone()
        if not row or not row["parent_id"]:
            break
        current = row["parent_id"]
    return path


def _collect_leaf_memories(db: MemoirsDB, node_id: str) -> list[str]:
    """BFS down to find all memory-kind descendants of node_id."""
    out: list[str] = []
    stack = [node_id]
    seen: set[str] = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        rows = db.conn.execute(
            "SELECT member_kind, member_id FROM summary_node_members "
            "WHERE node_id = ?",
            (cur,),
        ).fetchall()
        for r in rows:
            if r["member_kind"] == "memory":
                out.append(r["member_id"])
            elif r["member_kind"] == "summary":
                stack.append(r["member_id"])
    return out


def raptor_search(
    db: MemoirsDB,
    query: str,
    query_embedding: list[float] | None = None,
    *,
    top_k: int = 10,
    scope_kind: str = "global",
    scope_id: str | None = None,
    prefer_high_level: bool = False,
) -> list[dict[str, Any]]:
    """High-level retrieval entry point.

    Wrapper around :func:`retrieve_raptor` that:
      * embeds the query if no precomputed embedding is supplied,
      * returns dicts shaped like the rest of the retrieval pipeline so the
        result can be merged into ``_retrieve_candidates`` with minimal
        glue (TODO: wire into ``memory_engine._resolve_retrieval_mode`` —
        another agent owns that file in this sprint).
    """
    ensure_schema(db)
    if query_embedding is None:
        try:
            query_embedding = emb.embed_text_cached(query)
        except emb.EmbeddingsUnavailable:
            log.warning("raptor_search: embeddings unavailable, returning []")
            return []
    raw = retrieve_raptor(
        db,
        query_embedding,
        top_k=top_k,
        scope_kind=scope_kind,
        scope_id=scope_id,
        prefer_high_level=prefer_high_level,
    )
    out: list[dict[str, Any]] = []
    for memory_id, score, level, path in raw:
        out.append({
            "memory_id": memory_id,
            "score": float(score),
            "level": int(level),
            "path": list(path),
        })
    return out


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def delete_subtree(db: MemoirsDB, node_id: str) -> int:
    """Recursively delete a summary node + all descendants.

    Returns the count of summary_nodes rows deleted (members are ON DELETE
    cascaded manually since ``summary_node_members`` is not FK-bound).
    """
    ensure_schema(db)
    to_delete: list[str] = []
    stack = [node_id]
    seen: set[str] = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        to_delete.append(cur)
        rows = db.conn.execute(
            "SELECT member_id FROM summary_node_members "
            "WHERE node_id = ? AND member_kind = 'summary'",
            (cur,),
        ).fetchall()
        for r in rows:
            stack.append(r["member_id"])
        # Also pick up direct children via parent_id (covers cases where
        # member rows were lost).
        rows2 = db.conn.execute(
            "SELECT id FROM summary_nodes WHERE parent_id = ?", (cur,)
        ).fetchall()
        for r in rows2:
            stack.append(r["id"])
    if not to_delete:
        return 0
    cur = db.conn.cursor()
    chunk = 400
    for i in range(0, len(to_delete), chunk):
        batch = to_delete[i : i + chunk]
        placeholders = ",".join("?" * len(batch))
        cur.execute(
            f"DELETE FROM summary_node_members WHERE node_id IN ({placeholders})",
            batch,
        )
        cur.execute(
            f"DELETE FROM summary_nodes WHERE id IN ({placeholders})", batch
        )
    db.conn.commit()
    return len(to_delete)


def delete_subtree_by_scope(
    db: MemoirsDB,
    *,
    scope_kind: str = "global",
    scope_id: str | None = None,
) -> int:
    """Drop every node + members for a (scope_kind, scope_id) pair."""
    ensure_schema(db)
    rows = db.conn.execute(
        """
        SELECT id FROM summary_nodes
        WHERE COALESCE(scope_kind, 'global') = COALESCE(?, scope_kind, 'global')
          AND ((scope_id IS NULL AND ? IS NULL) OR scope_id = ?)
        """,
        (scope_kind, scope_id, scope_id),
    ).fetchall()
    ids = [r["id"] for r in rows]
    if not ids:
        return 0
    chunk = 400
    cur = db.conn.cursor()
    for i in range(0, len(ids), chunk):
        batch = ids[i : i + chunk]
        placeholders = ",".join("?" * len(batch))
        cur.execute(
            f"DELETE FROM summary_node_members WHERE node_id IN ({placeholders})",
            batch,
        )
        cur.execute(
            f"DELETE FROM summary_nodes WHERE id IN ({placeholders})", batch
        )
    db.conn.commit()
    return len(ids)


# ---------------------------------------------------------------------------
# CLI helpers (used by memoirs/cli.py)
# ---------------------------------------------------------------------------


def get_node(db: MemoirsDB, node_id: str) -> dict[str, Any] | None:
    """Lookup a single summary node + its members for ``raptor show``."""
    ensure_schema(db)
    row = db.conn.execute(
        "SELECT id, level, content, child_count, created_at, parent_id, "
        "scope_kind, scope_id FROM summary_nodes WHERE id = ? OR id LIKE ? "
        "LIMIT 1",
        (node_id, f"{node_id}%"),
    ).fetchone()
    if not row:
        return None
    members = [
        dict(r) for r in db.conn.execute(
            "SELECT member_kind, member_id, similarity FROM summary_node_members "
            "WHERE node_id = ? ORDER BY similarity DESC",
            (row["id"],),
        ).fetchall()
    ]
    return {**dict(row), "members": members}


def stats(
    db: MemoirsDB,
    *,
    scope_kind: str | None = None,
    scope_id: str | None = None,
) -> dict[str, Any]:
    """Aggregate counts per level + root_id for the given scope (or all)."""
    ensure_schema(db)
    if scope_kind is None:
        rows = db.conn.execute(
            "SELECT scope_kind, scope_id, level, COUNT(*) AS c "
            "FROM summary_nodes GROUP BY scope_kind, scope_id, level "
            "ORDER BY scope_kind, scope_id, level"
        ).fetchall()
        return {"levels": [dict(r) for r in rows]}
    rows = db.conn.execute(
        """
        SELECT level, COUNT(*) AS c
        FROM summary_nodes
        WHERE COALESCE(scope_kind, 'global') = COALESCE(?, scope_kind, 'global')
          AND ((scope_id IS NULL AND ? IS NULL) OR scope_id = ?)
        GROUP BY level
        ORDER BY level
        """,
        (scope_kind, scope_id, scope_id),
    ).fetchall()
    root = db.conn.execute(
        """
        SELECT id FROM summary_nodes
        WHERE parent_id IS NULL
          AND COALESCE(scope_kind, 'global') = COALESCE(?, scope_kind, 'global')
          AND ((scope_id IS NULL AND ? IS NULL) OR scope_id = ?)
        ORDER BY level DESC LIMIT 1
        """,
        (scope_kind, scope_id, scope_id),
    ).fetchone()
    leaves = db.conn.execute(
        "SELECT COUNT(*) AS c FROM memories WHERE archived_at IS NULL"
    ).fetchone()["c"]
    return {
        "scope_kind": scope_kind,
        "scope_id": scope_id,
        "leaf_count": int(leaves),
        "levels": [dict(r) for r in rows],
        "root_id": root["id"] if root else None,
    }
