"""Layer 5b — Maximal Marginal Relevance diversification (P2-4).

Carbonell & Goldstein, 1998. Greedy re-ranker that, at each step, picks
the candidate that maximizes::

    λ · rel(c)  -  (1 - λ) · max_sim(c, S)

where ``S`` is the set already chosen and ``max_sim`` is the highest cosine
similarity between ``c`` and any element of ``S``. Output is a re-ordered
subset of size ``k`` that trades some relevance for diversity, with ``λ``
controlling the trade-off (1.0 = pure relevance / top-K, 0.0 = pure
diversity).

Activation
----------
``MEMOIRS_MMR``        ∈ {``on``, ``off``} (default ``on``).
``MEMOIRS_MMR_LAMBDA`` float, default ``0.7``.

Caller contract
---------------
Embeddings are pulled via ``embedding_lookup(memory_id)``; that lookup is
cheap (DB hit + cache) but the MMR loop itself is O(k·n) similarity
products, so we keep all vectors in memory after the first miss.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

log = logging.getLogger("memoirs.mmr")


ENV_MMR = "MEMOIRS_MMR"
ENV_LAMBDA = "MEMOIRS_MMR_LAMBDA"
DEFAULT_LAMBDA = 0.7


# ---------------------------------------------------------------------------
# Candidate adapter
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """Lightweight MMR-side struct.

    Anything dict-like with ``id`` + ``score`` works in practice — we adapt
    via :func:`_to_candidate` so callers can pass their existing rows.
    """

    memory_id: str
    score: float
    embedding: Sequence[float] | None = None


def _to_candidate(c) -> Candidate:
    if isinstance(c, Candidate):
        return c
    if isinstance(c, dict):
        cid = c.get("memory_id") or c.get("id")
        if cid is None:
            raise ValueError(f"MMR candidate missing id: {c!r}")
        score = c.get("score")
        if score is None:
            score = c.get("similarity", 0.0)
        return Candidate(memory_id=str(cid), score=float(score or 0.0),
                         embedding=c.get("embedding"))
    cid = getattr(c, "memory_id", None) or getattr(c, "id", None)
    if cid is None:
        raise ValueError(f"MMR candidate missing id: {c!r}")
    return Candidate(
        memory_id=str(cid),
        score=float(getattr(c, "score", 0.0) or 0.0),
        embedding=getattr(c, "embedding", None),
    )


# ---------------------------------------------------------------------------
# Activation gate
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    raw = (os.environ.get(ENV_MMR) or "on").strip().lower()
    return raw in {"on", "1", "true", "yes"}


def get_lambda() -> float:
    raw = os.environ.get(ENV_LAMBDA)
    if raw is None or raw == "":
        return DEFAULT_LAMBDA
    try:
        v = float(raw)
    except ValueError:
        log.warning("invalid %s=%r — using %.2f", ENV_LAMBDA, raw, DEFAULT_LAMBDA)
        return DEFAULT_LAMBDA
    return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Falls back to numpy when both inputs are arrays for
    speed; pure-Python path keeps the test path zero-dep.
    """
    if a is None or b is None:
        return 0.0
    try:
        # Fast path: numpy arrays / lists of equal length.
        import numpy as np  # noqa: WPS433
        va = np.asarray(a, dtype=float)
        vb = np.asarray(b, dtype=float)
        if va.shape != vb.shape or va.size == 0:
            return 0.0
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))
    except ImportError:  # pragma: no cover - numpy is a transitive dep
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        na = math.sqrt(sum(float(x) * float(x) for x in a))
        nb = math.sqrt(sum(float(y) * float(y) for y in b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)


# ---------------------------------------------------------------------------
# Public algorithm
# ---------------------------------------------------------------------------


def mmr_select(
    candidates: Sequence,
    k: int,
    *,
    lambda_: float = DEFAULT_LAMBDA,
    embedding_lookup: Callable[[str], Sequence[float] | None] | None = None,
) -> list:
    """Greedy MMR selection. Preserves the input candidate objects.

    Parameters
    ----------
    candidates:
        Iterable of dicts / dataclasses. Each must expose an id and a score
        (see :func:`_to_candidate`). When a candidate already carries an
        ``embedding`` field it's used verbatim, else ``embedding_lookup``
        is consulted.
    k:
        Target output size. ``k <= 0`` returns ``[]``; ``k >= len`` returns
        the input unchanged (skipping the O(k·n) loop entirely).
    lambda_:
        Trade-off ∈ [0.0, 1.0]. ``1.0`` = relevance only (top-K); ``0.0`` =
        diversity only. Clamped to range.
    embedding_lookup:
        ``memory_id -> vector | None``. Returning None pushes the candidate
        to the tail (it'll still appear in the output, just without
        contributing to the diversity term).

    Returns
    -------
    list
        Subset of ``candidates`` (same Python objects) in MMR order, length
        ``min(k, len(candidates))``.
    """
    if k <= 0 or not candidates:
        return []
    cand_list = list(candidates)
    if k >= len(cand_list):
        # Nothing to diversify against; preserve input order.
        return cand_list
    lam = max(0.0, min(1.0, float(lambda_)))

    # Adapt to internal struct, but keep references back to the originals
    # so we can return the caller's exact dicts in MMR order.
    adapted: list[Candidate] = []
    originals: list = []
    for c in cand_list:
        try:
            adapted.append(_to_candidate(c))
        except ValueError:
            log.debug("mmr: skipping candidate without id: %r", c)
            continue
        originals.append(c)

    # Resolve embeddings up-front (memoised). None vectors are tolerated;
    # they contribute 0.0 to the diversity term and get pushed to the tail
    # only after every embedded candidate has been selected.
    vectors: list[Sequence[float] | None] = []
    for cand in adapted:
        vec = cand.embedding
        if vec is None and embedding_lookup is not None:
            try:
                vec = embedding_lookup(cand.memory_id)
            except Exception as e:  # pragma: no cover - defensive
                log.debug("mmr embedding_lookup error for %s: %s", cand.memory_id, e)
                vec = None
        vectors.append(vec)

    n = len(adapted)
    chosen: list[int] = []
    remaining = set(range(n))

    # Pre-compute pairwise similarity lazily inside the loop. For typical
    # n=20–50 the saved cycles versus an upfront matrix are negligible, but
    # we DO cache as we go so repeated MMR passes (rare) are fast.
    sim_cache: dict[tuple[int, int], float] = {}

    def pair_sim(i: int, j: int) -> float:
        if i == j:
            return 1.0
        key = (i, j) if i < j else (j, i)
        if key in sim_cache:
            return sim_cache[key]
        vi, vj = vectors[i], vectors[j]
        if vi is None or vj is None:
            sim_cache[key] = 0.0
            return 0.0
        s = _cosine(vi, vj)
        sim_cache[key] = s
        return s

    target = min(k, n)
    while len(chosen) < target and remaining:
        best_idx: int | None = None
        best_mmr = -math.inf
        # Among candidates without an embedding, defer them: only select
        # one if it's the best available _and_ everything else is exhausted
        # OR if their adjusted relevance is genuinely higher.
        for i in remaining:
            rel = adapted[i].score
            if not chosen:
                redundancy = 0.0
            else:
                redundancy = max(pair_sim(i, j) for j in chosen)
            mmr_score = lam * rel - (1.0 - lam) * redundancy
            # Tiny tie-break: prefer original order (stable selection across
            # runs / equal-score corpora).
            if mmr_score > best_mmr or (
                math.isclose(mmr_score, best_mmr) and (best_idx is None or i < best_idx)
            ):
                best_mmr = mmr_score
                best_idx = i
        if best_idx is None:
            break
        chosen.append(best_idx)
        remaining.discard(best_idx)

    return [originals[i] for i in chosen]
