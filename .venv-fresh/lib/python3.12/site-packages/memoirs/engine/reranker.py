"""Layer 5b — cross-encoder reranker (P2-2).

Pluggable reranker stage for the retrieval pipeline. Sits between the
candidate fetcher (BM25 / dense / hybrid / graph) and MMR diversification.

Backends
--------
* ``NoopReranker``  — identity. Default when no LLM-grade reranker is
  available, keeps the pipeline dep-free.
* ``BGERerankerLocal`` — ``BAAI/bge-reranker-v2-m3`` via
  ``sentence_transformers.CrossEncoder``. Loaded lazily; if the dep isn't
  installed (or the model can't load) we fall back to ``NoopReranker`` and
  emit a single warning.
* ``flashrank`` is reserved as a future backend; currently aliases to Noop
  with a warning so the env switch is forward-compatible.

Selection
---------
``get_reranker()`` returns a process-wide singleton whose backend is read
once from ``MEMOIRS_RERANKER_BACKEND`` ∈ {``none``, ``bge``, ``flashrank``}
(default ``none``). ``MEMOIRS_RERANK_TOP_N`` (default ``50``) caps the
number of candidates we feed into the cross-encoder — reranking the long
tail wastes latency for vanishingly small precision gains.
"""
from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Iterable, Sequence

log = logging.getLogger("memoirs.reranker")


# Public env knobs — re-read inside ``get_reranker`` so tests / runtime
# overrides take effect without restart.
ENV_BACKEND = "MEMOIRS_RERANKER_BACKEND"
ENV_TOP_N = "MEMOIRS_RERANK_TOP_N"
DEFAULT_TOP_N = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _candidate_id(c) -> str | None:
    """Best-effort: pull a memory id off a candidate (dict or dataclass)."""
    if isinstance(c, dict):
        return c.get("id") or c.get("memory_id")
    return getattr(c, "id", None) or getattr(c, "memory_id", None)


def _candidate_content(c) -> str:
    if isinstance(c, dict):
        return str(c.get("content") or "")
    return str(getattr(c, "content", "") or "")


def _candidate_score(c) -> float:
    if isinstance(c, dict):
        # Prefer the unified score, fall back to similarity.
        v = c.get("score")
        if v is None:
            v = c.get("similarity", 0.0)
        try:
            return float(v or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return float(getattr(c, "score", 0.0) or 0.0)


def _read_top_n() -> int:
    raw = os.environ.get(ENV_TOP_N)
    if not raw:
        return DEFAULT_TOP_N
    try:
        n = int(raw)
        return max(1, n)
    except ValueError:
        log.warning("invalid %s=%r, using default %d", ENV_TOP_N, raw, DEFAULT_TOP_N)
        return DEFAULT_TOP_N


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class Reranker(ABC):
    """Abstract reranker. Returns ``[(memory_id, score), ...]`` parallel
    to the input order is NOT required — callers must sort/merge by id."""

    name: str = "abstract"

    @abstractmethod
    def rerank(self, query: str, candidates: Sequence) -> list[tuple[str, float]]:
        ...

    def is_available(self) -> bool:  # pragma: no cover - trivial
        return True


class NoopReranker(Reranker):
    """Identity reranker. Returns each candidate's existing score unchanged."""

    name = "noop"

    def rerank(self, query: str, candidates: Sequence) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for c in candidates:
            cid = _candidate_id(c)
            if cid is None:
                continue
            out.append((cid, _candidate_score(c)))
        return out


class BGERerankerLocal(Reranker):
    """``BAAI/bge-reranker-v2-m3`` via ``sentence_transformers.CrossEncoder``.

    Loaded lazily on the first ``rerank`` call so the import cost is only
    paid by users who actually opt in via the env switch.
    """

    name = "bge"
    MODEL_ID = "BAAI/bge-reranker-v2-m3"

    def __init__(self) -> None:
        self._model = None
        self._load_failed = False

    def is_available(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        try:
            from sentence_transformers import CrossEncoder  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_loaded(self):
        if self._model is not None or self._load_failed:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            log.warning(
                "BGERerankerLocal: sentence-transformers not installed. "
                "Install with: pip install -e '.[reranker]'"
            )
            self._load_failed = True
            return None
        try:
            log.info("loading cross-encoder %s", self.MODEL_ID)
            self._model = CrossEncoder(self.MODEL_ID)
        except Exception as e:  # network/model error
            log.warning("BGERerankerLocal: failed to load %s: %s", self.MODEL_ID, e)
            self._load_failed = True
            return None
        return self._model

    def rerank(self, query: str, candidates: Sequence) -> list[tuple[str, float]]:
        model = self._ensure_loaded()
        if model is None:
            # Graceful fallback to Noop if model couldn't load mid-flight.
            return NoopReranker().rerank(query, candidates)
        pairs: list[tuple[str, str]] = []
        ids: list[str] = []
        for c in candidates:
            cid = _candidate_id(c)
            if cid is None:
                continue
            ids.append(cid)
            pairs.append((query, _candidate_content(c)))
        if not pairs:
            return []
        try:
            scores = model.predict(pairs)
        except Exception as e:
            log.warning("BGERerankerLocal.predict failed: %s — falling back to Noop", e)
            return NoopReranker().rerank(query, candidates)
        return [(cid, float(s)) for cid, s in zip(ids, scores)]


# ---------------------------------------------------------------------------
# Singleton + factory
# ---------------------------------------------------------------------------


_SINGLETON: Reranker | None = None
_SINGLETON_KEY: str | None = None
_LOCK = threading.Lock()


def _resolve_backend_name() -> str:
    raw = (os.environ.get(ENV_BACKEND) or "none").strip().lower()
    if raw not in {"none", "bge", "flashrank"}:
        log.warning("unknown %s=%r — using 'none'", ENV_BACKEND, raw)
        return "none"
    return raw


def _build_reranker(backend: str) -> Reranker:
    if backend == "bge":
        r = BGERerankerLocal()
        if not r.is_available():
            log.warning(
                "MEMOIRS_RERANKER_BACKEND=bge but sentence-transformers is not "
                "installed — falling back to NoopReranker. Install with "
                "`pip install -e '.[reranker]'`."
            )
            return NoopReranker()
        return r
    if backend == "flashrank":
        log.warning(
            "MEMOIRS_RERANKER_BACKEND=flashrank is not yet wired — "
            "falling back to NoopReranker."
        )
        return NoopReranker()
    return NoopReranker()


def get_reranker() -> Reranker:
    """Process-wide singleton reranker.

    The backend is read each call from :data:`ENV_BACKEND`; if it changed
    since the last call (e.g. test ``monkeypatch.setenv``) we rebuild.
    """
    global _SINGLETON, _SINGLETON_KEY
    backend = _resolve_backend_name()
    if _SINGLETON is not None and _SINGLETON_KEY == backend:
        return _SINGLETON
    with _LOCK:
        if _SINGLETON is not None and _SINGLETON_KEY == backend:
            return _SINGLETON
        _SINGLETON = _build_reranker(backend)
        _SINGLETON_KEY = backend
        log.debug("reranker backend resolved: %s -> %s", backend, _SINGLETON.name)
        return _SINGLETON


def reset_reranker_singleton() -> None:
    """Drop the cached reranker. Tests use this to swap backends mid-run."""
    global _SINGLETON, _SINGLETON_KEY
    with _LOCK:
        _SINGLETON = None
        _SINGLETON_KEY = None


# ---------------------------------------------------------------------------
# Pipeline integration helper
# ---------------------------------------------------------------------------


def apply_rerank(
    query: str,
    candidates: list[dict],
    *,
    reranker: Reranker | None = None,
    top_n: int | None = None,
    score_field: str = "score",
) -> list[dict]:
    """Re-score (and re-sort) the top-N candidates with the configured reranker.

    Only the first ``top_n`` items are sent to the cross-encoder; the tail
    is left untouched and appended after the reranked head.

    The returned list is a NEW list of the same dict objects (not copies);
    each reranked candidate has ``rerank_score`` set and its primary
    ``score_field`` overwritten with the reranker's score so downstream
    sorts (combined ranking, MMR's relevance term) pick it up automatically.
    """
    if not candidates:
        return candidates
    rk = reranker if reranker is not None else get_reranker()
    if isinstance(rk, NoopReranker):
        # Fast path: nothing to do. Don't mutate, don't allocate.
        return candidates
    n = top_n if top_n is not None else _read_top_n()
    head = candidates[:n]
    tail = candidates[n:]
    pairs = rk.rerank(query, head)
    score_by_id = {cid: s for cid, s in pairs}
    # Reorder head by new score, descending.
    reranked: list[dict] = []
    for c in head:
        cid = _candidate_id(c)
        if cid is None or cid not in score_by_id:
            reranked.append(c)
            continue
        new_score = float(score_by_id[cid])
        c["rerank_score"] = new_score
        c[score_field] = new_score
        reranked.append(c)
    reranked.sort(
        key=lambda m: float(m.get("rerank_score", m.get(score_field, 0.0)) or 0.0),
        reverse=True,
    )
    return reranked + tail
