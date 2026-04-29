"""Process pool for sentence-transformers embeds (GIL-bypass).

Background
----------
``sentence_transformers.SentenceTransformer.encode`` holds the Python GIL
for the bulk of its CPU work. Under concurrent load (e.g. SLO bench with
50 worker threads hitting ``assemble_context``) every call serialises on
the GIL and sustained throughput collapses (50-worker hybrid_graph audit
measured 2 rps vs the 50 rps SLO target — see GAP fix #4).

This module spawns a small pool of worker *processes*, each holding its
own ``SentenceTransformer`` instance. Submissions go through a
``concurrent.futures.ProcessPoolExecutor`` so multiple encodes run truly
in parallel — one Python interpreter per worker, no shared GIL.

Trade-off: each worker takes ~200 MB of RAM for the all-MiniLM-L6-v2
weights, so 4 workers ≈ 800 MB extra resident. That's the price for
3-5× sustained throughput. Operators who can't pay it should prefer the
``fastembed`` ONNX backend, which releases the GIL inside the C
extension and needs no extra processes.

Public surface
--------------
* ``EmbedPool(n_workers, model_name)`` — manage the pool yourself.
* ``get_default_pool()`` — lazy module-level singleton driven by
  ``MEMOIRS_EMBED_POOL_WORKERS`` (default 4) and ``EMBEDDING_MODEL``.
* ``configure_pool(n_workers=...)`` — tweak the singleton before first
  use (CLI startup hook).
* ``shutdown_pool()`` — tear the singleton down (test cleanup).

The worker entry point lives at module scope so the ``spawn`` start
method can pickle it without dragging the parent process state along.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable

from ..config import EMBEDDING_MODEL


log = logging.getLogger("memoirs.embed_pool")


# Each worker process keeps its own ``SentenceTransformer`` here. We
# stash it as a module global so the per-call entry point doesn't have to
# re-load the weights on every submission (loading the model takes
# seconds — orders of magnitude slower than an actual encode).
_WORKER_MODEL = None  # type: ignore[var-annotated]


def _worker_init(model_name: str) -> None:
    """ProcessPool initializer: load the embedder once per worker."""
    global _WORKER_MODEL
    # Defer the import: ``sentence_transformers`` pulls in torch + a chunk
    # of HuggingFace, which we want to pay only inside workers.
    from sentence_transformers import SentenceTransformer

    _WORKER_MODEL = SentenceTransformer(model_name)


def _worker_embed(texts: list[str]) -> list[list[float]]:
    """Encode ``texts`` in the worker process. Returns L2-normalized vectors."""
    global _WORKER_MODEL
    if _WORKER_MODEL is None:
        # Defensive: ``initializer`` should have populated this. If it
        # didn't (e.g. a child process forked without calling our init),
        # load lazily so we still produce correct output.
        from sentence_transformers import SentenceTransformer

        _WORKER_MODEL = SentenceTransformer(EMBEDDING_MODEL)
    vecs = _WORKER_MODEL.encode(texts, normalize_embeddings=True)
    # Normalise to a plain list-of-list so the result pickles cleanly
    # (numpy arrays pickle fine but downstream callers expect lists).
    out: list[list[float]] = []
    for v in vecs:
        out.append([float(x) for x in v])
    return out


class EmbedPool:
    """Process pool wrapper around sentence-transformers ``encode``.

    Lazy-initialised: the worker processes start on the first ``embed``
    call so importing the module is cheap and deterministic.
    """

    def __init__(self, n_workers: int = 4, model_name: str = EMBEDDING_MODEL) -> None:
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        self.n_workers = n_workers
        self.model_name = model_name
        self._executor: ProcessPoolExecutor | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _ensure_started(self) -> ProcessPoolExecutor:
        """Spin up the pool on first use; safe to call from many threads."""
        if self._executor is not None:
            return self._executor
        with self._lock:
            if self._executor is not None:
                return self._executor
            ctx = mp.get_context("spawn")
            log.info(
                "starting EmbedPool with %d workers, model=%s",
                self.n_workers, self.model_name,
            )
            self._executor = ProcessPoolExecutor(
                max_workers=self.n_workers,
                mp_context=ctx,
                initializer=_worker_init,
                initargs=(self.model_name,),
            )
            return self._executor

    def shutdown(self, wait: bool = True) -> None:
        """Stop the pool and reap workers. Idempotent."""
        with self._lock:
            ex = self._executor
            self._executor = None
        if ex is not None:
            ex.shutdown(wait=wait, cancel_futures=not wait)

    def __enter__(self) -> "EmbedPool":
        self._ensure_started()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True)

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #

    def embed(self, text: str) -> list[float]:
        """Embed a single string. Convenience wrapper around ``embed_batch``."""
        out = self.embed_batch([text])
        return out[0]

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed every text in ``texts``. Returns one vector per input.

        We submit the entire batch to a single worker — encoding K texts
        in one ``model.encode`` call is materially faster than K separate
        encodes (vectorised matmul + tokenizer batching). Concurrency
        comes from many *callers* hitting the pool, not from splitting
        one batch across workers.
        """
        items = list(texts)
        if not items:
            return []
        ex = self._ensure_started()
        future = ex.submit(_worker_embed, items)
        return future.result()


# ---------------------------------------------------------------------- #
# Module-level singleton (used by ``embed_text_pool``)
# ---------------------------------------------------------------------- #


_DEFAULT_POOL: EmbedPool | None = None
_DEFAULT_POOL_LOCK = threading.Lock()
_CONFIGURED_WORKERS: int | None = None
_CONFIGURED_MODEL: str | None = None


def configure_pool(n_workers: int | None = None, model_name: str | None = None) -> None:
    """Pin the singleton's parameters before first use.

    No-ops if the singleton has already been built — call this at process
    startup (CLI boot, server entry point) before anything else triggers
    ``get_default_pool``.
    """
    global _CONFIGURED_WORKERS, _CONFIGURED_MODEL
    if n_workers is not None:
        _CONFIGURED_WORKERS = n_workers
    if model_name is not None:
        _CONFIGURED_MODEL = model_name


def get_default_pool() -> EmbedPool:
    """Return (and lazy-build) the module-level pool.

    Worker count resolves from, in order: ``configure_pool``, the
    ``MEMOIRS_EMBED_POOL_WORKERS`` env, then 4. Model name resolves from
    ``configure_pool`` then the configured ``EMBEDDING_MODEL``.
    """
    global _DEFAULT_POOL
    if _DEFAULT_POOL is not None:
        return _DEFAULT_POOL
    with _DEFAULT_POOL_LOCK:
        if _DEFAULT_POOL is not None:
            return _DEFAULT_POOL
        n = _CONFIGURED_WORKERS
        if n is None:
            try:
                n = int(os.environ.get("MEMOIRS_EMBED_POOL_WORKERS", "4"))
            except ValueError:
                n = 4
        n = max(1, n)
        model = _CONFIGURED_MODEL or EMBEDDING_MODEL
        _DEFAULT_POOL = EmbedPool(n_workers=n, model_name=model)
        return _DEFAULT_POOL


def shutdown_pool() -> None:
    """Tear down the module-level pool. Safe to call repeatedly."""
    global _DEFAULT_POOL
    with _DEFAULT_POOL_LOCK:
        pool = _DEFAULT_POOL
        _DEFAULT_POOL = None
    if pool is not None:
        pool.shutdown(wait=True)
