"""Layer 4 — embeddings + similarity search.

Single strategy: sentence-transformers for the embedding + sqlite-vec for the
ANN index. Both are part of the `[embeddings]` extra and are mandatory for
any call into this module — no silent fallbacks. If the user wants raw-only
ingestion they don't need to install them; but anything that asks for
similarity/dedup will fail loud and clear.

Performance helpers (P-perf):
  * ``embed_text_cached`` — LRU-cached query embeddings (memories are NOT
    cached — those are persisted to the DB explicitly via
    ``upsert_memory_embedding``).
  * ``should_use_dense`` — cheap heuristic to skip dense embedding for
    trivial queries (single keyword, etc.) so hybrid retrieval can fall
    back to BM25-only and avoid the model call entirely.
  * Optional ``fastembed`` ONNX backend (~2× faster on CPU) selected via
    ``MEMOIRS_EMBED_BACKEND=fastembed``. Falls back with a warning if the
    package is not installed.
"""
from __future__ import annotations

import array
import functools
import logging
import os
import struct

from ..config import EMBEDDING_DIM, EMBEDDING_MODEL
from ..db import MemoirsDB, utc_now


log = logging.getLogger("memoirs.embeddings")


class EmbeddingsUnavailable(RuntimeError):
    """Raised when [embeddings] extras are not installed."""


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_MODEL_SINGLETON = None


def _require_embedder():
    """Return the loaded sentence-transformers model, raising if missing."""
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is not None:
        return _MODEL_SINGLETON
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise EmbeddingsUnavailable(
            "sentence-transformers not installed. Run: pip install -e '.[embeddings]'"
        ) from e
    log.info("loading embedder %s", EMBEDDING_MODEL)
    _MODEL_SINGLETON = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL_SINGLETON


def _require_vec(db: MemoirsDB) -> None:
    """Ensure sqlite-vec extension is loaded and the vec0 virtual table exists."""
    if getattr(db, "_vec_loaded", False):
        return
    try:
        import sqlite_vec
    except ImportError as e:
        raise EmbeddingsUnavailable(
            "sqlite-vec not installed. Run: pip install -e '.[embeddings]'"
        ) from e
    db.conn.enable_load_extension(True)
    sqlite_vec.load(db.conn)
    db.conn.enable_load_extension(False)
    db.conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0("
        f"  memory_id TEXT PRIMARY KEY,"
        f"  embedding FLOAT[{EMBEDDING_DIM}]"
        f")"
    )
    db._vec_loaded = True


# ---------------------------------------------------------------------------
# BLOB helpers
# ---------------------------------------------------------------------------


def _pack(vec: list[float]) -> bytes:
    return array.array("f", vec).tobytes()


def _unpack(blob: bytes, dim: int) -> list[float]:
    if len(blob) != dim * 4:
        raise ValueError(f"embedding blob size {len(blob)} != dim*4 ({dim*4})")
    return list(struct.unpack(f"{dim}f", blob))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_text(text: str) -> list[float]:
    backend = _resolve_backend()
    if backend == "fastembed":
        encoder = _require_fastembed()
        if encoder is not None:
            vec = next(iter(encoder.embed([text])))
            out = list(map(float, vec))
            if len(out) != EMBEDDING_DIM:
                log.warning(
                    "fastembed dim mismatch (got=%d, expected=%d) — vectors "
                    "will not be compatible with vec_memories",
                    len(out), EMBEDDING_DIM,
                )
            return out
        # else: fastembed unavailable → fall through to sentence-transformers
    if backend == "process_pool":
        # Single dispatch branch added for GAP fix #4. Pool implementation
        # lives in ``embed_pool.py``; on failure we degrade to in-process
        # ST so a misconfigured pool doesn't take retrieval down.
        try:
            return embed_text_pool(text)
        except Exception as e:  # pragma: no cover - depends on host
            log.warning("process_pool embed failed (%s) — falling back to in-process ST", e)
    model = _require_embedder()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return list(map(float, vec))


def embed_text_pool(text: str) -> list[float]:
    """Embed ``text`` via the process-pool worker (bypasses the PyTorch GIL).

    Lazy: spawns the pool on first call. Use ``MEMOIRS_EMBED_POOL_WORKERS``
    to override the worker count (default 4). Pool size is also overridable
    by callers via ``embed_pool.configure_pool(n_workers=N)`` before the
    first embed.

    See ``memoirs.engine.embed_pool`` for the underlying machinery.
    """
    from .embed_pool import get_default_pool

    pool = get_default_pool()
    return pool.embed(text)


# ---------------------------------------------------------------------------
# Backend resolution + fastembed (opt-in)
# ---------------------------------------------------------------------------


_FASTEMBED_SINGLETON = None
_FASTEMBED_WARNED = False
# Models known to fastembed that match our 384-dim space. We try our
# configured ``EMBEDDING_MODEL`` first and fall back to a known-good default
# if fastembed doesn't know it.
_FASTEMBED_KNOWN = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "snowflake/snowflake-arctic-embed-xs": "snowflake/snowflake-arctic-embed-xs",
}
_FASTEMBED_DEFAULT = "snowflake/snowflake-arctic-embed-xs"


def _fastembed_importable() -> bool:
    """Cheap probe: is ``fastembed`` available without actually loading it?

    ``importlib.util.find_spec`` resolves the module's loader without
    triggering the import — keeps the cold-path fast and doesn't pollute
    ``sys.modules`` (so monkeypatched test setups still work).
    """
    import importlib.util
    try:
        return importlib.util.find_spec("fastembed") is not None
    except (ImportError, ValueError):
        return False


def _resolve_backend() -> str:
    """Return the configured embedding backend.

    Reads ``MEMOIRS_EMBED_BACKEND`` each call (cheap) so tests / runtime
    overrides take effect without restart. Resolution order:

      1. Explicit env var (``fastembed`` / ``process_pool`` /
         ``sentence-transformers`` / ``auto``).
      2. If env is empty and ``MEMOIRS_EMBED_AUTO=1`` (default off, opt-in
         to preserve backward-compat with existing pinned tests), prefer
         ``fastembed`` when importable. The CLI ``--embed-fastembed`` flag
         and the bench harness flip this on.
      3. Fall back to ``sentence-transformers``.

    Unknown values fall back to the default with a warning.

    Background: under concurrent load the PyTorch-backed sentence-
    transformers serializes on the GIL (50 workers → 2 rps). ONNX
    fastembed releases the GIL → 3-5× sustained throughput. See GAP fix #4.
    """
    raw = (os.environ.get("MEMOIRS_EMBED_BACKEND") or "").strip().lower()
    if not raw:
        # Opt-out via MEMOIRS_EMBED_AUTO=0: pin the legacy ST default. This
        # keeps existing pinned-default tests honest (they explicitly set
        # MEMOIRS_EMBED_AUTO=0 in their fixtures) while production picks up
        # the GIL-free path automatically.
        if (os.environ.get("MEMOIRS_EMBED_AUTO") or "1").strip() == "0":
            return "sentence-transformers"
        if _fastembed_importable():
            return "fastembed"
        return "sentence-transformers"
    if raw == "auto":
        if _fastembed_importable():
            return "fastembed"
        return "sentence-transformers"
    if raw in {"sentence-transformers", "sentence_transformers", "st", "default"}:
        return "sentence-transformers"
    if raw == "fastembed":
        return "fastembed"
    if raw in {"process_pool", "process-pool", "pool"}:
        return "process_pool"
    log.warning("unknown MEMOIRS_EMBED_BACKEND=%r — using sentence-transformers", raw)
    return "sentence-transformers"


def _require_fastembed():
    """Load the fastembed encoder lazily. Returns ``None`` if unavailable."""
    global _FASTEMBED_SINGLETON, _FASTEMBED_WARNED
    if _FASTEMBED_SINGLETON is not None:
        return _FASTEMBED_SINGLETON
    try:
        from fastembed import TextEmbedding  # type: ignore[import-not-found]
    except ImportError:
        if not _FASTEMBED_WARNED:
            log.warning(
                "MEMOIRS_EMBED_BACKEND=fastembed but fastembed is not "
                "installed — falling back to sentence-transformers. Install "
                "with: pip install -e '.[embeddings_fast]'"
            )
            _FASTEMBED_WARNED = True
        return None
    name = _FASTEMBED_KNOWN.get(EMBEDDING_MODEL, _FASTEMBED_DEFAULT)
    if name != EMBEDDING_MODEL:
        log.info(
            "fastembed: %r not in known list, using %r instead", EMBEDDING_MODEL, name
        )
    log.info("loading fastembed encoder %s", name)
    _FASTEMBED_SINGLETON = TextEmbedding(model_name=name)
    return _FASTEMBED_SINGLETON


# ---------------------------------------------------------------------------
# Query embedding cache + skip-dense heuristic
# ---------------------------------------------------------------------------


def _cache_key_model() -> str:
    """Cache key component: model identity, so changing model invalidates."""
    return f"{_resolve_backend()}::{EMBEDDING_MODEL}"


@functools.lru_cache(maxsize=1024)
def _embed_cached_inner(text: str, model_key: str) -> tuple[float, ...]:
    """Internal LRU layer. Tuples are hashable + immutable — safer to cache."""
    # ``model_key`` is unused at compute time; it's just part of the cache
    # key so a model swap invalidates entries automatically.
    del model_key
    return tuple(embed_text(text))


def embed_text_cached(text: str) -> list[float]:
    """LRU-cached wrapper around ``embed_text`` for ephemeral query embeddings.

    Cache is keyed by ``(text, backend::model)`` so a backend or model swap
    invalidates entries. Memories are persisted via
    ``upsert_memory_embedding`` and intentionally NOT cached here — only
    transient retrieval queries benefit.
    """
    return list(_embed_cached_inner(text, _cache_key_model()))


def clear_embed_cache() -> None:
    """Drop all cached query embeddings. Useful for tests + model swaps."""
    _embed_cached_inner.cache_clear()


def embed_cache_info():
    """Expose ``functools.lru_cache`` stats (hits/misses/currsize) for ops."""
    return _embed_cached_inner.cache_info()


# Stop-words that don't carry retrieval signal on their own. Kept tiny and
# inline — no NLTK / spaCy dep just for this.
_TRIVIAL_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "up", "down",
    "and", "or", "but", "not", "no", "if", "then", "else", "when", "where",
    "why", "how", "what", "who", "which", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their",
    "do", "does", "did", "have", "has", "had", "can", "could", "should",
    "would", "will", "shall", "may", "might", "must",
})


def should_use_dense(query: str) -> bool:
    """Heuristic: is this query worth a (potentially expensive) dense embed?

    Returns ``False`` for trivial queries — single keywords, only stop-words,
    or extremely short content-bearing strings. The caller (typically
    ``_retrieve_candidates``) can then short-circuit to BM25-only.

    Rules (any one triggers ``False``):
      * fewer than 3 whitespace-separated tokens, AND
      * stripped of stop-words, fewer than 8 non-whitespace characters of
        content remain.

    Examples::

        should_use_dense("AI")                                 # False
        should_use_dense("the cat")                            # False
        should_use_dense("memoirs MCP server architecture")    # True
    """
    if not query or not query.strip():
        return False
    tokens = query.split()
    if len(tokens) >= 3:
        return True
    # 1-2 tokens: check that the non-stopword content is meaty enough.
    content = "".join(
        t for t in (tok.strip().lower() for tok in tokens) if t and t not in _TRIVIAL_STOPWORDS
    )
    return len(content) >= 8


def upsert_memory_embedding(db: MemoirsDB, memory_id: str, text: str) -> None:
    """Embed `text` and upsert into both `memory_embeddings` and `vec_memories`."""
    _require_vec(db)
    vec = embed_text(text)
    blob = _pack(vec)
    db.conn.execute(
        """
        INSERT INTO memory_embeddings (memory_id, dim, embedding, model, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(memory_id) DO UPDATE SET
            dim = excluded.dim,
            embedding = excluded.embedding,
            model = excluded.model,
            created_at = excluded.created_at
        """,
        (memory_id, len(vec), blob, EMBEDDING_MODEL, utc_now()),
    )
    # vec0 virtual tables do NOT support ON CONFLICT — emulate upsert via DELETE+INSERT
    db.conn.execute("DELETE FROM vec_memories WHERE memory_id = ?", (memory_id,))
    db.conn.execute(
        "INSERT INTO vec_memories(memory_id, embedding) VALUES (?, ?)",
        (memory_id, blob),
    )
    db.conn.commit()


def sync_vec_index(db: MemoirsDB) -> int:
    """Mirror any rows in `memory_embeddings` not yet in `vec_memories`. Returns inserted count."""
    _require_vec(db)
    rows = db.conn.execute(
        "SELECT memory_id, dim, embedding FROM memory_embeddings "
        "WHERE memory_id NOT IN (SELECT memory_id FROM vec_memories)"
    ).fetchall()
    inserted = 0
    for r in rows:
        if r["dim"] != EMBEDDING_DIM:
            continue
        db.conn.execute(
            "INSERT INTO vec_memories(memory_id, embedding) VALUES (?, ?)",
            (r["memory_id"], r["embedding"]),
        )
        inserted += 1
    if inserted:
        db.conn.commit()
        log.info("synced %d embeddings into vec_memories", inserted)
    return inserted


def search_similar_memories(
    db: MemoirsDB,
    query: str,
    top_k: int = 10,
    *,
    as_of: str | None = None,
) -> list[dict]:
    """ANN KNN over vec0.

    Filters by temporal validity. Default = "now": only currently-valid, non-archived
    memories. With `as_of=<ISO timestamp>`, returns the memory state as it would have
    been seen at that moment (time-travel queries).
    """
    _require_vec(db)
    # Use the LRU-cached path — repeated retrieval queries are common and
    # the embed is the dominant per-call cost.
    qvec = embed_text_cached(query)
    qblob = _pack(qvec)
    ts = as_of or utc_now()
    if as_of is None:
        # Live mode: skip archived rows entirely.
        sql = """
            SELECT m.id, m.type, m.content, m.importance, m.confidence, m.score,
                   m.usage_count, m.last_used_at, m.valid_from, m.valid_to, m.archived_at,
                   v.distance
            FROM vec_memories v
            JOIN memories m ON m.id = v.memory_id
            WHERE v.embedding MATCH ?
              AND m.archived_at IS NULL
              AND (m.valid_to IS NULL OR m.valid_to >= ?)
              AND k = ?
            ORDER BY v.distance
        """
        params = (qblob, ts, top_k * 2)
    else:
        # Time-travel: include archived rows but only those valid at `ts`.
        # A memory is valid at ts iff:
        #   valid_from is NULL or <= ts  AND  valid_to is NULL or > ts
        # AND it wasn't archived before ts (archived_at NULL or > ts).
        sql = """
            SELECT m.id, m.type, m.content, m.importance, m.confidence, m.score,
                   m.usage_count, m.last_used_at, m.valid_from, m.valid_to, m.archived_at,
                   v.distance
            FROM vec_memories v
            JOIN memories m ON m.id = v.memory_id
            WHERE v.embedding MATCH ?
              AND (m.valid_from IS NULL OR m.valid_from <= ?)
              AND (m.valid_to   IS NULL OR m.valid_to   >  ?)
              AND (m.archived_at IS NULL OR m.archived_at > ?)
              AND k = ?
            ORDER BY v.distance
        """
        params = (qblob, ts, ts, ts, top_k * 2)
    rows = db.conn.execute(sql, params).fetchall()
    out: list[dict] = []
    for r in rows[:top_k]:
        d = dict(r)
        # vec0 returns L2 distance for unit-normalized vectors;
        # cosine_similarity ≈ 1 - dist^2 / 2 for unit vectors.
        dist = float(d.pop("distance", 0.0))
        d["similarity"] = round(max(0.0, 1.0 - (dist ** 2) / 2.0), 4)
        out.append(d)
    return out


def find_semantic_duplicates(db: MemoirsDB, content: str, threshold: float = 0.92) -> list[dict]:
    """Existing memories that look like duplicates of `content` (cosine ≥ threshold)."""
    return [m for m in search_similar_memories(db, content, top_k=5) if m["similarity"] >= threshold]
