"""Adapter for **Memori** (GibsonAI's SQL-native memory engine, ``pip install memori``).

Memori 3.x is a SQL-backed memory store. Its public ergonomic API
(``Memori.augmentation.enqueue``) routes ingestion through an LLM
provider that extracts facts from a conversation pair — useful for
agent runtimes but not what we need for a retrieval bench.

For a fair head-to-head the adapter:

  1. Builds a Memori instance pointing at a local SQLite file (no
     ``MEMORI_API_KEY`` required — cloud mode auto-disables when a
     ``conn`` factory is supplied).
  2. Calls ``storage.build()`` to run the schema migrations.
  3. Writes each ``BenchMemory.content`` directly into
     ``memori_entity_fact`` via the SQLite driver, using the bundled
     sentence-transformers embedder (``all-MiniLM-L6-v2`` by default,
     384 dims). Bench IDs map to fact IDs via a side dict.
  4. Retrieves with ``Memori.recall(query, limit=k)`` which runs
     Memori's hybrid lexical + dense search against the same table.

Skips cleanly if ``memori`` is not installed. The ``OPENAI_API_KEY`` /
Ollama path is honoured but Memori's *local* mode does its own
embedding via sentence-transformers, so the adapter works offline as
long as the embedding model is downloadable.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from scripts.adapters._ollama import (
    apply_ollama_env,
    ollama_install_hint,
    ollama_is_up,
    use_ollama_requested,
)
from scripts.adapters.base import AdapterStatus, EngineAdapter
from scripts.bench_vs_others_dataset import BenchMemory, BenchQuery

log = logging.getLogger("bench.adapters.memori")


class MemoriAdapter(EngineAdapter):
    name = "memori"

    def __init__(self, *, db_path: Optional[Path] = None,
                 entity_id: str = "bench-user", **_unused) -> None:
        super().__init__()
        self._db_path: Optional[Path] = None
        self._tempdir: Optional[tempfile.TemporaryDirectory] = None
        self._memori = None
        self._entity_pk: Optional[int] = None
        self._fact_id_to_bench: dict[int, str] = {}
        self._content_to_bench: dict[str, str] = {}
        self._entity_id = entity_id
        self._use_ollama = use_ollama_requested()

        # Mirror the env-shim contract even though Memori's local
        # embedder doesn't read OpenAI env. Future Memori releases may
        # let users swap in an OpenAI-compatible embedder; we're staying
        # consistent with the other adapters.
        if self._use_ollama:
            if not ollama_is_up():
                self.status = AdapterStatus(ok=False, reason=ollama_install_hint())
                return
            apply_ollama_env()

        try:
            from memori import Memori  # noqa: F401
        except Exception as e:
            self.status = AdapterStatus(
                ok=False,
                reason=f"memori not installed (pip install memori): {e!r}",
            )
            return

        # Pick a writable SQLite path. Default: temp dir (cleaned in
        # shutdown). Caller can override with an explicit ``db_path``.
        if db_path is not None:
            self._db_path = Path(db_path)
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._tempdir = tempfile.TemporaryDirectory(prefix="memori-bench-")
            self._db_path = Path(self._tempdir.name) / "memori.db"

        try:
            from memori import Memori
            db_path_str = str(self._db_path)

            def _conn_factory():
                import sqlite3
                return sqlite3.connect(db_path_str)

            self._memori = Memori(conn=_conn_factory).attribution(
                entity_id=self._entity_id,
            ).new_session()
            self._memori.config.storage.build()
        except Exception as e:
            self.status = AdapterStatus(
                ok=False, reason=f"Memori init failed: {e!r}",
            )
            self._cleanup_tempdir()
            self._memori = None
            return

        self.status.meta["backend"] = "memori-sqlite"
        self.status.meta["db_path"] = str(self._db_path)
        self.status.meta["use_ollama"] = self._use_ollama

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def add_memories(self, memories: Sequence[BenchMemory]) -> None:
        if not self.status.ok or self._memori is None:
            raise RuntimeError(f"adapter not ready: {self.status.reason}")
        try:
            from memori import embed_texts
        except Exception as e:
            self.status = AdapterStatus(
                ok=False, reason=f"memori.embed_texts unavailable: {e!r}",
            )
            return

        driver = self._memori.config.storage.driver
        # Create the entity row and remember its primary key — entity_fact
        # rows are FKed to it.
        try:
            self._entity_pk = driver.entity.create(external_id=self._entity_id)
        except Exception as e:
            self.status = AdapterStatus(
                ok=False, reason=f"memori entity.create failed: {e!r}",
            )
            return

        contents = [m.content for m in memories]
        for m, c in zip(memories, contents):
            self._content_to_bench[c] = m.id

        try:
            embeddings = embed_texts(
                contents, model=self._memori.config.embeddings.model,
            )
        except Exception as e:
            log.warning("memori embed_texts failed (continuing without dense): %s", e)
            embeddings = [[] for _ in contents]

        # The driver's create() inserts each fact with autoincrement
        # IDs. Memori doesn't return the IDs from `create`, so we read
        # them back via a SELECT against the just-written rows.
        try:
            driver.entity_fact.create(
                entity_id=self._entity_pk,
                facts=contents,
                fact_embeddings=embeddings,
            )
        except Exception as e:
            self.status = AdapterStatus(
                ok=False, reason=f"memori entity_fact.create failed: {e!r}",
            )
            return

        # Build the fact_id -> bench_id index. We pull every fact for
        # the entity in a single query — cheap for 80-row benches.
        try:
            conn = driver.entity_fact.conn
            cur = conn.execute(
                "SELECT id, content FROM memori_entity_fact WHERE entity_id = ?",
                (self._entity_pk,),
            )
            for fact_id, content in cur.fetchall():
                bid = self._content_to_bench.get(content)
                if bid:
                    self._fact_id_to_bench[int(fact_id)] = bid
        except Exception as e:
            log.warning("memori fact->bench mapping failed: %s", e)

    def query(self, q: BenchQuery, top_k: int = 10) -> list[str]:
        if not self.status.ok or self._memori is None:
            return []
        try:
            results = self._memori.recall(q.query, limit=top_k)
        except Exception as e:
            log.warning("memori recall failed for %r: %s", q.query, e)
            return []
        # Local-mode recall returns ``list[FactSearchResult]``; cloud
        # mode returns a CloudRecallResponse-like dict. Handle both.
        items = results
        if isinstance(results, dict):
            items = results.get("facts") or results.get("results") or []

        out: list[str] = []
        seen: set[str] = set()
        for item in (items or [])[:top_k]:
            fact_id = None
            content = None
            if isinstance(item, dict):
                fact_id = item.get("id")
                content = item.get("content") or item.get("text")
            else:
                fact_id = getattr(item, "id", None)
                content = getattr(item, "content", None)
            bid = None
            if fact_id is not None:
                bid = self._fact_id_to_bench.get(int(fact_id))
            if bid is None and content:
                bid = self._content_to_bench.get(str(content))
            if bid and bid not in seen:
                out.append(bid)
                seen.add(bid)
        return out

    def shutdown(self) -> None:
        try:
            if self._memori is not None:
                close = getattr(self._memori, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
        finally:
            self._memori = None
            self._cleanup_tempdir()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cleanup_tempdir(self) -> None:
        if self._tempdir is not None:
            try:
                self._tempdir.cleanup()
            except Exception:
                pass
            self._tempdir = None


__all__ = ["MemoriAdapter"]
