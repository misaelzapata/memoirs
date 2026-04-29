"""Adapter for **LangMem** (LangChain memory SDK, ``pip install langmem``).

LangMem 0.0.x is built on top of LangGraph's ``BaseStore`` abstraction:
the high-level ``create_memory_store_manager`` runs an LLM to extract
durable facts from a conversation and writes them to the store. For a
retrieval bench we don't need the LLM-driven extraction — we want to
ingest each ``BenchMemory.content`` verbatim and then ask the store to
rank by semantic similarity.

So this adapter:

  1. Wires LangMem's underlying ``InMemoryStore`` with an
     ``IndexConfig`` that points at an OpenAI-compatible embedding
     endpoint (cloud OpenAI by default, Ollama via the shared shim when
     ``MEMOIRS_USE_OLLAMA=on``).
  2. ``add_memories`` writes ``{"content": ..., "bench_id": ...}`` per
     memory via ``store.put`` so we can recover IDs after retrieval.
  3. ``query`` calls ``store.search(..., query=q, limit=top_k)`` which
     embeds the query and runs ANN against the stored vectors.

Skips cleanly when ``langmem``/``langgraph`` aren't installed or when
``OPENAI_API_KEY`` is missing in cloud mode.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

from scripts.adapters._ollama import (
    DEFAULT_EMBED_MODEL,
    OLLAMA_OPENAI_BASE_URL,
    apply_ollama_env,
    ollama_install_hint,
    ollama_is_up,
    use_ollama_requested,
)
from scripts.adapters.base import AdapterStatus, EngineAdapter
from scripts.bench_vs_others_dataset import BenchMemory, BenchQuery

log = logging.getLogger("bench.adapters.langmem")


class LangMemAdapter(EngineAdapter):
    name = "langmem"

    def __init__(self, *, namespace: tuple[str, ...] = ("memories", "bench-user"),
                 **_unused) -> None:
        super().__init__()
        self._namespace = namespace
        self._store = None
        self._id_map: dict[str, str] = {}  # bench_id -> langmem key
        self._use_ollama = use_ollama_requested()

        # Apply env shim BEFORE importing langchain_openai so the
        # embeddings client picks up the right base URL.
        if self._use_ollama:
            if not ollama_is_up():
                self.status = AdapterStatus(ok=False, reason=ollama_install_hint())
                return
            apply_ollama_env()
        elif not os.environ.get("OPENAI_API_KEY"):
            self.status = AdapterStatus(ok=False, reason="OPENAI_API_KEY missing")
            return

        try:
            import langmem  # noqa: F401  (presence check)
            from langgraph.store.memory import InMemoryStore
        except Exception as e:
            self.status = AdapterStatus(
                ok=False, reason=f"langmem not installed (pip install langmem): {e!r}",
            )
            return

        # Pick an embedding model. In Ollama mode we rely on
        # ``EMBEDDING_MODEL`` from the shim (default nomic-embed-text,
        # 768 dims). In cloud mode we use ``text-embedding-3-small`` at
        # 1536 dims for parity with the Mem0 adapter.
        if self._use_ollama:
            embed_model = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBED_MODEL)
            embed_base = os.environ.get("OPENAI_BASE_URL", OLLAMA_OPENAI_BASE_URL)
            embed_key = os.environ.get("OPENAI_API_KEY", "ollama")
            dims = 768
        else:
            embed_model = "text-embedding-3-small"
            embed_base = os.environ.get("OPENAI_BASE_URL")  # may be None
            embed_key = os.environ.get("OPENAI_API_KEY", "")
            dims = 1536

        try:
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(
                model=embed_model,
                api_key=embed_key,
                **({"base_url": embed_base} if embed_base else {}),
            )
        except Exception as e:
            self.status = AdapterStatus(
                ok=False, reason=f"langchain_openai unavailable: {e!r}",
            )
            return

        try:
            self._store = InMemoryStore(
                index={"dims": dims, "embed": embeddings, "fields": ["content"]},
            )
        except Exception as e:
            self.status = AdapterStatus(
                ok=False, reason=f"InMemoryStore init failed: {e!r}",
            )
            return

        self.status.meta["backend"] = "langmem-inmemorystore"
        self.status.meta["embed_model"] = embed_model
        self.status.meta["use_ollama"] = self._use_ollama

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def add_memories(self, memories: Sequence[BenchMemory]) -> None:
        if not self.status.ok or self._store is None:
            raise RuntimeError(f"adapter not ready: {self.status.reason}")
        for m in memories:
            try:
                self._store.put(
                    self._namespace,
                    key=m.id,
                    value={"content": m.content, "bench_id": m.id, "type": m.type},
                )
                self._id_map[m.id] = m.id
            except Exception as e:
                log.warning("langmem put failed for %s: %s", m.id, e)

    def query(self, q: BenchQuery, top_k: int = 10) -> list[str]:
        if not self.status.ok or self._store is None:
            raise RuntimeError(f"adapter not ready: {self.status.reason}")
        try:
            results = self._store.search(self._namespace, query=q.query, limit=top_k)
        except Exception as e:
            log.warning("langmem search failed for %r: %s", q.query, e)
            return []
        out: list[str] = []
        for item in results[:top_k]:
            # SearchItem exposes ``key`` and ``value``. The bench_id is
            # the same as the key but we also embed it in value as a
            # belt-and-braces fallback.
            value = getattr(item, "value", None) or {}
            bid = (value.get("bench_id") if isinstance(value, dict) else None) \
                or getattr(item, "key", None)
            if bid:
                out.append(str(bid))
        return out

    def shutdown(self) -> None:
        # InMemoryStore has no explicit close; drop the reference.
        self._store = None
        self._id_map.clear()


__all__ = ["LangMemAdapter"]
