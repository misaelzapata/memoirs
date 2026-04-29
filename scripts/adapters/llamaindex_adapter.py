"""Adapter for **LlamaIndex Memory** (``pip install llama-index llama-index-llms-openai``).

LlamaIndex ships several memory abstractions; the one closest to the
"flat retrieval" shape every other other engine exposes is
``VectorStoreIndex`` over a list of ``Document`` objects. That's also
the path most people take in production for chat-memory retrieval, so
we get a fair head-to-head:

  1. Each ``BenchMemory`` becomes one ``Document(text=content,
     metadata={"bench_id": ...})``.
  2. ``VectorStoreIndex.from_documents(...)`` builds the in-memory ANN
     index (uses OpenAI embeddings by default; Ollama via the shared
     env shim when ``MEMOIRS_USE_OLLAMA=on``).
  3. ``query`` calls ``index.as_retriever(similarity_top_k=k).retrieve``
     and pulls the bench IDs back out of node metadata.

Skips cleanly when ``llama-index-core`` isn't installed or when
``OPENAI_API_KEY`` is missing in cloud mode.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

from scripts.adapters._ollama import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_MODEL,
    OLLAMA_OPENAI_BASE_URL,
    apply_ollama_env,
    ollama_install_hint,
    ollama_is_up,
    use_ollama_requested,
)
from scripts.adapters.base import AdapterStatus, EngineAdapter
from scripts.bench_vs_others_dataset import BenchMemory, BenchQuery

log = logging.getLogger("bench.adapters.llamaindex")


class LlamaIndexAdapter(EngineAdapter):
    name = "llamaindex"

    def __init__(self, **_unused) -> None:
        super().__init__()
        self._index = None
        self._retriever = None
        self._use_ollama = use_ollama_requested()

        # Env shim must precede the LlamaIndex import — its OpenAI
        # provider grabs ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` once
        # at module init.
        if self._use_ollama:
            if not ollama_is_up():
                self.status = AdapterStatus(ok=False, reason=ollama_install_hint())
                return
            apply_ollama_env()
        elif not os.environ.get("OPENAI_API_KEY"):
            self.status = AdapterStatus(ok=False, reason="OPENAI_API_KEY missing")
            return

        try:
            from llama_index.core import Settings  # noqa: F401
            from llama_index.core import VectorStoreIndex, Document  # noqa: F401
        except Exception as e:
            self.status = AdapterStatus(
                ok=False,
                reason=(
                    "llama-index-core not installed (pip install llama-index "
                    f"llama-index-llms-openai): {e!r}"
                ),
            )
            return

        # Configure the global ``Settings`` so ``from_documents`` uses
        # the right embedder/LLM. We swallow import errors per provider
        # because some installs only ship the core package.
        try:
            from llama_index.llms.openai import OpenAI as LIOpenAI
            from llama_index.embeddings.openai import OpenAIEmbedding
            from llama_index.core import Settings

            embed_model_name = (
                os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBED_MODEL)
                if self._use_ollama
                else "text-embedding-3-small"
            )
            llm_model_name = (
                os.environ.get("OPENAI_MODEL", DEFAULT_LLM_MODEL)
                if self._use_ollama
                else "gpt-4o-mini"
            )
            base_url = os.environ.get("OPENAI_BASE_URL")
            api_key = os.environ.get("OPENAI_API_KEY", "ollama")

            llm_kwargs = {"model": llm_model_name, "api_key": api_key}
            embed_kwargs = {"model": embed_model_name, "api_key": api_key}
            if base_url:
                llm_kwargs["api_base"] = base_url
                embed_kwargs["api_base"] = base_url

            Settings.llm = LIOpenAI(**llm_kwargs)
            Settings.embed_model = OpenAIEmbedding(**embed_kwargs)
        except Exception as e:
            # Fall back to library defaults; if the user has OPENAI_API_KEY
            # set, llama-index will pick the cloud defaults itself. We log
            # but don't fail because the global Settings may already be
            # configured by another adapter in the same process.
            log.info("llamaindex Settings setup skipped: %s", e)

        self.status.meta["backend"] = "llama-index-vectorstore"
        self.status.meta["use_ollama"] = self._use_ollama

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def add_memories(self, memories: Sequence[BenchMemory]) -> None:
        if not self.status.ok:
            raise RuntimeError(f"adapter not ready: {self.status.reason}")
        from llama_index.core import Document, VectorStoreIndex

        docs = [
            Document(
                text=m.content,
                metadata={"bench_id": m.id, "type": m.type},
                # Don't let llama-index leak metadata into the embedded
                # text — the bench_id is opaque and would skew retrieval.
                excluded_embed_metadata_keys=["bench_id", "type"],
                excluded_llm_metadata_keys=["bench_id", "type"],
            )
            for m in memories
        ]
        try:
            self._index = VectorStoreIndex.from_documents(docs)
        except Exception as e:
            self.status = AdapterStatus(
                ok=False, reason=f"VectorStoreIndex.from_documents failed: {e!r}",
            )
            self._index = None
            return
        # Build a retriever once so per-query overhead is just the query
        # embedding + ANN lookup.
        self._retriever = self._index.as_retriever(similarity_top_k=10)

    def query(self, q: BenchQuery, top_k: int = 10) -> list[str]:
        if not self.status.ok or self._index is None:
            return []
        # Replace the default retriever if the caller asked for a
        # different ``top_k`` than the one we configured at ingest time.
        retriever = self._retriever
        if retriever is None or getattr(retriever, "similarity_top_k", None) != top_k:
            try:
                retriever = self._index.as_retriever(similarity_top_k=top_k)
                self._retriever = retriever
            except Exception as e:
                log.warning("llamaindex retriever rebuild failed: %s", e)
                return []
        try:
            nodes = retriever.retrieve(q.query)
        except Exception as e:
            log.warning("llamaindex retrieve failed for %r: %s", q.query, e)
            return []
        out: list[str] = []
        for node in nodes[:top_k]:
            md = {}
            # NodeWithScore.node.metadata is the canonical place; older
            # versions exposed ``metadata`` directly on the wrapper.
            inner = getattr(node, "node", node)
            md = getattr(inner, "metadata", None) or {}
            bid = md.get("bench_id") if isinstance(md, dict) else None
            if bid:
                out.append(str(bid))
        return out

    def shutdown(self) -> None:
        self._retriever = None
        self._index = None


__all__ = ["LlamaIndexAdapter"]
