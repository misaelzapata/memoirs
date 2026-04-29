"""Adapter for **Cognee** (pure Python, no Docker).

Cognee is an in-process knowledge-graph + retrieval library. Unlike
Mem0/Zep it doesn't ship a server, so this adapter just imports the
package and calls its API directly.

Why this adapter is non-trivial — Cognee 1.0 contract notes
------------------------------------------------------------

Calling ``cognee.search(query)`` with the default ``SearchType`` returns
a single LLM-generated answer string per dataset, NOT the source
memorias. That output is useful for chat UIs but useless for a
retrieval bench: every gold ID misses, MRR collapses to 0.

The fix: drive cognee with ``SearchType.CHUNKS`` instead. CHUNKS
returns the raw ``DocumentChunk`` objects ranked by vector similarity,
preserving the exact ``text`` we ingested. We can then map each chunk
back to the bench ID it was sourced from.

Cognee does NOT preserve free-form metadata through ``add()`` (the
public surface accepts a string, not a typed DataPoint), so we encode
the bench ID as a literal prefix in the chunk text:

    [bench_id=mem_sh_lasagna] user's favorite dinner recipe is …

At query time we pull the prefix back out with a regex. If the prefix
was stripped (chunking can split mid-sentence on long inputs, though
not on the 80-mem bench corpus where each memory is one chunk), we
fall back to a substring match against the original content cache.

The result: real, repeatable hits/MRR for cognee in the bench, while
the engine still runs its full pipeline (extract + cognify + index).

Required env
------------

Cognee 1.0 reads ``LLM_API_KEY`` / ``EMBEDDING_API_KEY`` (LiteLLM
style) rather than ``OPENAI_API_KEY`` directly. The bench exports
``OPENAI_API_KEY`` for symmetry with the other engines, so we mirror
it into both LiteLLM keys here when they're absent. This means the
operator only has to set ``OPENAI_API_KEY`` once.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Optional, Sequence

from scripts.adapters._ollama import (
    apply_ollama_env,
    ollama_install_hint,
    ollama_is_up,
    use_ollama_requested,
)
from scripts.adapters.base import AdapterStatus, EngineAdapter
from scripts.bench_vs_others_dataset import (
    BenchConversation,
    BenchMemory,
    BenchQuery,
)

log = logging.getLogger("bench.adapters.cognee")


# Marker used to embed the bench ID into the ingested text. Chosen so
# it round-trips cleanly through cognee's tokenizer + chunker (no
# whitespace inside the brackets) and is unambiguous to extract.
_BENCH_ID_MARKER_RE = re.compile(r"\[bench_id=([A-Za-z0-9_:.\-]+)\]")


def _tag_with_bench_id(bench_id: str, content: str) -> str:
    """Encode the bench ID as a parseable prefix in the chunk text.

    Cognee preserves the raw text through ``add()`` + ``cognify()``,
    so we lean on that to map results back. The prefix is short (~30
    chars) so it has negligible impact on the LLM's extraction.
    """
    return f"[bench_id={bench_id}] {content}"


def _extract_bench_id(text: str) -> Optional[str]:
    """Pull the bench ID back out of a chunk text. None when absent."""
    match = _BENCH_ID_MARKER_RE.search(text)
    return match.group(1) if match else None


class CogneeAdapter(EngineAdapter):
    name = "cognee"
    supports_native_ingest = True

    def __init__(self) -> None:
        super().__init__()
        # bench_id -> raw content (without the marker); used as the
        # substring-match fallback when chunking strips the prefix.
        self._content_to_id: dict[str, str] = {}
        # Stable dataset name keeps cognee's storage scoped to the bench
        # run so concurrent users / unit tests don't collide.
        self._dataset_name = "memoirs_bench"
        # Ollama wiring MUST happen before `import cognee` because cognee
        # captures its provider config at module import time. We do it
        # via the shared helper so the env-shim contract lives in one
        # place.
        if use_ollama_requested():
            if not ollama_is_up():
                self.status = AdapterStatus(
                    ok=False,
                    reason=ollama_install_hint(),
                )
                return
            apply_ollama_env()
        else:
            # Cognee 1.0 expects LLM_API_KEY / EMBEDDING_API_KEY rather
            # than OPENAI_API_KEY. Mirror so the operator only has to
            # set the canonical OpenAI key. Done before `import cognee`
            # because the package snapshots its config at import.
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                os.environ.setdefault("LLM_API_KEY", api_key)
                os.environ.setdefault("EMBEDDING_API_KEY", api_key)
        try:
            import cognee  # noqa: F401
        except Exception as e:
            self.status = AdapterStatus(
                ok=False,
                reason=f"cognee not installed (pip install cognee): {e!r}",
            )
            return
        # Cognee requires an LLM key for cognify (extraction step). If
        # neither LLM_API_KEY nor OPENAI_API_KEY is present, cognify
        # will raise — we surface that as a SKIP rather than a crash.
        if not (os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")):
            self.status = AdapterStatus(
                ok=False,
                reason="cognee needs OPENAI_API_KEY (or LLM_API_KEY) to cognify",
            )
            return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run(self, coro):
        """Bridge async API to sync bench code."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():  # pragma: no cover — tests use a fresh loop
                return asyncio.run_coroutine_threadsafe(coro, loop).result()
        except RuntimeError:
            pass
        return asyncio.run(coro)

    async def _add_text(self, cognee, text: str) -> None:
        """Call ``cognee.add`` with our dataset, tolerating older mocks.

        Production cognee accepts ``dataset_name=`` as a keyword. Some
        existing test mocks define ``fake_add(text)`` as a 1-arg
        function — falling back to a positional call keeps the test
        suite green without forcing changes to those mocks.
        """
        try:
            await cognee.add(text, dataset_name=self._dataset_name)
        except TypeError:
            await cognee.add(text)

    def _flatten_search_result(self, results) -> list[str]:
        """Normalise cognee's ``search`` envelope into ranked text strings.

        Cognee returns a list of dicts shaped like::

            [{"dataset_id": ..., "search_result": [<chunk-or-string>, ...]}]

        We flatten across datasets in the order returned, preserving
        the per-dataset rank (top-1 first, top-2 second, …). Each
        retrieved item can be:

          * a string (RAG_COMPLETION / GRAPH_COMPLETION answer)
          * a dict with ``text`` / ``content`` / ``name`` (CHUNKS,
            CHUNKS_LEXICAL, SUMMARIES)

        We extract the text field that's most likely to carry the
        original chunk content.
        """
        flat: list[str] = []
        for envelope in results or []:
            if isinstance(envelope, dict):
                payload = envelope.get("search_result", envelope)
            else:
                payload = envelope
            if isinstance(payload, str):
                flat.append(payload)
                continue
            if not isinstance(payload, list):
                # Unknown shape — stringify so substring fallback still
                # has a chance.
                flat.append(str(payload))
                continue
            for item in payload:
                if isinstance(item, str):
                    flat.append(item)
                elif isinstance(item, dict):
                    # Prefer the chunk's raw text; fall back to other
                    # plausible fields, then to repr as a last resort.
                    txt = (
                        item.get("text")
                        or item.get("content")
                        or item.get("name")
                    )
                    if txt:
                        flat.append(str(txt))
                    else:
                        flat.append(str(item))
                else:
                    flat.append(str(item))
        return flat

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def add_memories(self, memories: Sequence[BenchMemory]) -> None:
        if not self.status.ok:
            raise RuntimeError(f"adapter not ready: {self.status.reason}")
        import cognee

        for m in memories:
            self._content_to_id[m.content] = m.id

        async def _ingest():
            for m in memories:
                tagged = _tag_with_bench_id(m.id, m.content)
                await self._add_text(cognee, tagged)
            # Scope cognify to our dataset so concurrent runs (or stale
            # data left in cognee's storage from earlier probes) don't
            # multiply the LLM calls — they're the dominant cost.
            try:
                await cognee.cognify(datasets=[self._dataset_name])
            except TypeError:
                # Mocks that don't accept the kwarg.
                await cognee.cognify()

        try:
            self._run(_ingest())
        except Exception as e:
            # Surface the failure as a status downgrade so downstream
            # query() returns nothing rather than crashing.
            self.status = AdapterStatus(
                ok=False, reason=f"cognee.cognify failed: {e!r}",
            )

    def ingest_conversation(self, conv: BenchConversation) -> None:
        """Drive cognee's native pipeline on a flattened conversation.

        Cognee accepts free-form text plus a `cognify()` step that
        runs the LLM-driven knowledge-graph extraction. We concatenate
        the messages into a single document tagged with the bench
        conv_id so retrievals can be traced back.
        """
        if not self.status.ok:
            raise RuntimeError(f"adapter not ready: {self.status.reason}")
        import cognee

        joined = "\n".join(
            f"{(m.get('role') or 'user').upper()}: {(m.get('content') or '').strip()}"
            for m in conv.messages
            if (m.get("content") or "").strip()
        )
        if not joined:
            return
        # Tag the joined doc with the conv_id so chunk-level retrievals
        # carry the gold ID inline. The substring cache keeps the
        # fallback path alive for chunkings that drop the prefix.
        tagged = _tag_with_bench_id(conv.id, joined)
        self._content_to_id[joined] = conv.id
        for m in conv.messages:
            content = (m.get("content") or "").strip()
            if content:
                self._content_to_id[content] = conv.id

        async def _ingest():
            await self._add_text(cognee, tagged)
            try:
                await cognee.cognify(datasets=[self._dataset_name])
            except TypeError:
                await cognee.cognify()

        try:
            self._run(_ingest())
        except Exception as e:
            self.status = AdapterStatus(
                ok=False, reason=f"cognee.cognify failed: {e!r}",
            )

    def query(self, q: BenchQuery, top_k: int = 10) -> list[str]:
        if not self.status.ok:
            return []
        import cognee
        from cognee.modules.search.types.SearchType import SearchType

        # CHUNKS returns the raw ``DocumentChunk`` text in vector-rank
        # order — exactly what we need to translate back to bench IDs.
        # The default GRAPH_COMPLETION / RAG_COMPLETION modes return a
        # single LLM-composed answer instead, which destroys the rank
        # signal a retrieval bench depends on.
        async def _search():
            return await cognee.search(
                query_text=q.query,
                query_type=SearchType.CHUNKS,
                datasets=[self._dataset_name],
                top_k=max(top_k, 5),
            )

        try:
            results = self._run(_search())
        except Exception as e:
            log.warning("cognee search failed for %r: %s", q.query, e)
            return []
        texts = self._flatten_search_result(results)

        # First pass: pull the bench_id directly from the chunk prefix.
        # This is unambiguous (one ID per chunk) and preserves rank.
        seen: set[str] = set()
        out: list[str] = []
        for txt in texts:
            mid = _extract_bench_id(txt)
            if mid and mid not in seen:
                out.append(mid)
                seen.add(mid)
                if len(out) >= top_k:
                    return out

        # Fallback: substring match against the content cache. Only
        # fires when chunking dropped the marker — rare on the bench
        # corpus, but keeps the adapter robust on larger payloads.
        for txt in texts:
            for content, mid in self._content_to_id.items():
                if mid in seen:
                    continue
                # Use both directions so partial matches still count.
                if content in txt or txt in content:
                    out.append(mid)
                    seen.add(mid)
                    if len(out) >= top_k:
                        return out
        return out[:top_k]

    def shutdown(self) -> None:
        try:
            import cognee
            # Best-effort: prune if cognee exposes it. Otherwise no-op.
            prune = getattr(cognee, "prune", None)
            if prune is not None:
                async def _p():
                    try:
                        await prune.prune_data()
                        await prune.prune_system(metadata=True)
                    except Exception as e:
                        log.info("cognee prune skipped: %s", e)
                try:
                    self._run(_p())
                except Exception:
                    pass
        except Exception:
            pass


__all__ = ["CogneeAdapter"]
