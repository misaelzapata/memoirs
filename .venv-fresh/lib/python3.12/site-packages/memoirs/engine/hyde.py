"""Layer 5b — HyDE / query rewriting (P2-3).

Hypothetical Document Embeddings (Gao et al., 2022) ask the LLM to imagine
a plausible answer to the query, then embed THAT instead of (or in addition
to) the literal question. This lifts recall on queries whose phrasing
diverges from the corpus, e.g. ``"how do I install memoirs?"`` vs the
README line ``"run pip install -e '.[all]' to set up memoirs locally"``.

Backends
--------
* ``gemma``    — uses the local Gemma model already loaded by the
  consolidation pipeline. Cost: one ~80-token completion.
* ``keyword``  — heuristic, no LLM. Tokenizes the query, drops
  stop-words / short tokens, returns up to 5 keyword expansions. Free.
* ``auto``     — try ``gemma``, fall back to ``keyword`` if Gemma is
  unavailable or errors. Default backend.

Activation
----------
``MEMOIRS_HYDE`` ∈ {``on``, ``off``} (default ``off``) gates the pipeline
hook in ``assemble_context_stream``. ``MEMOIRS_HYDE_BACKEND`` overrides
the backend choice if set.

The expanded query string returned in :class:`ExpandedQuery.combined` is
``original + ". " + hypothetical_doc + " " + " ".join(keywords)``, with
empty pieces stripped — safe to feed straight back into BM25 + dense.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Iterable

log = logging.getLogger("memoirs.hyde")


ENV_HYDE = "MEMOIRS_HYDE"
ENV_BACKEND = "MEMOIRS_HYDE_BACKEND"
HYDE_PROMPT_TEMPLATE = (
    "Write a brief plausible answer to this question (1-2 sentences): {query}"
)
HYDE_MAX_TOKENS = 80
HYDE_MAX_KEYWORDS = 5


# Compact stop-word set — matches the spirit of `embeddings._TRIVIAL_STOPWORDS`
# but kept local so this module stays import-light.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "up", "down",
    "and", "or", "but", "not", "no", "if", "then", "else", "when", "where",
    "why", "how", "what", "who", "which", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their",
    "do", "does", "did", "have", "has", "had", "can", "could", "should",
    "would", "will", "shall", "may", "might", "must", "about", "as", "into",
    "than", "so", "such", "very", "just", "also", "any", "some", "all",
})


@dataclass
class ExpandedQuery:
    original: str
    hypothetical_doc: str = ""
    keywords: list[str] = field(default_factory=list)
    combined: str = ""
    backend: str = "keyword"

    def is_empty(self) -> bool:
        """True if the expansion added no signal."""
        return not self.hypothetical_doc and not self.keywords


# ---------------------------------------------------------------------------
# Activation gate
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    """Return True if HyDE expansion should run for this process."""
    raw = (os.environ.get(ENV_HYDE) or "off").strip().lower()
    return raw in {"on", "1", "true", "yes"}


def _resolve_backend(explicit: str | None) -> str:
    raw = (explicit or os.environ.get(ENV_BACKEND) or "auto").strip().lower()
    if raw not in {"auto", "gemma", "keyword"}:
        log.warning("unknown HyDE backend %r — using 'auto'", raw)
        return "auto"
    return raw


# ---------------------------------------------------------------------------
# Keyword backend (free, deterministic)
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+")


def _extract_keywords(query: str, *, limit: int = HYDE_MAX_KEYWORDS) -> list[str]:
    """Heuristic keyword extraction. spaCy-free, deterministic, ~µs.

    Rules:
      * tokens ≥ 3 chars, alphanumeric (with internal dash/underscore allowed)
      * not in stop-word list
      * not already lowercase-duplicated
      * preserve first-seen order for stability across runs
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in _TOKEN_RE.findall(query or ""):
        tok = raw.lower()
        if len(tok) < 3:
            continue
        if tok in _STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Curator-LLM backend (Qwen / Phi / Gemma — auto-detected)
# ---------------------------------------------------------------------------


def _curator_hypothetical(query: str) -> str:
    """Ask the curator LLM for a brief hypothetical answer. Returns ``""`` on any failure."""
    # Local import: keeps `hyde.py` importable when llama_cpp / the curator
    # GGUF are unavailable. Use getattr defensively so renames in the curator
    # module don't break us.
    try:
        from . import curator as _c
    except Exception as e:  # pragma: no cover - defensive
        log.debug("hyde curator: import failed: %s", e)
        return ""
    have = getattr(_c, "_have_curator", None)
    get_llm = getattr(_c, "_get_llm", None)
    if have is None or get_llm is None or not have():
        return ""
    try:
        llm = get_llm()
    except Exception as e:
        log.debug("hyde curator: load failed: %s", e)
        return ""
    body = HYDE_PROMPT_TEMPLATE.format(query=query[:300])
    prompt = (
        "<start_of_turn>user\n"
        + body
        + "<end_of_turn>\n<start_of_turn>model\n"
    )
    try:
        out = llm.create_completion(
            prompt=prompt,
            max_tokens=HYDE_MAX_TOKENS,
            temperature=0.3,
            stop=["<end_of_turn>", "\n\n"],
        )
        text = (out["choices"][0]["text"] or "").strip()
    except Exception as e:
        log.debug("hyde curator: completion failed: %s", e)
        return ""
    # Cap length defensively in case the model ignores max_tokens.
    return text[:600]


# Backward-compat alias for tests / callers that patched the old name.
_gemma_hypothetical = _curator_hypothetical


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def expand_query(query: str, *, backend: str = "auto") -> ExpandedQuery:
    """Expand a retrieval query.

    Parameters
    ----------
    query:
        The original user query. Empty / whitespace-only inputs round-trip
        unchanged (no LLM call, no keyword extraction).
    backend:
        ``"auto" | "gemma" | "keyword"``. ``auto`` tries ``gemma`` first
        and falls back to ``keyword`` on any failure (no Gemma installed,
        empty completion, exception, …).

    Returns
    -------
    ExpandedQuery
        Always returns; ``combined`` is at minimum the original query.
    """
    q = (query or "").strip()
    if not q:
        return ExpandedQuery(original=query or "", combined=query or "", backend="keyword")

    chosen = _resolve_backend(backend)
    hypothetical = ""
    used = "keyword"
    if chosen in {"gemma", "auto"}:
        hypothetical = _gemma_hypothetical(q)
        if hypothetical:
            used = "gemma"
    keywords = _extract_keywords(q)

    parts: list[str] = [q]
    if hypothetical:
        parts.append(hypothetical)
    if keywords:
        parts.append(" ".join(keywords))
    combined = ". ".join(p.rstrip(".").strip() for p in parts if p)

    return ExpandedQuery(
        original=q,
        hypothetical_doc=hypothetical,
        keywords=keywords,
        combined=combined,
        backend=used,
    )
